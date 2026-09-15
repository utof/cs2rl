"""Tests for the Modal CLIENT modules -- scripts/run_modal.py,
scripts/modal_artifacts.py and scripts/modal_backfill_sidecar.py.

Split out of tests/test_modal_runner.py, which held two suites: the runner
library's tests (which stayed) and these 54. The 54 and the runner half's 138 are
live counts of module-level test functions, re-derivable from either file's AST.

PROVENANCE OF THE OTHER THREE FIGURES, stated because re-running them today
proves less than it looks like it does. "0 of the 137 runner-half tests reach the
client modules", "0 of these 54 are pure-runner" and "0 test bodies span the
boundary" are TASK 3 measurements, taken against the spec's line-3638 seam, which
no longer exists. Do not read them as live. In particular the first is now
definitionally true of the instrument rather than evidence about the split: under
classify_seam a test that reaches a client module IS a client-half test, so the
count cannot come out non-zero however wrong the seam is. A claim only its own
instrument can confirm is not a check. What DOES still refute a bad split is the
placement gate in tests/test_modal_packaging.py, which recomputes each name's
concern from the reference graph and compares it against where the name sits.

The seam is not a line number. tests/test_modal_packaging.py::classify_seam
recomputes every name's destination from the reference graph, and
test_the_modal_test_split_matches_concern_recomputed_from_source asserts this
file holds exactly the names that computation assigns to it.
"""
import importlib
import io
import json
import subprocess
import sys
from enum import IntEnum
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner_lib
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner_lib as mrl                                                 # noqa: E402, I001
from tests.modal_test_helpers import (                                                 # noqa: E402
    FakeChild, _aware, _git, _init_source_repo, _noop_heartbeat, _write_dumped_config,
    _write_metrics)

# ── Launch app: import-time purity, image build, object declarations ───────

PINNED_CUDA_CHILD_DIGEST = (
    "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7")
PINNED_CUDA_IMAGE = ("nvidia/cuda:12.8.1-devel-ubuntu22.04@" + PINNED_CUDA_CHILD_DIGEST)
PINNED_PUFFERLIB_SDIST = (
    "https://files.pythonhosted.org/packages/7c/e1/5292f9b69c6263707b40ba04a87e6b9bcc177281d31092f77afd90c412f1/"
    "pufferlib-3.0.0.tar.gz#sha256=7df3a3e3f5f894d78d2a1f5374097890aec01473183e748abefe4f3faa10eaa9"
)


class FakeModal:
    """In-process stand-in for the Modal SDK. Importing the app must not call it."""

    def __init__(self):
        self.__version__ = "1.4.3"
        self.apps = []
        self.images = []
        self.volume_creates = []
        self.dict_creates = []
        self.volume_lookups = []
        self.dict_lookups = []
        self.secret_lookups = []
        self.base_remote_calls = []
        self.configured_remote_calls = []
        self.configured_spawn_calls = []
        self.with_options_calls = []
        self.batch_upload_calls = []
        self.read_file_calls = []
        self.iterdir_calls = []
        self.volumes = {}
        self.dicts = {}
        self.known_secrets = set()
        self.invoke_remote = False
        self.Image = FakeImage
        self.Image._fake = self
        self.App = self._app_type()
        self.Volume = self._volume_type()
        self.Dict = self._dict_type()
        self.Secret = self._secret_type()

    def as_module(self) -> SimpleNamespace:
        return SimpleNamespace(
            Image=self.Image,
            App=self.App,
            Volume=self.Volume,
            Dict=self.Dict,
            Secret=self.Secret,
            exception=SimpleNamespace(NotFoundError=FakeNotFoundError),
            NotFoundError=FakeNotFoundError,
            __version__=self.__version__,
        )

    def _app_type(self):
        fake = self

        class App:

            def __init__(self, name: str, include_source=None, **kwargs):
                del kwargs
                self.name = name
                self.include_source = include_source
                self.app_id = "ap-ephemeral-test"
                self.functions: dict[str, object] = {}
                self.entrypoints: dict[str, object] = {}
                fake.apps.append(self)

            def function(self, **kwargs):

                def decorator(fn):
                    bound = FakeFunction(fake, fn, kwargs)
                    self.functions[fn.__name__] = bound
                    return bound

                return decorator

            def local_entrypoint(self, *args, **kwargs):
                del args, kwargs

                def decorator(fn):
                    self.entrypoints[fn.__name__] = fn
                    return fn

                return decorator

        return App

    def _volume_type(self):
        fake = self

        class Volume:

            class objects:

                @staticmethod
                def create(name: str, allow_existing: bool = False, **kwargs):
                    del kwargs
                    fake.volume_creates.append((name, allow_existing))
                    if name in fake.volumes and not allow_existing:
                        raise FileExistsError(name)
                    fake.volumes.setdefault(name, FakeVolume(fake, name))

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False):
                fake.volume_lookups.append((name, create_if_missing))
                if name not in fake.volumes:
                    if create_if_missing:
                        fake.volumes[name] = FakeVolume(fake, name)
                    else:
                        raise FakeNotFoundError(f"Volume {name!r} not found")
                return fake.volumes[name]

        return Volume

    def _dict_type(self):
        fake = self

        class Dict:

            class objects:

                @staticmethod
                def create(name: str, allow_existing: bool = False, **kwargs):
                    del kwargs
                    fake.dict_creates.append((name, allow_existing))
                    if name in fake.dicts and not allow_existing:
                        raise FileExistsError(name)
                    fake.dicts.setdefault(name, FakeDict(fake, name))

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False):
                fake.dict_lookups.append((name, create_if_missing))
                if name not in fake.dicts:
                    if create_if_missing:
                        fake.dicts[name] = FakeDict(fake, name)
                    else:
                        raise FakeNotFoundError(f"Dict {name!r} not found")
                return fake.dicts[name]

        return Dict

    def _secret_type(self):
        fake = self

        class Secret:

            def __init__(self, name: str):
                self.name = name

            def __repr__(self) -> str:
                return "Secret(<redacted>)"

            @staticmethod
            def from_name(name: str, **kwargs):
                del kwargs
                fake.secret_lookups.append(name)
                if name not in fake.known_secrets:
                    raise FakeNotFoundError("requested W&B Secret is missing")
                return Secret(name)

        return Secret


class FakeNotFoundError(Exception):
    """Stand-in for a missing named Modal object."""


class FakeImage:
    _fake: FakeModal | None = None

    def __init__(self):
        self.registry_tag: str | None = None
        self.add_python: str | None = None
        self.apt: list[str] = []
        self.pips: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.local_files: list[tuple[str, str, bool]] = []
        self.commands: list[str] = []

    @classmethod
    def from_registry(cls, tag: str, add_python: str | None = None, **kwargs):
        del kwargs
        image = cls()
        image.registry_tag = tag
        image.add_python = add_python
        if cls._fake is not None:
            cls._fake.images.append(image)
        return image

    def apt_install(self, *packages: str):
        self.apt.extend(packages)
        return self

    def pip_install(self, *packages: str):
        self.pips.extend(packages)
        return self

    def env(self, mapping: dict[str, str]):
        self.env_vars.update(mapping)
        return self

    def add_local_file(self, src: str, dst: str, copy: bool = False):
        self.local_files.append((src, dst, copy))
        return self

    def run_commands(self, *commands: str):
        self.commands.extend(commands)
        return self


class FakeFunction:
    """Decorated Function: base .remote is forbidden; with_options is the only path."""

    def __init__(self, fake: FakeModal, fn, kwargs: dict[str, object]):
        self._fake = fake
        self._fn = fn
        self.kwargs = kwargs
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        self._fake.base_remote_calls.append((args, kwargs))
        raise AssertionError("base Function must never be called")

    def spawn(self, *args, **kwargs):
        raise AssertionError("base Function must never be called")

    def with_options(self, **kwargs):
        self._fake.with_options_calls.append(dict(kwargs))
        return FakeConfiguredFunction(self._fake, self, kwargs)


class FakeConfiguredFunction:

    def __init__(self, fake: FakeModal, base: FakeFunction, options: dict[str, object]):
        self._fake = fake
        self.base = base
        self.options = options

    def remote(self, *args, **kwargs):
        self._fake.configured_remote_calls.append((self.options, args, kwargs))
        if self._fake.invoke_remote:
            return self.base._fn(*args, **kwargs)
        return {"status": "ok"}

    def spawn(self, *args, **kwargs):
        """ASYNC invocation. Does not wait; returns a FunctionCall-shaped handle.

        invoke_remote still runs the wrapper in-process so launch_run tests that
        need train_remote side effects keep working. A real Modal spawn would
        schedule the container and return immediately.
        """
        self._fake.configured_spawn_calls.append((self.options, args, kwargs))
        if self._fake.invoke_remote:
            self.base._fn(*args, **kwargs)
        return SimpleNamespace(object_id="fc-test")


class FakeVolume:

    def __init__(self, fake: FakeModal, name: str):
        self._fake = fake
        self.name = name
        self.files: dict[str, bytes] = {}
        self.pending_creates: dict[str, bytes] = {}
        self.replace_after_read: dict[str, bytes] = {}
        self.commit_count = 0
        self.reject_next_upload = False
        self.iterdir_entries: list[object] | None = None
        self.missing_prefix_exc: type[BaseException] | None = None
        self.fail_prefix: str | None = None

    def _client_path(self, path) -> str:
        text = str(path)
        if text.startswith("/artifacts"):
            raise AssertionError(f"/artifacts leaked to Volume client API: {text}")
        return text

    def batch_upload(self, force: bool = False):
        return FakeBatchUpload(self, force)

    def read_file(self, path):
        key = self._client_path(path)
        self._fake.read_file_calls.append(key)
        if key not in self.files:
            raise FileNotFoundError(key)
        data = self.files[key]
        if key in self.replace_after_read:
            self.files[key] = self.replace_after_read.pop(key)
        yield data

    def iterdir(self, path, *, recursive: bool = True):
        key = self._client_path(path)
        self._fake.iterdir_calls.append((key, recursive))
        if self.iterdir_entries is not None:
            yield from self.iterdir_entries
            return
        prefix = key.rstrip("/")
        matched = False
        for stored in sorted(self.files):
            if prefix == "" or stored == prefix or stored.startswith(prefix + "/"):
                matched = True
                yield SimpleNamespace(path=stored, type="file")
        if not matched and self.missing_prefix_exc is not None:
            raise self.missing_prefix_exc(key)

    def commit(self):
        self.commit_count += 1

    def reload(self):
        return None


class FakeBatchUpload:

    def __init__(self, volume: FakeVolume, force: bool):
        self.volume = volume
        self.force = force
        self.puts: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_file(self, local_path, remote_path):
        key = self.volume._client_path(remote_path)
        self.volume._fake.batch_upload_calls.append((key, self.force, str(local_path)))
        if self.force:
            raise AssertionError("force=True is never used")
        if key in self.volume.pending_creates:
            self.volume.files[key] = self.volume.pending_creates.pop(key)
            raise FileExistsError(key)
        if self.volume.reject_next_upload:
            self.volume.reject_next_upload = False
            raise FileExistsError(key)
        if self.volume.fail_prefix is not None and key.startswith(self.volume.fail_prefix):
            raise OSError(f"could not upload {key}")
        if key in self.volume.files:
            raise FileExistsError(key)
        data = Path(local_path).read_bytes() if not hasattr(local_path,
                                                            "read") else local_path.read()
        self.volume.files[key] = data
        self.puts.append((str(local_path), key))


class FakeDict:

    def __init__(self, fake: FakeModal, name: str):
        self._fake = fake
        self.name = name
        self.data: dict[str, object] = {}

    def put(self, key: str, value, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key: str):
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]


@pytest.fixture
def fake_modal():
    fake = FakeModal()
    previous = sys.modules.get("modal")
    sys.modules["modal"] = fake.as_module()
    for name in ("scripts.run_modal", "scripts.modal_artifacts", "scripts.modal_backfill_sidecar"):
        sys.modules.pop(name, None)
    try:
        yield fake
    finally:
        for name in ("scripts.run_modal", "scripts.modal_artifacts",
                     "scripts.modal_backfill_sidecar"):
            sys.modules.pop(name, None)
        if previous is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous


def _import_run_modal():
    return importlib.import_module("scripts.run_modal")


def test_importing_app_creates_no_function_call_or_gpu_work(fake_modal):
    module = _import_run_modal()
    assert fake_modal.base_remote_calls == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.with_options_calls == []
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.volume_lookups == []
    assert fake_modal.dict_lookups == []
    assert fake_modal.secret_lookups == []
    assert module.app.name == "cs2rl-training"
    assert "gpu" not in module.train_remote.kwargs or module.train_remote.kwargs["gpu"] is None


def test_base_function_has_no_static_named_object_dependency(fake_modal):
    module = _import_run_modal()
    kwargs = module.train_remote.kwargs
    assert kwargs.get("volumes") in (None, {})
    assert "volumes" not in kwargs or not kwargs["volumes"]
    assert kwargs.get("secrets") in (None, [])
    assert kwargs.get("retries") == 0
    assert kwargs.get("single_use_containers") is True
    assert module.app.include_source is False
    assert kwargs.get("include_source") is False
    assert module.app.name == "cs2rl-training"
    assert "main" in module.app.entrypoints
    assert mrl.VOLUME_NAME == "cs2rl-training-artifacts"
    assert mrl.REGISTRY_NAME == "cs2rl-training-run-registry"


def _run_modal_image_reqs(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "modal_image_reqs.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _requirement_names(text: str) -> list[str]:
    pins = [line for line in text.splitlines() if line.strip()]
    assert pins
    assert all("==" in line for line in pins)
    return [line.split("==", 1)[0] for line in pins]


def test_modal_image_reqs_from_repo_lock_includes_torch_numpy_not_pufferlib_or_modal():
    result = _run_modal_image_reqs(str(ROOT / "uv.lock"))
    names = _requirement_names(result.stdout)
    assert names == sorted(names)
    assert "torch" in names
    assert "numpy" in names
    assert "pufferlib" not in names
    assert "cs2rl" not in names
    assert "modal" not in names


def test_modal_image_reqs_writes_minus_o(tmp_path):
    out = tmp_path / "cs2rl-reqs.txt"
    result = _run_modal_image_reqs(str(ROOT / "uv.lock"), "-o", str(out))
    assert result.stdout == ""
    names = _requirement_names(out.read_text())
    assert names == sorted(names)
    assert "torch" in names
    assert "numpy" in names
    assert "pufferlib" not in names
    assert "modal" not in names


def test_modal_image_reqs_walks_runtime_graph_not_dev_or_modal_groups(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("""\
version = 1
[[package]]
name = "cs2rl"
version = "0.1.0"
dependencies = [
    { name = "numpy" },
    { name = "pufferlib" },
]

[package.dev-dependencies]
dev = [
    { name = "ruff" },
]
modal = [
    { name = "modal" },
]

[[package]]
name = "numpy"
version = "2.4.3"

[[package]]
name = "pufferlib"
version = "3.0.0"
dependencies = [
    { name = "torch" },
    { name = "shimmy", extra = ["gym-v21"] },
]

[[package]]
name = "torch"
version = "2.10.0"

[[package]]
name = "shimmy"
version = "1.3.0"

[package.optional-dependencies]
gym-v21 = [
    { name = "pyglet" },
]

[[package]]
name = "pyglet"
version = "2.1.0"

[[package]]
name = "modal"
version = "1.5.4"

[[package]]
name = "ruff"
version = "0.11.13"
""")
    result = _run_modal_image_reqs(str(lock))
    assert result.stdout == "numpy==2.4.3\npyglet==2.1.0\nshimmy==1.3.0\ntorch==2.10.0\n"


def test_image_pins_cuda_digest_arch_list_and_hashed_pufferlib_sdist(fake_modal):
    module = _import_run_modal()
    image = module.dependency_image
    assert image.registry_tag == PINNED_CUDA_IMAGE
    assert image.add_python == "3.12"
    assert image.env_vars["TORCH_CUDA_ARCH_LIST"] == "7.5;8.6;8.9"
    assert image.env_vars["NO_OCEAN"] == "1"
    assert "uv==0.11.1" in image.pips
    assert "ziglang==0.14.1" in image.pips
    assert (str(ROOT / "pyproject.toml"), "/opt/cs2rl/pyproject.toml", True) in image.local_files
    assert (str(ROOT / "uv.lock"), "/opt/cs2rl/uv.lock", True) in image.local_files
    assert (str(ROOT / "scripts" / "modal_image_reqs.py"), "/opt/cs2rl/modal_image_reqs.py",
            True) in image.local_files
    commands = "\n".join(image.commands)
    assert "modal_image_reqs.py" in commands
    assert "uv pip install" in commands
    assert "-r" in commands
    locked_dep_installs = [
        part.strip() for command in image.commands for part in command.split("&&")
        if "uv pip install" in part and "-r" in part
    ]
    assert locked_dep_installs
    assert all("--directory /tmp" in cmd for cmd in locked_dep_installs)
    locked_dep_command = " && ".join(locked_dep_installs)
    assert "/tmp/cs2rl-reqs.txt" in locked_dep_command
    assert "--no-deps" in locked_dep_command
    assert "pufferlib" not in locked_dep_command
    assert "uv export" not in locked_dep_command
    assert "uv sync" not in locked_dep_command
    assert "uv export" not in commands
    assert "uv sync" not in commands
    assert "--no-build-isolation" in commands
    assert "--no-deps" in commands
    assert "--no-binary pufferlib" in commands
    assert PINNED_PUFFERLIB_SDIST in commands
    assert "c_extension_paths = []" in commands
    hashed_sdist_installs = [
        part.strip() for command in image.commands for part in command.split("&&")
        if "uv pip install" in part and "--no-binary pufferlib" in part
    ]
    assert hashed_sdist_installs
    assert all("pufferlib" in cmd for cmd in hashed_sdist_installs)
    assert all("/tmp/pufferlib-3.0.0" in cmd for cmd in hashed_sdist_installs)
    assert all("CXX=g++" in cmd for cmd in hashed_sdist_installs)
    assert "Python.h" in commands
    assert "release 12.8" in commands
    assert "pufferlib._C" in commands
    assert "compute_puff_advantage" in commands
    assert "all('sm_'+arch in elf for arch in ('75','86','89'))" in commands
    runner = module.runner_image
    assert (str(ROOT / "scripts" / "modal_runner_lib.py"), "/opt/app/scripts/modal_runner_lib.py",
            True) in runner.local_files
    assert (str(ROOT / "scripts" / "run_modal.py"), "/opt/app/scripts/run_modal.py",
            True) in runner.local_files
    # include_source=False: Modal imports module_name "run_modal", while
    # run_modal.py does "import scripts.modal_runner_lib". Both path entries
    # are required; /opt/app alone raises ModuleNotFoundError: run_modal.
    assert runner.env_vars["PYTHONPATH"] == "/opt/app:/opt/app/scripts"
    for src, _dst, _copy in (*image.local_files, *runner.local_files):
        assert Path(src).is_absolute()
        assert Path(src).is_relative_to(ROOT)


# ── Launch CLI: run-only parser, omitted sentinels, defaults ───────────────


def _launch_sentinels(**overrides):
    """Every launch option starts as None; callers supply only explicit values."""
    kwargs = {
        "action": None,
        "run_id": None,
        "git_sha": None,
        "map": None,
        "gpu": None,
        "cpu_cores": None,
        "memory_mib": None,
        "num_envs": None,
        "vec_workers": None,
        "timeout_minutes": None,
        "save_every_seconds": None,
        "train_args": None,
        "resume_local_checkpoint": None,
        "resume_run_id": None,
        "wandb_secret_name": None,
    }
    kwargs.update(overrides)
    return kwargs


def _valid_launch_sentinels(**overrides):
    kwargs = _launch_sentinels(
        action="run",
        run_id="140826-b7r-seed2-shared",
        git_sha="a" * 40,
        map="simple",
        train_args="--timesteps 30000000 --seed 2",
    )
    kwargs.update(overrides)
    return kwargs


def test_status_and_download_are_not_app_actions(fake_modal):
    module = _import_run_modal()
    assert list(module.app.entrypoints) == ["main"]
    for action in ("status", "download"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(action=action))
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(action=None))
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(action="train"))


def test_omitted_gpu_defaults_to_t4_invalid_explicit_gpu_rejected(fake_modal):
    module = _import_run_modal()
    request = module.resolve_launch_request(**_valid_launch_sentinels(gpu=None))
    assert request.gpu == mrl.DEFAULT_GPU == "T4"
    for gpu in ("T4", "L4", "A10"):
        assert module.resolve_launch_request(**_valid_launch_sentinels(gpu=gpu)).gpu == gpu
    for gpu in ("A10G", "A100", "H100", "any", "T4,L4", "t4", "T4:2", "T4;L4"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(gpu=gpu))


def test_map_has_no_default_and_must_be_allowlisted(fake_modal):
    module = _import_run_modal()
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(map=None))
    for effective_map in ("simple", "dust2"):
        request = module.resolve_launch_request(**_valid_launch_sentinels(map=effective_map))
        assert request.effective_map == effective_map
    for effective_map in ("", "cs2-dust2", "DUST2", "dust"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(map=effective_map))


def test_omitted_resource_sentinels_apply_defaults_and_smoke_values_pass(fake_modal):
    module = _import_run_modal()
    request = module.resolve_launch_request(**_valid_launch_sentinels())
    assert request.cpu_cores == 8
    assert request.memory_mib == 16384
    assert request.cpu_request_limit == (8, 8)
    assert request.memory_request_limit == (16384, 16384)
    assert request.num_envs == 256
    assert request.vec_workers == 8
    assert request.timeout_minutes == 120
    assert request.save_every_seconds == 300
    smoke = module.resolve_launch_request(**_valid_launch_sentinels(
        cpu_cores=4,
        memory_mib=8192,
        vec_workers=4,
        timeout_minutes=15,
        train_args="--timesteps 163840 --seed 2",
    ))
    assert smoke.cpu_request_limit == (4, 4)
    assert smoke.memory_request_limit == (8192, 8192)
    assert smoke.vec_workers == 4
    assert smoke.timeout_minutes == 15


# ── Modal adapters: Volume namespace + reservation/blob adapters ───────────


def _named_volume(fake_modal, name=mrl.VOLUME_NAME):
    fake_modal.Volume.objects.create(name, allow_existing=True)
    return fake_modal.Volume.from_name(name, create_if_missing=False)


def _named_dict(fake_modal, name=mrl.REGISTRY_NAME):
    fake_modal.Dict.objects.create(name, allow_existing=True)
    return fake_modal.Dict.from_name(name, create_if_missing=False)


def test_volume_adapter_uses_root_relative_client_paths(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    artifacts = module.ModalVolumeIndex(volume)
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    artifacts.put_file(reservation, b'{"attempt_id":"a"}\n')
    artifacts.commit()
    assert artifacts.exists(reservation)
    assert reservation.as_posix() in volume.files
    assert all(not path.startswith("/artifacts") for path in volume.files)
    assert all(not remote.startswith("/artifacts")
               for remote, _force, _local in fake_modal.batch_upload_calls)
    source_client = mrl.SOURCES_ROOT / "deadbeef.tar.gz"
    assert mrl.mounted_path(source_client) == Path("/artifacts/sources/deadbeef.tar.gz")
    ckpt_client = mrl.INPUTS_ROOT / "sha256" / "abcd.pt"
    assert mrl.mounted_path(ckpt_client) == Path("/artifacts/inputs/sha256/abcd.pt")


def test_modal_volume_index_read_file(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    path = mrl.RUNS_ROOT / "ok-id" / mrl.STATUS_FILENAME
    volume.files[path.as_posix()] = b'{"ok": true}'
    index = module.ModalVolumeIndex(volume)
    assert index.read_file(path) == b'{"ok": true}'
    assert index.read_file(mrl.RUNS_ROOT / "missing-id" / mrl.STATUS_FILENAME) is None
    with pytest.raises(mrl.ValidationError, match="refusing Volume client path"):
        index.read_file(PurePosixPath("/artifacts/runs/ok-id/STATUS.json"))


def test_volume_adapter_commit_does_not_call_client_volume_commit(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    commit_calls = {"count": 0}

    def raising_commit():
        commit_calls["count"] += 1
        raise RuntimeError("commit() can only be called on a mounted volume inside a container")

    volume.commit = raising_commit
    artifacts = module.ModalVolumeIndex(volume)
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    artifacts.put_file(reservation, b'{"attempt_id":"a"}\n')
    artifacts.commit()
    assert artifacts.exists(reservation)
    assert volume.files[reservation.as_posix()] == b'{"attempt_id":"a"}\n'
    assert artifacts._staged == []
    assert all(force is False for _path, force, _local in fake_modal.batch_upload_calls)
    assert all(not remote.startswith("/artifacts")
               for remote, _force, _local in fake_modal.batch_upload_calls)
    assert commit_calls["count"] == 0


def test_ensure_blob_uploads_missing_and_reuses_after_streamed_verify(fake_modal, tmp_path):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    blob = tmp_path / "src.tar.gz"
    blob.write_bytes(b"source-bytes")
    digest = mrl.sha256_file(blob)
    client_path = mrl.SOURCES_ROOT / f"{digest}.tar.gz"
    module.ensure_blob(volume, client_path, blob)
    assert fake_modal.batch_upload_calls == [(client_path.as_posix(), False, str(blob))]
    assert volume.files[client_path.as_posix()] == b"source-bytes"
    assert mrl.mounted_path(client_path) == Path("/artifacts/sources") / f"{digest}.tar.gz"

    fake_modal.batch_upload_calls.clear()
    module.ensure_blob(volume, client_path, blob)
    assert fake_modal.batch_upload_calls == []
    assert fake_modal.read_file_calls[-1] == client_path.as_posix()


def test_ensure_blob_handles_concurrent_create_and_rejects_mismatch(fake_modal, tmp_path):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    blob = tmp_path / "warm.pt"
    blob.write_bytes(b"ckpt-bytes")
    digest = mrl.sha256_file(blob)
    client_path = mrl.INPUTS_ROOT / "sha256" / f"{digest}.pt"
    volume.pending_creates[client_path.as_posix()] = b"ckpt-bytes"
    module.ensure_blob(volume, client_path, blob)
    assert mrl.mounted_path(client_path) == Path("/artifacts/inputs/sha256") / f"{digest}.pt"
    volume.files[client_path.as_posix()] = b"other-bytes"
    with pytest.raises(mrl.ValidationError):
        module.ensure_blob(volume, client_path, blob)
    assert all(force is False for _path, force, _local in fake_modal.batch_upload_calls)


def test_reserve_run_through_modal_adapters_stays_in_client_namespace(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    registry = module.ModalDictRegistry(_named_dict(fake_modal))
    artifacts = module.ModalVolumeIndex(volume)
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    assert reservation.as_posix() in volume.files
    assert all(not path.startswith("/artifacts") for path in volume.files)
    assert registry.get(mrl.run_registry_key("ok-id"))["attempt_id"] == "attempt-a"
    assert fake_modal.volume_creates == [(mrl.VOLUME_NAME, True)]
    assert fake_modal.dict_creates == [(mrl.REGISTRY_NAME, True)]
    assert fake_modal.volume_lookups == [(mrl.VOLUME_NAME, False)]
    assert fake_modal.dict_lookups == [(mrl.REGISTRY_NAME, False)]


# ── Launch invocation: configured run, W&B gating, payload, train_remote ───


def _capture_stdout():
    return io.StringIO()


def _write_parent_artifacts(volume, parent_id, *, status, updated_at, ckpt_bytes, sidecar):
    status_path = (mrl.RUNS_ROOT / parent_id / mrl.STATUS_FILENAME).as_posix()
    ckpt_path = (mrl.RUNS_ROOT / parent_id / "checkpoints" / mrl.CHECKPOINT_NAME).as_posix()
    sidecar_path = (mrl.RUNS_ROOT / parent_id / "checkpoints" /
                    mrl.CHECKPOINT_SIDECAR_NAME).as_posix()
    volume.files[status_path] = json.dumps({
        "schema_version": 1,
        "status": status,
        "attempt_id": "parent-attempt",
        "updated_at": updated_at,
    }).encode()
    volume.files[ckpt_path] = ckpt_bytes
    if sidecar is not None:
        volume.files[sidecar_path] = json.dumps(sidecar).encode()
    return ckpt_path, sidecar_path


def test_invalid_inputs_validate_before_claim_or_gpu(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    junk = tmp_path / "nope.pt"
    junk.write_text("not-a-checkpoint")
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(junk)))
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app)
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.base_remote_calls == []


def test_configured_run_uses_with_options_defaults_and_prints_ids(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    digest = mrl.sha256_file(ckpt)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(ckpt)))
    stdout = _capture_stdout()
    result = module.launch_run(request, repo=repo, app_obj=module.app, stdout=stdout)
    assert result["status"] == "spawned"
    assert fake_modal.base_remote_calls == []
    assert fake_modal.configured_remote_calls == []
    assert len(fake_modal.with_options_calls) == 1
    options = fake_modal.with_options_calls[0]
    assert options["gpu"] == "T4"
    assert options["cpu"] == (8, 8)
    assert options["memory"] == (16384, 16384)
    assert options["timeout"] == 120 * 60
    assert list(options["volumes"]) == ["/artifacts"]
    assert "secrets" not in options
    _opts, args, kwargs = fake_modal.configured_spawn_calls[0]
    payload = args[0] if args else kwargs["payload"]
    assert payload["run_id"] == request.run_id
    assert payload["resume_mount_path"] == f"/artifacts/inputs/sha256/{digest}.pt"
    assert payload["resume_sha256"] == digest
    assert "wandb_enabled" not in payload
    assert "wandb_secret_name" not in payload
    dumped = json.dumps(payload)
    assert "wandb" not in dumped
    assert all(
        isinstance(value, (str, int, float, bool, list, type(None))) for value in payload.values())
    printed = stdout.getvalue()
    assert module.app.app_id in printed
    assert request.run_id in printed
    source_name = next(path for path in fake_modal.volumes[mrl.VOLUME_NAME].files
                       if path.startswith("sources/"))
    assert source_name.endswith(".tar.gz")
    assert not source_name.startswith("/artifacts")
    assert f"inputs/sha256/{digest}.pt" in fake_modal.volumes[mrl.VOLUME_NAME].files


def test_smoke_resource_tuples_and_cpu_memory_semantics(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        cpu_cores=4,
        memory_mib=8192,
        vec_workers=4,
        timeout_minutes=15,
        train_args="--timesteps 163840 --seed 2",
    ))
    # CPU tuple is a soft throttling limit; memory tuple is a hard OOM limit.
    assert request.cpu_request_limit == (4, 4)
    assert request.memory_request_limit == (8192, 8192)
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    options = fake_modal.with_options_calls[0]
    assert options["cpu"] == (4, 4)
    assert options["memory"] == (8192, 8192)
    assert options["timeout"] == 15 * 60


def test_wandb_secret_missing_fails_before_claim_without_leaking_name(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    secret_name = "prod-wandb-key"
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        train_args="--timesteps 163840 --seed 2 --wandb",
        wandb_secret_name=secret_name,
    ))
    with pytest.raises(mrl.ValidationError) as excinfo:
        module.launch_run(request, repo=repo, app_obj=module.app)
    assert secret_name not in str(excinfo.value)
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.secret_lookups == [secret_name]


def test_wandb_attaches_secret_and_records_enabled_flag_only(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    secret_name = "prod-wandb-key"
    fake_modal.known_secrets.add(secret_name)
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        train_args="--timesteps 163840 --seed 2 --wandb",
        wandb_secret_name=secret_name,
    ))
    stdout = _capture_stdout()
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=stdout)
    options = fake_modal.with_options_calls[0]
    assert "secrets" in options
    assert len(options["secrets"]) == 1
    payload = fake_modal.configured_spawn_calls[0][1][0]
    assert payload["wandb_enabled"] is True
    assert "wandb_secret_name" not in payload
    assert secret_name not in json.dumps(payload)
    assert secret_name not in stdout.getvalue()


@pytest.mark.parametrize(
    "case",
    ["active", "missing", "stale", "mismatch", "replaced"],
)
def test_prior_run_resume_fails_closed_without_consuming_new_id(fake_modal, tmp_path, case):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "parent.pt"
    torch.save({"weight": torch.tensor([3.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    sidecar = {
        "sha256": digest,
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    status = "completed" if case != "active" else "training"
    _ckpt_path, sidecar_path = _write_parent_artifacts(
        volume,
        "parent-run",
        status=status,
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=None if case == "missing" else sidecar,
    )
    if case == "stale":
        volume.files[sidecar_path] = json.dumps({
            **sidecar,
            "size": len(ckpt_bytes) + 1,
            "mtime_ns": 99,
        }).encode()
    elif case == "mismatch":
        volume.files[sidecar_path] = json.dumps({**sidecar, "sha256": "0" * 64}).encode()
    elif case == "replaced":
        volume.replace_after_read = {
            sidecar_path: json.dumps({
                **sidecar, "sha256": "1" * 64
            }).encode()
        }
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="parent-run"))
    creates_before = list(fake_modal.volume_creates)
    dicts_before = list(fake_modal.dict_creates)
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app, now=_aware())
    assert fake_modal.volume_creates == creates_before
    assert fake_modal.dict_creates == dicts_before
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.dicts.get(mrl.REGISTRY_NAME) is None


def test_prior_run_resume_sends_only_immutable_digest_path(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "parent.pt"
    torch.save({"weight": torch.tensor([4.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    _write_parent_artifacts(
        volume,
        "parent-run",
        status="completed",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar={
            "sha256": digest,
            "size": len(ckpt_bytes),
            "mtime_ns": 1,
            "validated_at": _aware().isoformat(),
        },
    )
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="parent-run"))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_spawn_calls[0][1][0]
    assert payload["resume_mount_path"] == f"/artifacts/inputs/sha256/{digest}.pt"
    assert payload["resume_sha256"] == digest
    assert payload["resumed_from_run_id"] == "parent-run"
    assert module.build_remote_manifest(payload).resumed_from_run_id == "parent-run"
    assert "runs/parent-run" not in payload["resume_mount_path"]
    assert f"inputs/sha256/{digest}.pt" in fake_modal.volumes[mrl.VOLUME_NAME].files
    assert fake_modal.volumes[mrl.VOLUME_NAME].files[f"inputs/sha256/{digest}.pt"] == ckpt_bytes


PROTOCOL_TOKENS = (
    "missing_sidecar",
    "corrupt_sidecar",
    "missing_checkpoint",
    "stale_size",
    "digest_mismatch",
    "not_loadable",
    "replaced",
    "ok",
)


def test_launch_checkpoint_errors_is_total(fake_modal):
    # prior_checkpoint_or_raise indexes _LAUNCH_CHECKPOINT_ERRORS[verdict.reason]
    # directly, so a reason token with no row there escapes launch as a bare
    # KeyError instead of the ValidationError callers handle. Nothing derives
    # that map from the protocol, so pin the two sets against each other.
    #
    # Known gap, measured rather than assumed: PROTOCOL_TOKENS is hand-written
    # too, so adding a real eighth token to verify_checkpoint and touching
    # neither this tuple nor the map leaves this test green. It guards the
    # map-vs-tuple pairing only, not the protocol.
    module = _import_run_modal()
    assert set(module._LAUNCH_CHECKPOINT_ERRORS) == set(PROTOCOL_TOKENS) - {"ok"}


def _install_protocol_parent(volume, tmp_path, run_id, case):
    import torch

    sidecar_path = (mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_SIDECAR_NAME).as_posix()
    ckpt_path = (mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_NAME).as_posix()
    if case == "not_loadable":
        ckpt_bytes = b"torn-bytes"
    else:
        ckpt = tmp_path / f"{run_id}.pt"
        torch.save({"weight": torch.tensor([3.0])}, ckpt)
        ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_bytes(ckpt_bytes)
    sidecar = {
        "sha256": digest,
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    _write_parent_artifacts(
        volume,
        run_id,
        status="completed",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=None if case == "missing_sidecar" else sidecar,
    )
    if case == "corrupt_sidecar":
        volume.files[sidecar_path] = b"{not-json"
    elif case == "missing_checkpoint":
        volume.files.pop(ckpt_path, None)
    elif case == "stale_size":
        volume.files[sidecar_path] = json.dumps({**sidecar, "size": len(ckpt_bytes) + 8}).encode()
    elif case == "digest_mismatch":
        volume.files[sidecar_path] = json.dumps({**sidecar, "sha256": "0" * 64}).encode()
    elif case == "replaced":
        volume.replace_after_read = {
            sidecar_path: json.dumps({
                **sidecar, "sha256": "1" * 64
            }).encode()
        }
    return ckpt_bytes, digest


@pytest.mark.parametrize("case", PROTOCOL_TOKENS)
def test_protocol_tokens_through_collect_status(fake_modal, tmp_path, case):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    _install_protocol_parent(volume, tmp_path, "ok-id", case)
    report = module.collect_status("ok-id", now=_aware())
    assert report["run_id"] == "ok-id"
    assert report["status"] == "completed"
    assert report["checkpoint_loadable"] is (case == "ok")


@pytest.mark.parametrize("case", PROTOCOL_TOKENS)
def test_protocol_tokens_through_launch_run(fake_modal, tmp_path, case):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    ckpt_bytes, digest = _install_protocol_parent(volume, tmp_path, "parent-run", case)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="parent-run"))
    if case == "ok":
        module.launch_run(request,
                          repo=repo,
                          app_obj=module.app,
                          now=_aware(),
                          stdout=_capture_stdout())
        payload = fake_modal.configured_spawn_calls[0][1][0]
        assert payload["resume_sha256"] == digest
        assert fake_modal.volumes[mrl.VOLUME_NAME].files[f"inputs/sha256/{digest}.pt"] == ckpt_bytes
        return
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app, now=_aware())


def test_launch_missing_parent_run_message(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    _named_volume(fake_modal)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="missing-parent"))
    with pytest.raises(mrl.ValidationError, match="parent run was not found"):
        module.launch_run(request, repo=repo, app_obj=module.app, now=_aware())


def _rearm_replaced(volume, run_id):
    sidecar_path = (mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_SIDECAR_NAME).as_posix()
    ckpt_path = (mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_NAME).as_posix()
    ckpt_bytes = volume.files[ckpt_path]
    matching = {
        "sha256": mrl.sha256_bytes(ckpt_bytes),
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    volume.files[sidecar_path] = json.dumps(matching).encode()
    volume.replace_after_read = {
        sidecar_path: json.dumps({
            **matching, "sha256": "1" * 64
        }).encode()
    }


def test_dual_caller_mutation_pin_replaced(fake_modal, tmp_path, monkeypatch):
    arts = _import_artifacts()
    launch = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    real_verify = mrl.verify_checkpoint

    def _ok_skipping_reread(sidecar_bytes, checkpoint_bytes, sidecar_reread_bytes, *, load=None):
        del sidecar_reread_bytes, load
        if sidecar_bytes is None or checkpoint_bytes is None:
            return mrl.CheckpointVerdict(False, "missing_sidecar", None, None)
        digest = mrl.sha256_bytes(checkpoint_bytes)
        return mrl.CheckpointVerdict(True, None, checkpoint_bytes, digest)

    _install_protocol_parent(volume, tmp_path, "parent-run", "replaced")
    monkeypatch.setattr(mrl, "verify_checkpoint", _ok_skipping_reread)
    assert arts.collect_status("parent-run", now=_aware())["checkpoint_loadable"] is True
    _rearm_replaced(volume, "parent-run")
    request = launch.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-patched", resume_run_id="parent-run"))
    launch.launch_run(request,
                      repo=repo,
                      app_obj=launch.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    monkeypatch.setattr(mrl, "verify_checkpoint", real_verify)
    _rearm_replaced(volume, "parent-run")
    assert arts.collect_status("parent-run", now=_aware())["checkpoint_loadable"] is False
    _rearm_replaced(volume, "parent-run")
    request2 = launch.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha, run_id="child-unpatched", resume_run_id="parent-run"))
    with pytest.raises(mrl.ValidationError):
        launch.launch_run(request2, repo=repo, app_obj=launch.app, now=_aware())


def _expected_thread_caps():
    return [f"{key}={value}" for key, value in sorted(mrl._THREAD_CAP_ENV.items())]


def test_launch_payload_includes_design_contract_fields(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    digest = mrl.sha256_file(ckpt)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(ckpt)))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_spawn_calls[0][1][0]
    run_root = mrl.mounted_path(mrl.RUNS_ROOT / request.run_id)
    resume_mount = f"/artifacts/inputs/sha256/{digest}.pt"
    requested = request.timesteps
    batch_size = request.batch_size
    effective = (requested // batch_size) * batch_size
    assert payload["training_argv"] == request.training_argv(run_root, resume_mount)
    assert payload["requested_timesteps"] == requested
    assert payload["effective_timesteps"] == effective
    assert payload["batch_size"] == batch_size
    assert payload["seed"] == 2
    assert payload["created_at"] == _aware().isoformat()
    assert payload["resume_sha256"] == digest
    assert payload["resume_size"] == ckpt.stat().st_size
    assert payload["resume_source_path"] == resume_mount
    assert payload["runner_commit"] == sha
    assert payload["config_hash"] == "0" * 64
    assert payload["modal_version"] == fake_modal.__version__
    assert payload["image_digest"] == PINNED_CUDA_CHILD_DIGEST
    assert payload["effective_map"] == "simple"
    assert payload["gpu"] == "T4"
    assert payload["cpu_request"] == payload["cpu_soft_limit"] == 8
    assert payload["memory_request_mib"] == payload["memory_hard_limit_mib"] == 16384
    assert payload["vec_workers"] == 8
    assert payload["thread_caps"] == _expected_thread_caps()
    assert payload["resumed_from_run_id"] is None
    assert all(
        isinstance(value, (str, int, float, bool, list, type(None))) for value in payload.values())


def test_build_remote_manifest_records_contract_and_rejects_digest_drift(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_spawn_calls[0][1][0]
    manifest = module.build_remote_manifest(payload)
    assert manifest.modal_version == fake_modal.__version__
    assert manifest.image_digest == PINNED_CUDA_CHILD_DIGEST
    assert manifest.effective_map == "simple"
    assert manifest.gpu == "T4"
    assert manifest.cpu_request == manifest.cpu_soft_limit == 8
    assert manifest.memory_request_mib == manifest.memory_hard_limit_mib == 16384
    assert manifest.vec_workers == 8
    assert manifest.training_argv == payload["training_argv"]
    assert manifest.requested_timesteps == request.timesteps
    assert manifest.effective_timesteps == (request.timesteps //
                                            request.batch_size) * request.batch_size
    assert manifest.batch_size == request.batch_size
    assert manifest.seed == 2
    assert manifest.created_at == _aware().isoformat()
    assert manifest.resume_sha256 is None
    assert manifest.resume_size is None
    assert manifest.resume_source_path is None
    assert manifest.runner_commit == sha
    assert manifest.config_hash == "0" * 64
    assert manifest.thread_caps == _expected_thread_caps()
    assert manifest.resumed_from_run_id is None
    assert manifest.commit == sha
    drifted = dict(payload)
    drifted["image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(mrl.ValidationError, match="digest"):
        module.build_remote_manifest(drifted)


def test_train_remote_writes_manifest_and_rejects_completed_without_evidence(
        fake_modal, tmp_path, monkeypatch):
    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))

    def fake_run_root(run_id: str) -> Path:
        path = tmp_path / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(module, "_remote_run_root", fake_run_root)
    real_execute = mrl.execute_training_attempt
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs):
        manifest = kwargs["manifest"]
        assert isinstance(manifest, mrl.Manifest)
        captured["prepare_manifest"] = manifest
        run_root = Path(kwargs["run_root"])
        run_root.mkdir(parents=True, exist_ok=True)
        lock = kwargs["lock"]
        attempt_id = kwargs["attempt_id"]
        mrl.transition_status(run_root,
                              mrl.Status.PREPARING,
                              now=_aware(),
                              attempt_id=attempt_id,
                              lock=lock)
        mrl.atomic_write_json(run_root / mrl.MANIFEST_FILENAME, manifest.to_dict())
        mrl.transition_status(run_root,
                              mrl.Status.BUILDING,
                              now=_aware(),
                              attempt_id=attempt_id,
                              lock=lock)
        source_dir = tmp_path / "extracted"
        source_dir.mkdir(exist_ok=True)
        return mrl.PreparedSource(
            source_dir=source_dir,
            child_env={
                "PATH": "/usr/bin",
                "OMP_NUM_THREADS": "1"
            },
            train_command=["python", "-c", "pass"],
            heartbeat=_noop_heartbeat(),
            config_hash=manifest.config_hash,
        )

    def fake_execute(**kwargs):
        captured["execute_manifest"] = kwargs["manifest"]
        kwargs["process_factory"] = lambda *a, **k: FakeChild(returncode=0, stdout=b"done\n")
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", fake_prepare)
    monkeypatch.setattr(mrl, "execute_training_attempt", fake_execute)
    result = module.launch_run(request,
                               repo=repo,
                               app_obj=module.app,
                               now=_aware(),
                               stdout=_capture_stdout())
    run_root = tmp_path / "runs" / request.run_id
    written = json.loads((run_root / mrl.MANIFEST_FILENAME).read_text())
    assert written["image_digest"] == PINNED_CUDA_CHILD_DIGEST
    assert written["modal_version"] == fake_modal.__version__
    assert written["effective_map"] == "simple"
    assert written["gpu"] == "T4"
    assert written["cpu_request"] == written["cpu_soft_limit"] == 8
    assert written["memory_request_mib"] == written["memory_hard_limit_mib"] == 16384
    assert written["vec_workers"] == 8
    assert written["training_argv"][0] == "--train"
    assert written["requested_timesteps"] == request.timesteps
    assert written["effective_timesteps"] == (request.timesteps //
                                              request.batch_size) * request.batch_size
    assert written["batch_size"] == request.batch_size
    assert written["seed"] == 2
    assert written["created_at"] == _aware().isoformat()
    assert written["runner_commit"] == sha
    assert written["resume_sha256"] is None
    assert written["resume_size"] is None
    assert written["resume_source_path"] is None
    assert written["thread_caps"] == _expected_thread_caps()
    assert written["resumed_from_run_id"] is None
    assert captured["execute_manifest"].config_hash == captured["prepare_manifest"].config_hash
    assert captured["execute_manifest"].thread_caps == _expected_thread_caps()
    assert result["status"] == "spawned"
    assert json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"] == "failed"
    assert json.loads((run_root / mrl.RESULT_FILENAME).read_text())["status"] == "failed"


def test_train_remote_completes_against_post_dump_manifest_hash(fake_modal, tmp_path, monkeypatch):
    import torch

    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    mount = tmp_path / "artifacts"
    mount.mkdir()
    monkeypatch.setattr(mrl, "VOLUME_MOUNT", mount)
    captured: dict[str, object] = {}
    real_prepare = mrl.prepare_remote_source
    real_execute = mrl.execute_training_attempt

    def materialize(self):
        for key, data in self.files.items():
            dest = mrl.VOLUME_MOUNT.joinpath(*PurePosixPath(key).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    monkeypatch.setattr(FakeVolume, "reload", materialize)

    def fake_run(cmd, **kwargs):
        if "--dump-config" in list(cmd):
            _write_dumped_config(Path(captured["run_root"]))
        return subprocess.CompletedProcess(cmd, 0)

    def prepare_with_real_hash_rewrite(**kwargs):
        captured["prepare_in_manifest"] = kwargs["manifest"]
        captured["run_root"] = Path(kwargs["run_root"])
        kwargs["run"] = fake_run
        kwargs["start_heartbeat"] = _noop_heartbeat
        prepared = real_prepare(**kwargs)
        captured["prepared"] = prepared
        return prepared

    def execute_with_valid_evidence(**kwargs):
        captured["execute_manifest"] = kwargs["manifest"]
        run_root = Path(kwargs["run_root"])

        def factory(*_args, **_kwargs):
            ckpt_dir = run_root / "checkpoints"
            torch.save({"weight": torch.tensor([1.0])}, ckpt_dir / "dust2_policy.pt")
            effective = (request.timesteps // request.batch_size) * request.batch_size
            _write_metrics(ckpt_dir / "metrics.jsonl", [request.batch_size, effective])
            return FakeChild(returncode=0, stdout=b"done\n")

        kwargs["process_factory"] = factory
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", prepare_with_real_hash_rewrite)
    monkeypatch.setattr(mrl, "execute_training_attempt", execute_with_valid_evidence)
    result = module.launch_run(request,
                               repo=repo,
                               app_obj=module.app,
                               now=_aware(),
                               stdout=_capture_stdout())
    run_root = Path(captured["run_root"])
    dumped = json.loads((run_root / "checkpoints" / "config.json").read_text())
    expected_hash = mrl.sha256_bytes(
        json.dumps(mrl.normalize_config_for_transport(dumped),
                   sort_keys=True,
                   separators=(",", ":")).encode())
    on_disk = json.loads((run_root / mrl.MANIFEST_FILENAME).read_text())
    prepared = captured["prepared"]
    execute_manifest = captured["execute_manifest"]
    assert captured["prepare_in_manifest"].config_hash == "0" * 64
    assert prepared.config_hash == expected_hash
    assert on_disk["config_hash"] == expected_hash
    assert execute_manifest.config_hash == expected_hash
    assert execute_manifest.config_hash != "0" * 64
    assert execute_manifest.thread_caps == _expected_thread_caps()
    assert execute_manifest.resumed_from_run_id is None
    assert on_disk["thread_caps"] == _expected_thread_caps()
    assert on_disk["resumed_from_run_id"] is None
    assert result["status"] == "spawned"
    assert json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"] == "completed"
    assert json.loads((run_root / mrl.RESULT_FILENAME).read_text())["status"] == "completed"


def test_train_remote_redelivery_claims_before_prepare(fake_modal, tmp_path, monkeypatch):
    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    factory_calls: list[object] = []
    prepare_calls: list[int] = []

    def fake_run_root(run_id: str) -> Path:
        path = tmp_path / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(module, "_remote_run_root", fake_run_root)
    real_execute = mrl.execute_training_attempt

    def fake_prepare(**kwargs):
        prepare_calls.append(1)
        kwargs["volume"].commit()
        manifest = kwargs["manifest"]
        run_root = Path(kwargs["run_root"])
        run_root.mkdir(parents=True, exist_ok=True)
        lock = kwargs["lock"]
        attempt_id = kwargs["attempt_id"]
        now = _aware(minute=len(prepare_calls))
        mrl.transition_status(run_root,
                              mrl.Status.PREPARING,
                              now=now,
                              attempt_id=attempt_id,
                              lock=lock)
        mrl.atomic_write_json(run_root / mrl.MANIFEST_FILENAME, manifest.to_dict())
        mrl.transition_status(run_root,
                              mrl.Status.BUILDING,
                              now=now,
                              attempt_id=attempt_id,
                              lock=lock)
        source_dir = tmp_path / "extracted"
        source_dir.mkdir(exist_ok=True)
        return mrl.PreparedSource(
            source_dir=source_dir,
            child_env={
                "PATH": "/usr/bin",
                "OMP_NUM_THREADS": "1"
            },
            train_command=["python", "-c", "pass"],
            heartbeat=_noop_heartbeat(),
            config_hash=manifest.config_hash,
        )

    def fake_execute(**kwargs):

        def factory(*args, **factory_kwargs):
            factory_calls.append((args, factory_kwargs))
            return FakeChild(returncode=0, stdout=b"done\n")

        kwargs["process_factory"] = factory
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", fake_prepare)
    monkeypatch.setattr(mrl, "execute_training_attempt", fake_execute)
    first = module.launch_run(request,
                              repo=repo,
                              app_obj=module.app,
                              now=_aware(),
                              stdout=_capture_stdout())
    assert first["status"] != mrl.REDELIVERED
    assert factory_calls
    payload = fake_modal.configured_spawn_calls[0][1][0]
    run_root = tmp_path / "runs" / request.run_id
    status_bytes = (run_root / mrl.STATUS_FILENAME).read_bytes()
    manifest_bytes = (run_root / mrl.MANIFEST_FILENAME).read_bytes()
    volume = fake_modal.volumes[mrl.VOLUME_NAME]
    commits_after_first = volume.commit_count
    factory_count = len(factory_calls)
    prepare_count = len(prepare_calls)
    second = module.train_remote.with_options(
        gpu=request.gpu,
        cpu=request.cpu_request_limit,
        memory=request.memory_request_limit,
        timeout=request.timeout_minutes * 60,
        volumes={
            "/artifacts": volume
        },
    ).remote(payload)
    assert second == {"status": mrl.REDELIVERED, "run_id": request.run_id}
    assert len(prepare_calls) == prepare_count
    assert len(factory_calls) == factory_count
    assert (run_root / mrl.STATUS_FILENAME).read_bytes() == status_bytes
    assert (run_root / mrl.MANIFEST_FILENAME).read_bytes() == manifest_bytes
    assert volume.commit_count == commits_after_first


# ── Artifact client: status / download without importing the launch app ────


def _import_artifacts():
    return importlib.import_module("scripts.modal_artifacts")


def test_artifact_client_never_imports_app_or_creates_objects(fake_modal):
    module = _import_artifacts()
    assert "scripts.run_modal" not in sys.modules
    assert fake_modal.images == []
    assert fake_modal.apps == []
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id")
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.volume_lookups == [(mrl.VOLUME_NAME, False)]
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.base_remote_calls == []


def test_collect_status_missing_run_message(fake_modal):
    module = _import_artifacts()
    _named_volume(fake_modal)
    with pytest.raises(mrl.ValidationError, match="run not found: missing-id"):
        module.collect_status("missing-id", now=_aware())


def test_status_rejects_launch_only_options_via_client(fake_modal):
    module = _import_artifacts()
    with pytest.raises(mrl.ValidationError):
        module.main(["status", "--run-id", "ok-id", "--gpu", "T4"])
    assert fake_modal.volume_lookups == []
    assert fake_modal.volume_creates == []


@pytest.mark.parametrize("case", ["missing", "stale", "mismatch", "replaced", "ok"])
def test_status_checkpoint_loadable_protocol(fake_modal, tmp_path, case):
    import torch

    module = _import_artifacts()
    ckpt = tmp_path / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([5.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    sidecar = {
        "sha256": digest,
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    _ckpt_path, sidecar_path = _write_parent_artifacts(
        volume,
        "ok-id",
        status="completed",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=None if case == "missing" else sidecar,
    )
    if case == "stale":
        volume.files[sidecar_path] = json.dumps({**sidecar, "size": len(ckpt_bytes) + 8}).encode()
    elif case == "mismatch":
        volume.files[sidecar_path] = json.dumps({**sidecar, "sha256": "0" * 64}).encode()
    elif case == "replaced":
        volume.replace_after_read = {
            sidecar_path: json.dumps({
                **sidecar, "sha256": "1" * 64
            }).encode()
        }
    report = module.collect_status("ok-id", now=_aware())
    assert report["run_id"] == "ok-id"
    assert report["status"] == "completed"
    assert report["checkpoint_loadable"] is (case == "ok")


def test_status_does_not_interrupt_on_mere_file_presence(fake_modal, tmp_path):
    import torch

    module = _import_artifacts()
    ckpt = tmp_path / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([6.0])}, ckpt)
    volume = _named_volume(fake_modal)
    _write_parent_artifacts(
        volume,
        "ok-id",
        status="training",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt.read_bytes(),
        sidecar=None,
    )
    dead = (mrl.RUNS_ROOT / "ok-id" / "checkpoints" / mrl.DEAD_CHECKPOINT_NAME).as_posix()
    volume.files[dead] = b"autopsy"
    report = module.collect_status("ok-id", now=_aware())
    assert report["status"] == "training"
    assert report["stale"] is False
    assert report["checkpoint_loadable"] is False


def test_download_stages_renames_and_refuses_overwrite(fake_modal, tmp_path):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b'{"status":"completed"}\n'
    volume.files["runs/ok-id/checkpoints/config.json"] = b"{}\n"
    dest_root = tmp_path / "outputs" / "modal"
    dest = module.download_run("ok-id", dest_root=dest_root)
    assert dest == dest_root / "ok-id"
    assert (dest / "STATUS.json").read_bytes() == b'{"status":"completed"}\n'
    assert (dest / "checkpoints" / "config.json").read_bytes() == b"{}\n"
    assert list(dest_root.glob(".ok-id.tmp-*")) == []
    assert fake_modal.iterdir_calls == [("runs/ok-id", True)]
    assert all(not path.startswith("/artifacts") for path, _rec in fake_modal.iterdir_calls)
    assert all(not path.startswith("/artifacts") for path in fake_modal.read_file_calls)
    with pytest.raises(mrl.ValidationError):
        module.download_run("ok-id", dest_root=dest_root)
    escaped = dest_root / "escaped"
    volume.files["runs/ok-id/../../secret"] = b"nope"
    # Existing dest still blocks; use a new id for the escape case.
    volume.files["runs/evil/../../secret"] = b"nope"
    volume.files["runs/evil/STATUS.json"] = b"{}\n"
    with pytest.raises(mrl.ValidationError):
        module.download_run("evil", dest_root=dest_root)
    assert not escaped.exists()
    assert not (dest_root / "evil").exists()
    assert not (dest_root / "secret").exists()


def test_detached_run_can_be_downloaded_later_by_id(fake_modal, tmp_path):
    module = _import_artifacts()
    assert "scripts.run_modal" not in sys.modules
    volume = _named_volume(fake_modal)
    volume.files["runs/detached-1/result.json"] = b'{"status":"completed"}\n'
    dest = module.download_run("detached-1", dest_root=tmp_path / "outputs" / "modal")
    assert (dest / "result.json").read_text() == '{"status":"completed"}\n'
    assert fake_modal.apps == []
    assert fake_modal.images == []
    assert fake_modal.configured_remote_calls == []


def test_detach_is_a_modal_run_cli_flag_not_an_app_option(fake_modal):
    # Modal 1.4.3 places --detach on `modal run` before FUNC_REF:
    # `modal run --detach scripts/run_modal.py --action run ...`
    module = _import_run_modal()
    assert "detach" not in module.main.__code__.co_varnames


def test_launch_run_spawns_async_so_client_death_does_not_cancel_training(fake_modal, tmp_path):
    """Client SIGTERM must not cancel the GPU input (150826-trunk-seed2-split).

    WHAT: launch_run invokes the configured Function via spawn, never remote.

    WHY: Function.remote() is FUNCTION_CALL_INVOCATION_TYPE_SYNC. Modal's
    client cancels that input on shutdown (`Successfully canceled input`).
    `modal run --detach` only keeps the App; it does not keep a SYNC input
    alive. Function.spawn() is ASYNC — the same invocation type Modal's own
    `--detach` Function CLI uses (cli/run.py: spawn then get).

    PITFALL: adding --detach to the App or catching SIGTERM locally does not
    fix this. The invocation type is the load-bearing bit. Live evidence:
    21.3M/29.98M T4 cancelled at 2026-08-15T13:51:35Z while STATUS stayed
    training because the container was hard-cancelled before finalize.
    """
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(ckpt)))
    result = module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.base_remote_calls == []
    assert len(fake_modal.configured_spawn_calls) == 1
    assert result["status"] == "spawned"
    assert result["function_call_id"] == "fc-test"


# ── FileEntry types, empty prefixes, reservation ───────────────────────────


class FileEntryType(IntEnum):
    """Stand-in for modal.types.FileEntryType. str() is fileentrytype.directory."""

    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3


def test_download_skips_fileentry_directories_and_refuses_symlinks(fake_modal, tmp_path):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b'{"status":"completed"}\n'
    volume.files["runs/ok-id/checkpoints/config.json"] = b"{}\n"
    volume.iterdir_entries = [
        SimpleNamespace(path="runs/ok-id/checkpoints", type=FileEntryType.DIRECTORY),
        SimpleNamespace(path="runs/ok-id/STATUS.json", type=FileEntryType.FILE),
        SimpleNamespace(path="runs/ok-id/checkpoints/config.json", type=FileEntryType.FILE),
    ]
    dest_root = tmp_path / "outputs" / "modal"
    dest = module.download_run("ok-id", dest_root=dest_root)
    assert (dest / "STATUS.json").read_bytes() == b'{"status":"completed"}\n'
    assert (dest / "checkpoints" / "config.json").read_bytes() == b"{}\n"
    assert not (dest / "checkpoints").is_file()

    volume.files["runs/link-id/STATUS.json"] = b"{}\n"
    volume.iterdir_entries = [
        SimpleNamespace(path="runs/link-id/outside", type=FileEntryType.SYMLINK),
        SimpleNamespace(path="runs/link-id/STATUS.json", type=FileEntryType.FILE),
    ]
    with pytest.raises(mrl.ValidationError, match="symlink"):
        module.download_run("link-id", dest_root=dest_root)
    assert not (dest_root / "link-id").exists()


def test_iterdir_paths_treats_missing_prefix_not_found_as_empty(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    volume.missing_prefix_exc = FakeNotFoundError
    assert module._iterdir_paths(volume, "sources") == []
    assert module._iterdir_paths(volume, "runs") == []
    assert not module._volume_has_client_path(volume, "sources/deadbeef.tar.gz")


def test_first_launch_lists_empty_volume_prefixes(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    volume.missing_prefix_exc = FakeNotFoundError
    _named_dict(fake_modal)
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    assert fake_modal.configured_spawn_calls
    assert fake_modal.configured_remote_calls == []
    source_name = next(path for path in volume.files if path.startswith("sources/"))
    assert source_name.endswith(".tar.gz")


def test_launch_upload_failure_records_failure_code_without_freeing_id(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    volume.fail_prefix = "sources/"
    _named_dict(fake_modal)
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    with pytest.raises(OSError, match="could not upload"):
        module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    claim = fake_modal.dicts[mrl.REGISTRY_NAME].get(mrl.run_registry_key(request.run_id))
    assert claim["failure_code"] == mrl.FAILURE_UPLOAD
    assert claim["attempt_id"]
    assert "secret" not in json.dumps(claim)
    assert (mrl.RUNS_ROOT / request.run_id / mrl.RESERVATION_FILENAME).as_posix() in volume.files
    assert fake_modal.configured_remote_calls == []
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())


def test_lookup_helpers_chain_unexpected_errors(fake_modal):
    launch = _import_run_modal()
    artifacts = _import_artifacts()

    class BoomFactory:

        @staticmethod
        def from_name(name, create_if_missing=False):
            del name, create_if_missing
            raise RuntimeError("modal backend exploded")

    with pytest.raises(RuntimeError, match="modal backend exploded") as launch_info:
        launch._lookup_named(BoomFactory, mrl.VOLUME_NAME, missing="artifact volume is missing")
    assert launch_info.value.__cause__ is None

    class BoomModal:

        class Volume:

            @staticmethod
            def from_name(name, create_if_missing=False):
                del name, create_if_missing
                raise RuntimeError("volume backend exploded")

    with pytest.raises(RuntimeError, match="volume backend exploded") as artifact_info:
        artifacts.lookup_volume(BoomModal)
    assert artifact_info.value.__cause__ is None


def test_corrupt_volume_json_is_validation_error(fake_modal):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b"{not-json"
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id", now=_aware())
    del volume.files["runs/ok-id/STATUS.json"]
    volume.files["runs/ok-id/reservation.json"] = b'{"created_at":"not-a-timestamp"}'
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id", now=_aware())


# ── Client-side sidecar backfill ───────────────────────────────────────────
#
# Runs interrupted before ae7dd7b have a good dust2_policy.pt and no sidecar,
# so --resume-run-id refuses them forever: the container that could publish is
# gone. The same hole reopens whenever a container dies without running
# finalize at all (hard preemption, OOM kill, node loss). Backfill closes it
# from the laptop, which has torch, without spending a GPU minute.


def _import_backfill():
    return importlib.import_module("scripts.modal_backfill_sidecar")


def _volume_with_orphan_checkpoint(fake_modal,
                                   tmp_path,
                                   *,
                                   status="interrupted",
                                   sidecar=None,
                                   ckpt_bytes=None):
    import torch

    if ckpt_bytes is None:
        ckpt = tmp_path / "dust2_policy.pt"
        torch.save({"weight": torch.tensor([7.0])}, ckpt)
        ckpt_bytes = ckpt.read_bytes()
    volume = _named_volume(fake_modal)
    _write_parent_artifacts(
        volume,
        "orphan-id",
        status=status,
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=sidecar,
    )
    return volume, ckpt_bytes


def test_backfill_publishes_sidecar_that_satisfies_the_status_client(fake_modal, tmp_path):
    module = _import_backfill()
    artifacts = _import_artifacts()
    volume, ckpt_bytes = _volume_with_orphan_checkpoint(fake_modal, tmp_path)
    assert artifacts.collect_status("orphan-id", now=_aware())["checkpoint_loadable"] is False

    report = module.backfill_sidecar("orphan-id", now=_aware())

    assert report["sha256"] == mrl.sha256_bytes(ckpt_bytes)
    assert report["size"] == len(ckpt_bytes)
    assert artifacts.collect_status("orphan-id", now=_aware())["checkpoint_loadable"] is True
    written = json.loads(volume.files[(mrl.RUNS_ROOT / "orphan-id" / "checkpoints" /
                                       mrl.CHECKPOINT_SIDECAR_NAME).as_posix()])
    # Provenance must be explicit: a client cannot observe the container's mtime.
    assert written["backfilled"] is True
    assert written["mtime_ns"] is None


def test_backfill_refuses_a_run_id_with_neither_status_nor_reservation(fake_modal):
    # An empty volume means the run id does not exist. The backfiller keeps its
    # own copy of this guard (collect_status has the other), so it needs its own
    # pin: without one, the guard could drift below the upload and this test's
    # message assertion would still pass on a run that had already been written
    # to. Assert no upload happened, not just that it raised.
    module = _import_backfill()
    volume = _named_volume(fake_modal)

    with pytest.raises(mrl.ValidationError, match="run not found: missing-id"):
        module.backfill_sidecar("missing-id", now=_aware())

    assert fake_modal.batch_upload_calls == []
    assert (mrl.RUNS_ROOT / "missing-id" / "checkpoints" /
            mrl.CHECKPOINT_SIDECAR_NAME).as_posix() not in volume.files


def test_backfill_refuses_a_run_that_is_still_active(fake_modal, tmp_path):
    module = _import_backfill()
    volume, _ = _volume_with_orphan_checkpoint(fake_modal, tmp_path, status="training")

    with pytest.raises(mrl.ValidationError, match="still active"):
        module.backfill_sidecar("orphan-id", now=_aware())

    assert fake_modal.batch_upload_calls == []
    assert (mrl.RUNS_ROOT / "orphan-id" / "checkpoints" /
            mrl.CHECKPOINT_SIDECAR_NAME).as_posix() not in volume.files


def test_backfill_refuses_to_overwrite_an_existing_sidecar(fake_modal, tmp_path):
    module = _import_backfill()
    existing = {"sha256": "0" * 64, "size": 1, "mtime_ns": 1, "validated_at": _aware().isoformat()}
    _volume_with_orphan_checkpoint(fake_modal, tmp_path, sidecar=existing)

    with pytest.raises(mrl.ValidationError, match="already"):
        module.backfill_sidecar("orphan-id", now=_aware())

    assert fake_modal.batch_upload_calls == []


def test_backfill_refuses_a_checkpoint_that_is_not_weights_only_loadable(fake_modal, tmp_path):
    module = _import_backfill()
    volume, _ = _volume_with_orphan_checkpoint(fake_modal, tmp_path, ckpt_bytes=b"torn-bytes")

    with pytest.raises(mrl.ValidationError, match="weights-only"):
        module.backfill_sidecar("orphan-id", now=_aware())

    assert fake_modal.batch_upload_calls == []
    assert (mrl.RUNS_ROOT / "orphan-id" / "checkpoints" /
            mrl.CHECKPOINT_SIDECAR_NAME).as_posix() not in volume.files


def test_backfill_never_imports_the_launch_app(fake_modal, tmp_path):
    module = _import_backfill()
    _volume_with_orphan_checkpoint(fake_modal, tmp_path)
    module.backfill_sidecar("orphan-id", now=_aware())
    assert "scripts.run_modal" not in sys.modules
    assert fake_modal.images == []
    assert fake_modal.apps == []
    # The fixture creates the volume; the module must only ever look it up, and
    # never with create_if_missing=True.
    assert fake_modal.volume_lookups
    assert all(lookup == (mrl.VOLUME_NAME, False) for lookup in fake_modal.volume_lookups)
    assert fake_modal.dict_creates == []
    assert fake_modal.dict_lookups == []


# ── Gate (a), criterion 3: the library imports on a container-equivalent run ──


def _materialise_recorded_mounts(
        prefix: Path,
        mount_list: list[tuple[str, str, bool]],
        *,
        omit: frozenset[str] = frozenset(),
        plant: str | None = None,
) -> list[Path]:
    """Copy exactly the recorded `(src, dst, copy)` triples into `prefix`; return what was written.

    WHAT: every `dst` is an absolute container path (`/opt/app/scripts/modal_runner_lib.py`), so it
    is rebased to `prefix / dst.lstrip("/")` and the bytes of `src` are copied there. Nothing else
    is created. The result is the image's file set, not the developer's checkout. Per GC1 the
    return value is evidence -- the list of written paths -- so a caller can pin the file count
    instead of trusting this helper's word for it.

    WHY a helper and not `shutil.copytree(ROOT / "scripts")`: the mount list IS the subject. A gate
    that copied the repo would stay green after somebody deleted an `add_local_file` call, which is
    precisely the regression gate (a) exists to catch.

    PITFALL -- a recorded triple whose `src` does not exist is a FAILURE, never a skip. The two
    implementations read identically in prose, and the skipping one stays green on a mount list
    naming a deleted file; a round-2 review built it and measured `GREEN, skipped srcs: [...]`.
    Missing sources therefore raise `FileNotFoundError` here. They never shrink the set quietly.

    PITFALL -- `plant` is inserted after line 19 of whichever copy lands on `modal_runner_lib.py`,
    because line 19 of `scripts/modal_runner_lib.py` is `from __future__ import annotations` and a
    `__future__` import must precede all other code (GC2). A plant above it yields `SyntaxError:
    from __future__ imports must occur at the beginning of the file` -- red for the wrong reason,
    which scores as a working knock-out while proving nothing. Line 19 is re-read here so that a
    future edit to the module header fails loudly rather than mis-planting in silence.

    `omit` drops recorded destinations by container path; that is how knock-out 1 removes a mounted
    file without touching the repo. `scripts/modal_runner_lib.py` itself is never modified -- not
    one byte -- and every plant lives under the caller's `tmp_path`.
    """
    written: list[Path] = []
    for src, dst, _copy in mount_list:
        if dst in omit:
            continue
        source = Path(src)
        if not source.is_file():
            raise FileNotFoundError(f"recorded mount src does not exist: {src!r} -> {dst!r}")
        payload = source.read_bytes()
        if plant is not None and PurePosixPath(dst).name == "modal_runner_lib.py":
            lines = payload.decode("utf-8").splitlines(keepends=True)
            if lines[18].rstrip("\n") != "from __future__ import annotations":
                raise ValueError(f"expected the __future__ import on line 19, found {lines[18]!r}; "
                                 "plants must land after it or they fail with SyntaxError")
            lines.insert(19, plant)
            payload = "".join(lines).encode("utf-8")
        target = prefix / dst.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append(target)
    return written


def _container_equivalent_import(
    prefix: Path,
    pythonpath: str,
    *,
    cwd: Path,
    module: str = "scripts.modal_runner_lib",
    isolate_cwd: bool = True,
) -> tuple[int, str]:
    """Import `module` in a subprocess that can see only `prefix`; return `(exit_code, stderr)`.

    Runs `python -S -B -P -c "import <module>"` with `PYTHONPATH` built by rebasing each
    colon-separated entry of `pythonpath` onto `prefix`. Per GC1 it returns evidence and never
    asserts, so the same helper serves the green case, the three knock-outs, and the I3 regression
    test that deliberately removes `-P`.

    `pythonpath` is a PARAMETER because requirement 1 forbids writing the literal into the gate:
    callers pass `runner_image.env_vars["PYTHONPATH"]`, so a bad edit to the image reaches the
    gate. A gate that hardcoded `/opt/app:/opt/app/scripts` would prove nothing about the value it
    exists to protect.

    `module` defaults to the spelling PRODUCTION uses -- `scripts/run_modal.py:33` is `import
    scripts.modal_runner_lib as mrl`. That is load-bearing, not stylistic: measured, with only
    `<prefix>/opt/app/scripts` on the path the bare name `modal_runner_lib` still imports while the
    packaged name raises `ModuleNotFoundError: No module named 'scripts'`. A gate written on the
    bare spelling is therefore GREEN in exactly the configuration where production breaks.

    FLAGS, and why each is load-bearing:
      * `-P` keeps `sys.path[0]` off the caller's cwd. Without it, and with the cwd at the repo
        root, the subject resolves from the real tree and the gate's own positive control goes
        green.
      * `-S` skips `site.py`, which does two jobs. It hides `modal` and `uv`, which the developer's
        interpreter has and the container's `PYTHONPATH` must not need; and it stops `site.py`
        processing `.venv/lib/python3.12/site-packages/__editable__.cs2rl-0.1.0.pth`, whose entire
        content is `<repo>/src`. Measured without `-S`, neutral cwd, `PYTHONPATH=/nonexistent`:
        `LEAK: <repo>/src/env_config.py`; with `-S`: `ModuleNotFoundError`. `scripts` does not leak
        this way (the `.pth` names `src`), but any future mounted module importing a top-level
        `src/` module would silently resolve from the developer's checkout.
      * `-B` only suppresses `__pycache__` writes, so the "exactly N files" assertions stay true
        and the I3 test -- which runs with `cwd=ROOT` on purpose -- cannot litter the repo.

    `-I` AND `-E` ARE DISQUALIFYING, not merely unnecessary, and this note exists so they are not
    re-proposed as "stronger". `-I` implies `-E`, and `-E` drops `PYTHONPATH` entirely; measured,
    both turn the green case into `ModuleNotFoundError: No module named 'scripts'`. Requirement 1
    says `PYTHONPATH` comes from the image, so an isolated interpreter tests one that cannot see
    the subject at all -- a gate that is red no matter what the package looks like.

    The environment is `PYTHONPATH` alone rather than an inherited `os.environ` copy: measured,
    both give exit 0 on the real subject, and the minimal one cannot import a stray `PYTHONPATH`
    or `PYTHONHOME` from whoever launched pytest.
    """
    entries = [str(prefix / entry.lstrip("/")) for entry in pythonpath.split(":") if entry]
    flags = ["-S", "-B", "-P"] if isolate_cwd else ["-S", "-B"]
    completed = subprocess.run(
        [sys.executable, *flags, "-c", f"import {module}"],
        cwd=str(cwd),
        env={"PYTHONPATH": ":".join(entries)},
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stderr


def test_gate_a_criterion_3_package_imports_on_a_container_equivalent_interpreter(
        fake_modal, tmp_path):
    """Gate (a), criterion 3: only the mounted files, on the image's own PYTHONPATH, import.

    The highest-value gate on this branch, because every other gate and every other criterion can
    pass on a package that cannot be imported on the container. Materialise exactly the triples
    recorded in `runner_image.local_files` under `tmp_path`, then import the packaged spelling on
    an interpreter that can see nothing else.

    WHICH ENTRY THIS GATE COVERS. The image sets `PYTHONPATH=/opt/app:/opt/app/scripts` and the two
    entries carry different spellings: the packaged name `scripts.modal_runner_lib` resolves via
    `/opt/app`, the bare name `modal_runner_lib` via `/opt/app/scripts`. This gate covers the
    `/opt/app` entry -- it is green on `/opt/app` alone. The `/opt/app/scripts` entry is pinned
    only by `tests/test_modal_client.py:626`; nothing here would notice its removal, which is why
    the assertions below record both single-entry results explicitly rather than describing them.

    `runner_image is dependency_image` (`FakeImage.add_local_file` returns `self`, and
    `scripts/run_modal.py:113` chains off `dependency_image`), so the recorded list is FIVE triples
    and all five are materialised. Reading mounts off `dependency_image` and off `runner_image`
    gives the same object; reading them off a locally rebuilt image would not.

    DO NOT INVENT `scripts/__init__.py`. The repo has none, so `scripts` is a PEP 420 namespace
    package and `sys.modules['scripts'].__file__ is None` in the subprocess. Creating one "to be
    safe" changes the failure shape of the unmounted-intra-repo-import knock-out, which is the
    single control that distinguishes this gate from a smoke test.

    STATED LIMIT -- pre-split this gate exercises 1 of the 5 recorded triples. The three
    `/opt/cs2rl/*` mounts (`pyproject.toml`, `uv.lock`, `modal_image_reqs.py`) are never on
    `sys.path`, and `scripts/run_modal.py` imports `modal` at module scope so `-S` puts it out of
    reach by construction (measured: `ModuleNotFoundError: No module named 'modal'`). The gate
    therefore covers the library, not the image or the entrypoint. That is a known, unclosed gap,
    stated rather than implied away; the post-split figure in the spec's 6.1 is a different number
    and does not transfer backwards to this commit.
    """
    module = _import_run_modal()
    runner = module.runner_image
    assert runner is module.dependency_image
    mount_list = runner.local_files
    assert len(mount_list) == 5

    prefix = tmp_path / "image"
    neutral = tmp_path / "cwd"
    neutral.mkdir()
    written = _materialise_recorded_mounts(prefix, mount_list)
    assert len(written) == 5
    # Exactly the recorded files and nothing else: a stray source-adjacent artefact would make a
    # file the knock-out below has removed look present.
    assert sorted(path for path in prefix.rglob("*") if path.is_file()) == sorted(written)

    pythonpath = runner.env_vars["PYTHONPATH"]
    assert _container_equivalent_import(prefix, pythonpath, cwd=neutral) == (0, "")

    # The image's two entries, separately, so the division of labour is measured and not asserted
    # in prose. `/opt/app` alone: this gate green, Modal's own `import run_modal` red.
    assert _container_equivalent_import(prefix, "/opt/app", cwd=neutral) == (0, "")
    run_modal_status, run_modal_stderr = _container_equivalent_import(prefix,
                                                                      "/opt/app",
                                                                      cwd=neutral,
                                                                      module="run_modal")
    assert run_modal_status != 0
    assert "No module named 'run_modal'" in run_modal_stderr

    # `/opt/app/scripts` alone: the packaged spelling this gate uses is red, the bare spelling a
    # weaker gate might have used is green. This is the configuration a bare-spelling gate misses.
    packaged_status, packaged_stderr = _container_equivalent_import(prefix,
                                                                    "/opt/app/scripts",
                                                                    cwd=neutral)
    assert packaged_status != 0
    assert "No module named 'scripts'" in packaged_stderr
    assert _container_equivalent_import(prefix,
                                        "/opt/app/scripts",
                                        cwd=neutral,
                                        module="modal_runner_lib") == (0, "")


def test_gate_a_criterion_3_reddens_when_the_mounted_library_is_absent(fake_modal, tmp_path):
    """Knock-out 1: drop a recorded mount and the gate goes red.

    THIS CONTROL IS DEGENERATE PRE-SPLIT AND IS LABELLED AS SUCH. Only one mounted file is ever on
    `sys.path` at this commit, so "a mounted file removed" can only mean
    `/opt/app/scripts/modal_runner_lib.py` -- the import target itself. Removing it proves that the
    subprocess's exit status is actually checked and that materialisation is driven by the mount
    list, and little more. It becomes a real instrument after the split, when the package has
    siblings that can go missing individually. Keep it, do not oversell it.

    The file-count assertion is what defends it: without pinning the prefix to exactly four files,
    a stray `__pycache__` entry beside the removed source could make the removal invisible.

    The tail of this test gates construction requirement 4 -- a recorded triple whose `src` does
    not exist must FAIL, never be skipped. An omitted mount and a skipped-because-missing mount
    produce the SAME red here, so `omit` alone cannot tell the two implementations apart; the
    `FileNotFoundError` assertion is the only thing that does.
    """
    module = _import_run_modal()
    runner = module.runner_image
    prefix = tmp_path / "image"
    neutral = tmp_path / "cwd"
    neutral.mkdir()

    written = _materialise_recorded_mounts(prefix,
                                           runner.local_files,
                                           omit=frozenset({"/opt/app/scripts/modal_runner_lib.py"}))
    assert len(written) == 4
    assert sorted(path for path in prefix.rglob("*") if path.is_file()) == sorted(written)

    status, stderr = _container_equivalent_import(prefix,
                                                  runner.env_vars["PYTHONPATH"],
                                                  cwd=neutral)
    assert status != 0
    assert "No module named 'scripts.modal_runner_lib'" in stderr

    deleted = str(ROOT / "scripts" / "modal_runner_lib_that_was_deleted.py")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _materialise_recorded_mounts(tmp_path / "missing-src",
                                     [(deleted, "/opt/app/scripts/deleted.py", True)])


def test_gate_a_criterion_3_reddens_on_an_unmounted_intra_repo_import(fake_modal, tmp_path):
    """Knock-out 2, THE DISCRIMINATOR: an import of an intra-repo module outside the mounted set.

    `scripts/modal_image_reqs.py` is mounted at `/opt/cs2rl/modal_image_reqs.py`, which is never on
    `PYTHONPATH`, so a module-scope import of it is unreachable on the container even though the
    file exists in the repo. This is the failure the split can actually produce -- a new sibling
    that nobody added to `add_local_file` -- and it is the only control here that a gate with a
    cwd leak cannot pass. A gate that passes knock-outs 1 and 3 but not this one is leaking and
    must be fixed, not waived.

    ASSERT ON THE FAILURE AND THE MODULE NAME, NEVER ON THE LITERAL MESSAGE. The error takes two
    shapes depending on how the import is spelled, and both are measured here rather than picked:

        from scripts import modal_image_reqs   -> ImportError: cannot import name
                                                  'modal_image_reqs' from 'scripts' (unknown
                                                  location)
        import scripts.modal_image_reqs        -> ModuleNotFoundError: No module named
                                                  'scripts.modal_image_reqs'

    Both are red. A control pinned to the second string mis-scores the first as a gate failure,
    which is why both plants run through the same shape-agnostic assertion. The first shape is a
    consequence of `scripts` being a PEP 420 namespace package -- see the green test on why an
    invented `scripts/__init__.py` would change it.
    """
    module = _import_run_modal()
    runner = module.runner_image
    neutral = tmp_path / "cwd"
    neutral.mkdir()

    plants = {
        "from_import": "from scripts import modal_image_reqs as _unmounted\n",
        "dotted_import": "import scripts.modal_image_reqs as _unmounted\n",
    }
    for label, plant in plants.items():
        prefix = tmp_path / label
        _materialise_recorded_mounts(prefix, runner.local_files, plant=plant)
        status, stderr = _container_equivalent_import(prefix,
                                                      runner.env_vars["PYTHONPATH"],
                                                      cwd=neutral)
        assert status != 0, f"{label}: gate stayed green on an unmounted intra-repo import"
        assert "modal_image_reqs" in stderr, f"{label}: {stderr}"
        assert "ImportError" in stderr or "ModuleNotFoundError" in stderr, f"{label}: {stderr}"


def test_gate_a_criterion_3_reddens_on_a_module_scope_third_party_import(fake_modal, tmp_path):
    """Knock-out 3: a module-scope third-party import is red, because the container has no venv.

    `scripts/modal_runner_lib.py` promises in its own docstring to use only the standard library at
    module scope. The Modal function runs on the image's standalone python, which carries only `uv`
    and the modal client -- not this venv's site-packages -- so a stray `import numpy` at module
    scope is an outage, not a lint nit.

    THIS CONTROL IS ALSO THE `-S` WATCHDOG, AND THAT IS FREE COVERAGE WORTH CLAIMING. numpy 2.4.3
    is installed in this venv, so the same plant was measured `ModuleNotFoundError` with `-S` and
    EXIT 0 without it. If `-S` is ever dropped from `_container_equivalent_import`, this test goes
    green and the suite reddens here. Nothing watches `-P` the same way, which is why the I3 test
    below exists separately.

    STATED LIMIT: for an UNCONDITIONAL module-scope import, `-S` makes this gate stricter than the
    container, deliberately, matching the module docstring's contract. The strictness does not
    extend to a header that merely TESTS for a name -- see the conditional-probe test below.

    The `line 20` assertion is the only direct evidence in this file that GC2's plant-placement
    rule is honoured: line 19 is `from __future__ import annotations`, so a plant that lands on
    line 20 landed immediately after it. A plant above the `__future__` import raises `SyntaxError`
    instead, which is red for the wrong reason and would otherwise score as a working knock-out.
    """
    module = _import_run_modal()
    runner = module.runner_image
    prefix = tmp_path / "image"
    neutral = tmp_path / "cwd"
    neutral.mkdir()

    _materialise_recorded_mounts(prefix, runner.local_files, plant="import numpy as _third_party\n")
    status, stderr = _container_equivalent_import(prefix,
                                                  runner.env_vars["PYTHONPATH"],
                                                  cwd=neutral)
    assert status != 0
    assert "No module named 'numpy'" in stderr
    assert 'modal_runner_lib.py", line 20, in <module>' in stderr


def test_gate_a_criterion_3_stays_green_on_a_conditional_modal_probe(fake_modal, tmp_path):
    """The green half of criterion 13's pair: this gate does NOT object to a `find_spec` header.

    GC2 asks every knock-out to have a counterpart that pins the green half, and this is criterion
    13's. The plant is a module-scope header that TESTS for `modal` rather than importing it:

        import importlib.util
        if importlib.util.find_spec('modal') is not None:
            _ON_CONTAINER = True
        else:
            _ON_CONTAINER = False

    WHY THE STRICTNESS OF `-S` STOPS HERE. Under an unconditional `import modal`, `-S` makes this
    gate stricter than the container and the plant goes red. Under a header that tests for the
    name, `-S` does not make the gate strict -- it makes it take the OTHER BRANCH and prove the
    wrong half of the file. A round-2 review built this plant and measured it green here, green on
    all three knock-outs, and green on criterion 5. Measured under this gate's own `-S`, the
    observed value is `_ON_CONTAINER = False`; the container's is `True`, because the container
    really does have the modal client. The gate certifies an import that resolves a different way
    on the machine it is certifying for.

    THE OTHER HALF IS NOT OPTIONAL. Criterion 13 -- the module-shape gate -- is the only thing that
    catches this, and it lives in `tests/test_modal_packaging.py`. Its counterpart asserts the same
    plant is RED there:

        tests/test_modal_packaging.py::test_criterion_13_reddens_on_the_conditional_modal_probe

    Together the two halves establish that criterion 13 is the SOLE objector. This half lives here,
    beside the subprocess helper, rather than next to its counterpart because
    `tests/test_modal_packaging.py` imports zero `tests.*` modules and that exclusion is
    load-bearing (`_seam_sources()` omits the file because self-feeding the classifier returns 300
    destinations instead of 261). Duplicating the helper into that file would put two copies of
    this gate in the tree -- the defect class this branch exists to close.
    """
    module = _import_run_modal()
    runner = module.runner_image
    prefix = tmp_path / "image"
    neutral = tmp_path / "cwd"
    neutral.mkdir()

    _materialise_recorded_mounts(prefix,
                                 runner.local_files,
                                 plant=("import importlib.util\n"
                                        "if importlib.util.find_spec('modal') is not None:\n"
                                        "    _ON_CONTAINER = True\n"
                                        "else:\n"
                                        "    _ON_CONTAINER = False\n"))
    # Pin that the plant LANDED. An unplanted tree also returns (0, ""), so without this the test
    # would pass just as happily on an instrument that never ran -- the defect class this branch
    # exists to close. The knock-outs pin the same helper's plant path from the red side.
    assert "_ON_CONTAINER" in (prefix /
                               "opt/app/scripts/modal_runner_lib.py").read_text(encoding="utf-8")
    assert _container_equivalent_import(prefix, runner.env_vars["PYTHONPATH"],
                                        cwd=neutral) == (0, "")


def test_gate_a_criterion_3_minus_p_and_the_neutral_cwd_are_load_bearing(fake_modal, tmp_path):
    """I3: assert the leak EXISTS once `-P` and the neutral cwd are removed, so nothing else can.

    "A gate that passes knock-outs 1 and 3 but not 2 is leaking" is true only of the JOINT
    regression. Measured on the knock-out 2 plant, one variable at a time:

        -S -P, neutral cwd : (1, ImportError: cannot import name 'modal_image_reqs' ...)
        -S   , neutral cwd : (1, ImportError: ...)   <- `-P` dropped, the control is STILL red
        -S   , cwd=ROOT    : (0, '')                 <- the leak
        -S -P, cwd=ROOT    : (1, ImportError: ...)

    So dropping `-P` while keeping the neutral cwd leaves all three knock-outs red: the regression
    is invisible and the gate is then one cwd edit away from blind. Knock-out 3 watches `-S` for
    free; nothing watched `-P` until this test.

    BOTH ROWS AT `cwd=ROOT` ARE ASSERTED, AND THE PAIR IS THE POINT. The `isolate_cwd=False` row
    asserts exit 0 -- that the leak is real, GC2's "assert the green half" turned on the gate's own
    construction, so nobody can later argue `-P` is redundant given the neutral cwd. The
    `isolate_cwd=True` row asserts the same run is RED, and that is the half with teeth: on its own
    the exit-0 row does NOT catch `-P` being deleted from the helper's default flags, because the
    row that omits `-P` would keep passing. Only holding `cwd` fixed and toggling `-P` makes the
    flag's removal visible. It also catches the opposite edit -- hardcoding `-P` unconditionally
    and ignoring `isolate_cwd`.

    `cwd=ROOT` is deliberate here and is the one place this file's subprocess runs inside the repo.
    WHAT THE EXIT 0 ACTUALLY MEANS IS WORSE THAN A RESOLVED IMPORT, and the mechanism is worth
    stating precisely because the obvious reading is wrong. With `sys.path[0]` back at the repo
    root, `scripts` becomes a PEP 420 namespace package spanning BOTH trees, repo portion first:

        sys.path[0]      = ''                                   (the cwd, i.e. the repo root)
        scripts.__path__ = ['<repo>/scripts', '<prefix>/opt/app/scripts']
        module.__file__  = '<repo>/scripts/modal_runner_lib.py'  <- the real, UNPLANTED file

    So `scripts.modal_runner_lib` ITSELF is served from the checkout, the prefix copy is never
    loaded, and the plant never runs at all -- measured, `'scripts.modal_image_reqs' in sys.modules`
    is `False` and the planted `_unmounted` attribute is absent. The exit 0 does not say "the
    unreachable import resolved"; it says the gate has stopped testing the mounted set entirely and
    is certifying the developer's tree. Do not read this row as evidence about the prefix -- this
    row does not load the prefix.

    `-B` keeps that run from writing `<repo>/scripts/__pycache__/modal_runner_lib.cpython-*.pyc`
    (measured: delete the `.pyc` first and it is rewritten without `-B`, not written with it). No
    `modal_image_reqs` bytecode appears either way, for the reason above. NOTHING WATCHES `-B` --
    removing it leaves all six of these tests green. It is hygiene, not a gated property.
    """
    module = _import_run_modal()
    runner = module.runner_image
    prefix = tmp_path / "image"

    _materialise_recorded_mounts(prefix,
                                 runner.local_files,
                                 plant="from scripts import modal_image_reqs as _unmounted\n")
    assert _container_equivalent_import(prefix,
                                        runner.env_vars["PYTHONPATH"],
                                        cwd=ROOT,
                                        isolate_cwd=False) == (0, "")

    isolated_status, isolated_stderr = _container_equivalent_import(prefix,
                                                                    runner.env_vars["PYTHONPATH"],
                                                                    cwd=ROOT)
    assert isolated_status != 0, "`-P` no longer closes the cwd leak"
    assert "modal_image_reqs" in isolated_stderr
