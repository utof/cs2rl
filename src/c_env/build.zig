// build.zig — Zig build script for the cs2rl C environment.
//
// Targets:
//   (default)   binding            — CPython extension module (.so / .pyd)
//   cs2_demo                       — standalone Raylib demo (requires X11 dev libs on Linux)
//   demo_events_test (optional)    — headless detect-helper tests; no Raylib, not in binding
//
// Include paths are passed by setup.py as -D flags because Zig has no
// equivalent of CMake's find_package(Python NumPy). This keeps discovery
// in Python where sysconfig/numpy APIs are available.
//
// SOABI suffix (.cpython-312-x86_64-linux-gnu.so) is also handled by
// setup.py after the build; Zig always outputs libbinding.so.
//
// Zig issue #9013: do NOT add a .version field to addSharedLibrary on
// Windows — it causes a linker panic with versioned shared libs.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Build options passed by setup.py shim ──────────────────────────────
    // python_include: output of sysconfig.get_path("include")
    // numpy_include:  output of numpy.get_include()
    // link_python:    true on Windows only (symbols not pre-loaded in process)
    const python_include = b.option([]const u8, "python_include",
        "Python include directory (passed by setup.py)") orelse "";
    const numpy_include  = b.option([]const u8, "numpy_include",
        "NumPy include directory (passed by setup.py)")  orelse "";
    const link_python    = b.option(bool, "link_python",
        "Link libpython — required on Windows, wrong on Linux/macOS") orelse false;

    // fast_math: R0-F (#136). Default true = production flags. `-Dfast_math=false`
    // builds a diagnostic variant WITHOUT -ffast-math so tests/test_fast_math_variant.py
    // can prove the reward guards do not depend on the optimiser folding isfinite().
    // PITFALL: never build the production .so with false — setup.py does not
    // pass this option, so `setup.py build_ext` always yields the fast-math build.
    const fast_math      = b.option(bool, "fast_math",
        "Compile binding.c with -ffast-math (default true; false = diagnostic variant)") orelse true;
    const c_flags_fast: []const []const u8 = &.{
        "-std=c99",
        "-O3",           // intentional: C-level flag overrides -Doptimize for this file
        "-march=native", // safe: all users build from source, no .so committed
        "-ffast-math",
        "-Wall",
        "-Wno-unused-function",
    };
    const c_flags_strict: []const []const u8 = &.{
        "-std=c99", "-O3", "-march=native", "-Wall", "-Wno-unused-function",
    };

    // ── binding: CPython extension module ──────────────────────────────────
    const lib = b.addSharedLibrary(.{
        .name = "binding", // setup.py renames output to SOABI-suffixed name
        .root_module = b.createModule(.{
            .target   = target,
            .optimize = optimize,
        }),
    });

    lib.root_module.addCSourceFile(.{
        .file  = b.path("binding.c"),
        .flags = if (fast_math) c_flags_fast else c_flags_strict,
    });

    // Only add include paths when provided — empty string means build.zig
    // was invoked directly (e.g. during development); headers must be on
    // the system include path in that case.
    if (python_include.len > 0)
        lib.root_module.addSystemIncludePath(.{ .cwd_relative = python_include });
    if (numpy_include.len > 0)
        lib.root_module.addSystemIncludePath(.{ .cwd_relative = numpy_include });

    lib.linkLibC();
    lib.linkSystemLibrary("m");

    // Windows: Python symbols are not pre-loaded via dlopen, must link explicitly.
    // Linux/macOS: linking libpython causes double-load issues — do NOT add.
    if (link_python)
        lib.linkSystemLibrary("python3");

    b.installArtifact(lib);

    // ── cs2_demo: standalone Raylib visualisation demo ─────────────────────
    // Invoked explicitly: `zig build cs2_demo`
    // NOT built by default (does not affect `uv sync` / pip install).
    // Raylib is fetched from build.zig.zon on first run; subsequent runs use
    // the Zig package cache (~/.cache/zig).
    // Force X11 backend on Linux: wayland-scanner is often absent on dev/CI
    // machines. The option name and enum literal `.X11` match raylib's own
    // `b.option(LinuxDisplayBackend, "linux_display_backend", ...)` declaration.
    // Zig 0.14 passes these as typed dependency options.
    // .Debug is intentional — cs2_demo is a dev/visualisation tool, always
    // built with debug info regardless of -Doptimize. The C flag -O2 below
    // provides code-gen optimisation while preserving debug symbols (-g).
    const raylib_dep    = b.dependency("raylib", .{
        .target                = target,
        .optimize              = .Debug,
        .linux_display_backend = .X11,
    });
    const cs2_demo_step = b.step("cs2_demo", "Build the standalone Raylib cs2 demo");

    const demo = b.addExecutable(.{
        .name = "cs2_demo",
        .root_module = b.createModule(.{
            .target   = target,
            .optimize = .Debug, // intentional: see comment above raylib_dep
        }),
    });
    demo.root_module.addCSourceFile(.{
        .file  = b.path("cs2_demo.c"),
        // -Wno-comment: cs2_types.h has `*_SIZE/*_STRIDE` inside a block
        // comment (pre-existing). Zig's clang treats that -Wall warning as
        // a hard error; do not touch the binding-shared header here.
        .flags = &.{ "-std=c99", "-O2", "-Wall", "-Wno-comment", "-g" },
    });
    // Same TU as libcs2_play — compiled into the exe, not linked from the .so.
    demo.root_module.addCSourceFile(.{
        .file  = b.path("cs2_play_host.c"),
        .flags = &.{ "-std=c99", "-O2", "-Wall", "-Wno-comment", "-g" },
    });
    // cs2_demo.c includes cs2_types.h from the same directory
    demo.root_module.addIncludePath(b.path("."));
    demo.root_module.linkLibrary(raylib_dep.artifact("raylib"));
    demo.linkLibC();

    // Raylib's own build.zig handles platform link flags automatically:
    // opengl32 + gdi32 + winmm on Windows, X11/GL on Linux, etc.

    const install_demo = b.addInstallArtifact(demo, .{});
    cs2_demo_step.dependOn(&install_demo.step);

    // Voices live in demo_assets/ — src/c_env/resources is a pufferlib
    // symlink (and gitignored). Copy next to zig-out/bin/cs2_demo so the
    // runtime walk (GetApplicationDirectory() + "resources/") finds them.
    const install_voices = b.addInstallDirectory(.{
        .source_dir     = b.path("demo_assets"),
        .install_dir    = .bin,
        .install_subdir = "resources",
    });
    cs2_demo_step.dependOn(&install_voices.step);

    // libcs2_play: Raylib attach ABI for src/play.py. Same source as the
    // statue exe. Installed only via cs2_demo_step — do not
    // b.installArtifact this on the default/binding path.
    const play_lib = b.addSharedLibrary(.{
        .name = "cs2_play",
        .root_module = b.createModule(.{
            .target   = target,
            .optimize = .Debug,
        }),
    });
    play_lib.root_module.addCSourceFile(.{
        .file  = b.path("cs2_play_host.c"),
        .flags = &.{ "-std=c99", "-O2", "-Wall", "-Wno-comment", "-g" },
    });
    play_lib.root_module.addIncludePath(b.path("."));
    play_lib.root_module.linkLibrary(raylib_dep.artifact("raylib"));
    play_lib.linkLibC();

    const install_play_lib = b.addInstallArtifact(play_lib, .{});
    cs2_demo_step.dependOn(&install_play_lib.step);

    // ── demo_events_test: raylib-free detect-helper unit tests ─────────────
    // Invoked explicitly: `zig build demo_events_test`
    // Compiles demo_events_test.c + cs2_demo_events.h only. Must NOT link
    // Raylib or join the default binding install — training stays display-free.
    // linkSystemLibrary("m") is required for hypotf on this Linux toolchain.
    const demo_events_test = b.addExecutable(.{
        .name = "demo_events_test",
        .root_module = b.createModule(.{
            .target   = target,
            .optimize = optimize,
        }),
    });
    demo_events_test.root_module.addCSourceFile(.{
        .file  = b.path("demo_events_test.c"),
        // Same -Wno-comment as cs2_demo: cs2_types.h has `*_SIZE/*_STRIDE`
        // inside a block comment. Do not touch the binding-shared header.
        .flags = &.{ "-std=c99", "-Wall", "-Wno-comment", "-g" },
    });
    demo_events_test.root_module.addIncludePath(b.path("."));
    demo_events_test.linkLibC();
    demo_events_test.linkSystemLibrary("m");

    const run_demo_events_test = b.addRunArtifact(demo_events_test);
    const demo_events_test_step = b.step(
        "demo_events_test",
        "Run headless demo event-detect tests (no Raylib)",
    );
    demo_events_test_step.dependOn(&run_demo_events_test.step);
}
