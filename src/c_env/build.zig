// build.zig — Zig build script for the cs2rl C environment.
//
// Targets:
//   (default)   binding   — CPython extension module (.so / .pyd)
//   cs2_demo              — standalone Raylib demo (requires X11 dev libs on Linux)
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
        .flags = &.{
            "-std=c99",
            "-O3",           // intentional: C-level flag overrides -Doptimize for this file
            "-march=native", // safe: all users build from source, no .so committed
            "-ffast-math",
            "-Wall",
            "-Wno-unused-function",
        },
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
            .optimize = .Debug,
        }),
    });
    demo.root_module.addCSourceFile(.{
        .file  = b.path("cs2_demo.c"),
        .flags = &.{ "-std=c99", "-O2", "-Wall", "-g" },
    });
    // cs2_demo.c includes cs2_types.h from the same directory
    demo.root_module.addIncludePath(b.path("."));
    demo.root_module.linkLibrary(raylib_dep.artifact("raylib"));
    demo.linkLibC();

    // Raylib's own build.zig handles platform link flags automatically:
    // opengl32 + gdi32 + winmm on Windows, X11/GL on Linux, etc.

    const install_demo = b.addInstallArtifact(demo, .{});
    cs2_demo_step.dependOn(&install_demo.step);
}
