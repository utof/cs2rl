cmake_minimum_required(VERSION 3.18)
project(cs2rl_binding C)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

python_add_library(binding MODULE binding.c WITH_SOABI)

target_include_directories(binding PRIVATE ${Python_NumPy_INCLUDE_DIRS})

target_compile_options(binding PRIVATE
    -O3 -march=native -ffast-math -std=c99
    -Wall -Wno-unused-function
)

target_link_libraries(binding PRIVATE m)

install(TARGETS binding DESTINATION .)

# ── Phase 6: standalone Raylib demo ──────────────────────────────────────────

include(FetchContent)
FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

# cs2_demo — standalone interactive executable (NOT linked into the Python binding)
# DO NOT add cs2_env.c here: it defines its own main() and is not compiled by any target.
add_executable(cs2_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/cs2_demo.c
)
target_include_directories(cs2_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cs2_demo PRIVATE raylib m)
target_compile_definitions(cs2_demo PRIVATE PLATFORM_DESKTOP)
target_compile_options(cs2_demo PRIVATE -O2 -Wall -g -std=c99)
