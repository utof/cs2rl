cmake_minimum_required(VERSION 3.18)
project(cs2rl_binding C)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

python_add_library(binding MODULE binding.c cs2_env.c WITH_SOABI)

target_include_directories(binding PRIVATE ${Python_NumPy_INCLUDE_DIRS})

target_compile_options(binding PRIVATE
    -O3 -march=native -ffast-math -std=c99
    -Wall -Wno-unused-function
)

target_link_libraries(binding PRIVATE m)

install(TARGETS binding DESTINATION .)
