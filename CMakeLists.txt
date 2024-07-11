cmake_minimum_required(VERSION 3.28)
project(cuda_02 CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)
find_package(SFML)


add_executable(cuda_02 kernel.cu main.cpp)

target_link_libraries(cuda_02 fmt::fmt sfml::sfml)

target_include_directories(cuda_02
        PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(cuda_02 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)