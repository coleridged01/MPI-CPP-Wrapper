cmake_minimum_required(VERSION 3.22)
project("MPI C++ Wrapper")

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate 'build' directory and build in there.")
endif()

find_package(doctest REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wconversion -pedantic -Wfatal-errors)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W3 /WX)
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})

add_library(MPIWrapper
    src/MPIEnvironment.cpp
    src/Process.cpp
)

target_link_libraries(MPIWrapper MPI::MPI_CXX)

target_include_directories(MPIWrapper PUBLIC
        include
)

add_subdirectory("test")
