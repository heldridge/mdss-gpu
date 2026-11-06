cmake_minimum_required(VERSION 3.22)
project(decoder LANGUAGES CXX CUDA)

option(BUILD_ALL_ARCHS "Build for all supported CUDA architectures" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BUILD_ALL_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 120)
else()
    set(CMAKE_CUDA_ARCHITECTURES 121)
endif()


# Needed to make clangd see includes for .cu files
# Would love another fix if available!
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Compile with known modulus for optimization
set(FIELD_MODULUS "100003" CACHE STRING "The modulus used for the finite field operations")

include(CTest)
include(FetchContent)

# --- Dependencies ---
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

if(BUILD_TESTING)
    find_package(Boost 1.83.0 COMPONENTS unit_test_framework REQUIRED)
endif()

add_subdirectory(instance)
add_subdirectory(config)

add_executable(decoder src/main.cpp)
target_link_libraries(decoder PRIVATE
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    ntl
    gmp
    Instance
    Serialization
    Config
    MDSS
)
target_include_directories(decoder PRIVATE include)

add_subdirectory(serialization)
add_subdirectory(finite_field)
add_subdirectory(math)
add_subdirectory(mdss)
add_subdirectory(array3d)
