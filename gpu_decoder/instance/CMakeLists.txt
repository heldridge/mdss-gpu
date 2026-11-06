add_library(Instance STATIC src/instance.cpp)
target_include_directories(Instance PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(Instance PUBLIC nlohmann_json::nlohmann_json ntl gmp)
    

