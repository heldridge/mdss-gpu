add_library(Config STATIC src/config.cpp)
target_include_directories(Config PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(Config PUBLIC nlohmann_json::nlohmann_json)
    

