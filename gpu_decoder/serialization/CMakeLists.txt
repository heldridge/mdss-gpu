add_library(Serialization INTERFACE)

target_include_directories(Serialization INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(Serialization INTERFACE ntl nlohmann_json::nlohmann_json)
