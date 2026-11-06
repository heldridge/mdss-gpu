add_library(Array3D STATIC src/array3d.cpp)

target_include_directories(Array3D PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(Array3D PUBLIC ntl gmp)
