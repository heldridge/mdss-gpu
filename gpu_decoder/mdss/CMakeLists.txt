add_library(MDSS STATIC src/mdss.cu)
target_include_directories(MDSS PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(MDSS PUBLIC Math FiniteField Instance Config Array3D Serialization ntl gmp)

set_target_properties(MDSS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)
    add_executable(mdss_test tests/mdss.cu)

    target_link_libraries(mdss_test PRIVATE 
        MDSS
        Boost::unit_test_framework
    )

    add_test(NAME mdss_test COMMAND mdss_test)
endif()


