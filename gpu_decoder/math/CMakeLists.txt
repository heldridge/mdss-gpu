add_library(Math STATIC src/ff_math.cu src/wpf_utils.cu src/poly_matrix.cpp src/poly_utils.cpp)
target_include_directories(Math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# Link the header-only utils
target_link_libraries(Math PUBLIC FiniteField Array3D ntl gmp)

set_target_properties(Math PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Inject field modulus
target_compile_definitions(Math PUBLIC FIELD_MODULUS=${FIELD_MODULUS})

if(BUILD_TESTING)
    add_executable(math_test tests/main.cu tests/ff_math.cu tests/wpf.cu tests/poly_matrix.cpp)

    target_link_libraries(math_test PRIVATE 
        Math
        Boost::unit_test_framework
    )

    add_test(NAME math_test COMMAND math_test)
endif()


