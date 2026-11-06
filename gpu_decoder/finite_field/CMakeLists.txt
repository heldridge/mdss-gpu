add_library(FiniteField INTERFACE)
target_include_directories(FiniteField INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTING)
    add_executable(field_test tests/field.cu)

    target_link_libraries(field_test PRIVATE 
        FiniteField
        Boost::unit_test_framework
    )

    add_test(NAME field_test COMMAND field_test)
endif()
