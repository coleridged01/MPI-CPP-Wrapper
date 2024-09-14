include(doctest)

add_executable(testMPIWrapper
        testMPIWrapper.cpp
)

add_executable(testScatter
        testScatter.cpp
)

target_link_libraries(testMPIWrapper PRIVATE MPIWrapper doctest::doctest pthread)

target_compile_features(testMPIWrapper PUBLIC cxx_std_20)

doctest_discover_tests(testMPIWrapper TEST_PREFIX "testMPIWrapper_")
