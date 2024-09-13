
add_executable(testMPIWrapper
        testMPIWrapper.cpp
)

add_executable(testScatter
        testScatter.cpp
)

target_link_libraries(testMPIWrapper PRIVATE MPIWrapper pthread)
target_link_libraries(testScatter PRIVATE MPIWrapper pthread)

target_compile_features(testMPIWrapper PUBLIC cxx_std_20)
target_compile_features(testScatter PUBLIC cxx_std_20)
