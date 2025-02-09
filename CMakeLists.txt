cmake_minimum_required(VERSION 3.16)
project(YieldCurveFitting LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Ninja support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(OPTIMIZERS_DIR ${INCLUDE_DIR}/Optimizers)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

# Include directories
include_directories(${INCLUDE_DIR} ${OPTIMIZERS_DIR})
include_directories(${EXTERNAL_DIR}/LBFGSpp/include)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# List all source files
set(SOURCES
    ${SRC_DIR}/NelsonSiegel.cpp
    ${SRC_DIR}/Svensson.cpp
    ${SRC_DIR}/CubicSplineYieldCurve.cpp
    ${SRC_DIR}/Optimizers/GradientDescent.cpp
    ${SRC_DIR}/Optimizers/LevenbergMarquardt.cpp
    ${SRC_DIR}/Optimizers/BFGSOptimizer.cpp
    ${SRC_DIR}/Optimizers/NelderMead.cpp
)

# Create library target
add_library(yield_curve_lib STATIC ${SOURCES})
target_link_libraries(yield_curve_lib
    PUBLIC
    Eigen3::Eigen
)

# Build main executable
add_executable(YieldCurveFitting main.cpp)
target_link_libraries(YieldCurveFitting
    PRIVATE
    yield_curve_lib
)

# Enable Google Test
enable_testing()
