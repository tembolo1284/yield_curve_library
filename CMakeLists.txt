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
include_directories(${EXTERNAL_DIR}/ceres-solver/install/include)
include_directories(${EXTERNAL_DIR}/abseil-cpp/install/include)

# Explicitly set Abseil path
set(Abseil_DIR "${EXTERNAL_DIR}/abseil-cpp/install/lib/cmake/absl")
find_package(absl REQUIRED)

# Explicitly set Ceres Solver path
set(Ceres_DIR "${EXTERNAL_DIR}/ceres-solver/install/lib/cmake/Ceres")
find_package(Ceres REQUIRED)

# Find other dependencies
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
    ${SRC_DIR}/Optimizers/CeresOptimizer.cpp
)

# Build main executable
add_executable(YieldCurveFitting main.cpp ${SOURCES})

# Link required libraries
target_link_libraries(YieldCurveFitting Eigen3::Eigen Ceres::ceres absl::log absl::check absl::flags)

# Enable Google Test
enable_testing()

add_executable(TestYieldCurveModels test/TestYieldCurveModels.cpp)
target_link_libraries(TestYieldCurveModels PRIVATE GTest::GTest GTest::Main Eigen3::Eigen Ceres::ceres absl::log absl::check absl::flags)

add_executable(TestOptimizers test/TestOptimizers.cpp)
target_link_libraries(TestOptimizers PRIVATE GTest::GTest GTest::Main Eigen3::Eigen Ceres::ceres absl::log absl::check absl::flags)

# Register tests
add_test(NAME YieldCurveModels COMMAND TestYieldCurveModels)
add_test(NAME Optimizers COMMAND TestOptimizers)

