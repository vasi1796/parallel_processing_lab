cmake_minimum_required(VERSION 3.5)
project(cuda_reduction LANGUAGES CXX CUDA)

include_directories(../utilities)

find_package(CUDA REQUIRED)

set(SOURCES src/main.cu)

cuda_compile(CUDA_FILES ${SOURCES})

cuda_add_executable(${PROJECT_NAME} ${CUDA_FILES} OPTIONS -arch sm_35)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})