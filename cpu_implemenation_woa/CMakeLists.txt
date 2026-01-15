cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)

project(CPU_WOA)

add_executable(${CMAKE_PROJECT_NAME})

file(GLOB SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

target_sources(${PROJECT_NAME} PRIVATE ${SRC_FILES})

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/headers/)

target_compile_options(${PROJECT_NAME} PRIVATE -static -g)
