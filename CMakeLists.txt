cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "navigation_assignment")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_BUILD_TYPE Debug)
 
set(CMAKE_ENABLE_EXPORTS ON)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-export-dynamic")
set(PARENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(DCMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install/libs)
set(THIRD_PARTY_LIBS_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/third_party_libs)
add_compile_options(-O3)
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Ofast>)


project("${PROJECT_NAME}"
    VERSION 0.1
    DESCRIPTION "navigation_assignment"
    LANGUAGES C CXX)


##################### Build Navigation Libs ##########################
add_subdirectory(src/libs/navigation_libs)

