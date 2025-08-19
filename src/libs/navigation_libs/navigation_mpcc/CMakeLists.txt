message("############# Navigation MPCC Lib Cmake ##################")

set(HPIPM_PATH "../../../third_party_libs/hpipm/")
set(EIGEN_PATH "../../../third_party_libs/Eigen/")
set(JSON_PATH "../../../third_party_libs/Json/")
set(BLASFEO_PATH "../../../third_party_libs/blasfeo/")
set(MATPLOT_PATH "../../../third_party_libs/matplotlib/")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB SOURCES "src/*.cpp")
add_library(navigation_mpcc SHARED ${SOURCES})

# HPIPM library (binary)
find_library(HPIPM_LIB
    NAMES hpipm
    PATHS ${HPIPM_PATH}/lib/lib/
)

find_library(BLASFEO_LIB
    NAMES blasfeo
    PATHS ${BLASFEO_PATH}/lib/lib/
)

# Include directories
target_include_directories(navigation_mpcc PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${HPIPM_PATH}/include/
  ${EIGEN_PATH}/
  ${JSON_PATH}/include/
  ${BLASFEO_PATH}/include/
  ${MATPLOT_PATH}/
   ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(navigation_mpcc PRIVATE
    ${HPIPM_LIB}
    ${BLASFEO_LIB}
    ${Python3_LIBRARIES}
)

# Install targets
INSTALL(TARGETS navigation_mpcc
        LIBRARY DESTINATION ${DCMAKE_INSTALL_PREFIX}/navigation_libs/navigation_mpcc/libs)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${DCMAKE_INSTALL_PREFIX}/navigation_libs/navigation_mpcc/include/navigation_mpcc)

