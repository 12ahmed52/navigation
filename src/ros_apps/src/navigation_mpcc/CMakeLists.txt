cmake_minimum_required(VERSION 3.0.2)
project(navigation_mpcc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  ackermann_msgs
  nav_msgs
  
)

# Path where you installed your lib
set(NAV_MPCC_PATH ../../../../install/libs/navigation_libs/navigation_mpcc)
set(JSON_PATH "../../../third_party_libs/Json/")
set(BLASFEO_PATH "../../../third_party_libs/blasfeo/")
set(HPIPM_PATH "../../../third_party_libs/hpipm/")
set(EIGEN_PATH "../../../third_party_libs/Eigen/")
set(MATPLOT_PATH "../../../third_party_libs/matplotlib/")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_library(NAV_MPCC_LIB
  NAMES navigation_mpcc
  PATHS ${NAV_MPCC_PATH}/libs/
)

find_library(BLASFEO_LIB
    NAMES blasfeo
    PATHS ${BLASFEO_PATH}/lib/lib/
)

find_library(HPIPM_LIB
    NAMES hpipm
    PATHS ${HPIPM_PATH}/lib/lib/
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs ackermann_msgs nav_msgs
  INCLUDE_DIRS ${NAV_MPCC_PATH}/include 
  LIBRARIES navigation_mpcc
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${NAV_MPCC_PATH}/include/
  ${HPIPM_PATH}/include/
  ${EIGEN_PATH}/
  ${JSON_PATH}/include/
  ${BLASFEO_PATH}/include/
  ${MATPLOT_PATH}/
  ${Python3_INCLUDE_DIRS}
)

link_directories(${NAV_MPCC_PATH}/libs)

add_executable(${PROJECT_NAME}_node
  src/main.cpp
  src/navigation_mpcc_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  navigation_mpcc
  ${catkin_LIBRARIES}
  ${NAV_MPCC_LIB}
  ${BLASFEO_LIB}
  ${HPIPM_LIB}
  ${Python3_LIBRARIES}
)
