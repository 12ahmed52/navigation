cmake_minimum_required(VERSION 3.0.2)
project(navigation_prediction_model)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  ackermann_msgs
  nav_msgs
  
)

# Path where you installed your lib
set(NAV_ONNX_MODEL_DIR ../../../../install/libs/navigation_libs/navigation_onxx_runtime_prediction_model)
set(ONNXRUNTIME_PATH "../../../third_party_libs/onnxruntime/")


find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_PATH}/lib
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs ackermann_msgs nav_msgs
  INCLUDE_DIRS ${NAV_ONNX_MODEL_DIR}/include ${ONNXRUNTIME_PATH}/include
  LIBRARIES navigation_onxx_runtime_prediction_model
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${NAV_ONNX_MODEL_DIR}/include
  ${ONNXRUNTIME_PATH}/include
)

link_directories(${NAV_ONNX_MODEL_DIR}/libs)

add_executable(${PROJECT_NAME}_node
  src/main.cpp
  src/navigation_prediction_model_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  navigation_onxx_runtime_prediction_model
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)
