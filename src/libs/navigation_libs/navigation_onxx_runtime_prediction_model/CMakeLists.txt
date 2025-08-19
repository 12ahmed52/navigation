message("############# Navigation OnxxRuntime Prediction Model Lib Cmake ##################")

set(ONNXRUNTIME_PATH "../../../third_party_libs/onnxruntime/")

add_library(navigation_onxx_runtime_prediction_model SHARED src/navigation_onxx_runtime_prediction_model.cpp)


find_library(ONNXRUNTIME_LIB
NAMES onnxruntime
PATHS ${ONNXRUNTIME_PATH}/lib
)

target_include_directories(navigation_onxx_runtime_prediction_model PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${ONNXRUNTIME_PATH}/include/
  )

target_link_libraries(navigation_onxx_runtime_prediction_model PRIVATE
    ${ONNXRUNTIME_LIB}
    pthread dl
)

INSTALL(TARGETS navigation_onxx_runtime_prediction_model
        LIBRARY DESTINATION ${DCMAKE_INSTALL_PREFIX}/navigation_libs/navigation_onxx_runtime_prediction_model/libs)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${DCMAKE_INSTALL_PREFIX}/navigation_libs/navigation_onxx_runtime_prediction_model/include/navigation_onxx_runtime_prediction_model)

        