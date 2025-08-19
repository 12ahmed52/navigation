#include "navigation_onxx_runtime_prediction_model.hpp"

int main()
{
    ONNXInference model("mpc_dynamic_model.onnx", 15, 7, 3);
    model.run_inference(5); // run 5 iterations
    return 0;
}