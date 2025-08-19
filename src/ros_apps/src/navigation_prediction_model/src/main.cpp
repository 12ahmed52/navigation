#include "navigation_prediction_model/navigation_prediction_model_node.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "navigation_onxx_runtime_prediction_node");
    ros::NodeHandle nh("~");

    // Get model path from ROS param or default
    std::string model_path;
    nh.param<std::string>("model_path", model_path, "/home/ashraf/navigation_assignment/models/model.onnx");

    NavigationPredictionNode node(nh, model_path);
    node.spin();

    return 0;
}