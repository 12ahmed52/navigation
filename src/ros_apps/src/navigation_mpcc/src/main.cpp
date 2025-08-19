#include <ros/ros.h>
#include "navigation_mpcc/navigation_mpcc_node.hpp"

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nh("~");

    // Get configuration path from ROS parameter or use default
    std::string config_path;
    nh.param<std::string>("config_path", config_path, "/home/ashraf/MPCC/C++/Params/config.json");

    // Create MPC node instance
    mpc_ros_node::MPCNode mpc_node(nh, config_path);

    // Run the MPC node
    mpc_node.run();

    return 0;
}

