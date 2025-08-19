#pragma once

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <ackermann_msgs/AckermannDrive.h>

#include "navigation_mpcc/mpc.h"
#include "navigation_mpcc/integrator.h"
#include "navigation_mpcc/track.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <tf/tf.h>
#include <nav_msgs/Path.h>

#include <nlohmann/json.hpp>

#include "navigation_mpcc/plotting.h"

using json = nlohmann::json;

namespace mpc_ros_node {

class MPCNode {
public:
    MPCNode(const ros::NodeHandle& nh, const std::string& config_path);
    // ~MPCNode();

    void run();

private:
    // === Helper functions ===
    void setupMPC(const std::string& config_path);
    void setupTrack();

    mpcc::State odomToState(const nav_msgs::Odometry::ConstPtr& msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void pathCallback(const nav_msgs::Path::ConstPtr& msg);

    // === ROS members ===
    ros::NodeHandle nh_;
    ros::Publisher drive_pub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber path_sub_;

    // === Config ===
    std::string config_path_;
    json jsonConfig_;

    // === MPC components ===
    std::unique_ptr<mpcc::MPC> mpc_;
    std::unique_ptr<mpcc::Integrator> integrator_;
    std::unique_ptr<mpcc::Track> track_;
    mpcc::TrackPos track_xy_;

    double current_speed_ = 0.0;
    double current_steering_ = 0.0;

    bool initialized_speed_ = false;
    bool pathSet;
    ros::Time last_time_;
    std::list<mpcc::MPCReturn> log;
    std::unique_ptr<mpcc::Plotting> plotter;
    mpcc::TrackPos track_xy;
    mpcc::State x0;
    

};

} // namespace mpc_ros_node
