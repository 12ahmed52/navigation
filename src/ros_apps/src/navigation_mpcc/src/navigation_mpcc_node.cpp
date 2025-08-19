#include "navigation_mpcc/navigation_mpcc_node.hpp"

#include <fstream>

namespace mpc_ros_node
{

MPCNode::MPCNode(const ros::NodeHandle& nh, const std::string& config_path)
    : nh_(nh), config_path_(config_path)
{
    pathSet = false;
    // Load configuration
    setupMPC(config_path_);

    // Setup publishers and subscribers
    drive_pub_ = nh_.advertise<ackermann_msgs::AckermannDrive>("/gem/ackermann_cmd", 1);
    odom_sub_  = nh_.subscribe("/gem/base_footprint/odom", 1, &MPCNode::odomCallback, this);
    path_sub_ = nh_.subscribe("/navigation/path", 1, &MPCNode::pathCallback, this);


    ROS_INFO("MPCNode initialized successfully");
}

// MPCNode::~MPCNode()
// {
//     if (plotter) {
//         plotter->plotRun(log, track_xy);
//         plotter->plotSim(log,track_xy);
//         ROS_INFO("Final MPC log plotted at shutdown.");
//     }
// }


void MPCNode::setupMPC(const std::string& config_path)
{
    std::ifstream iConfig("/home/ashraf/MPCC/C++/Params/config.json");
    json jsonConfig;
    iConfig >> jsonConfig;
    mpcc::PathToJson json_paths {jsonConfig["model_path"],
                           jsonConfig["cost_path"],
                           jsonConfig["bounds_path"],
                           jsonConfig["track_path"],
                           jsonConfig["normalization_path"]};
    integrator_ = std::make_unique<mpcc::Integrator>(jsonConfig["Ts"],json_paths);
    plotter = std::make_unique<mpcc::Plotting>(jsonConfig["Ts"],json_paths);

    // mpcc::Track track = mpcc::Track(json_paths.track_path);
    // mpcc::TrackPos track_xy = track.getTrack();

    mpc_ = std::make_unique<mpcc::MPC>(jsonConfig["n_sqp"],jsonConfig["n_reset"],jsonConfig["sqp_mixing"],jsonConfig["Ts"],json_paths);
}


mpcc::State MPCNode::odomToState(const nav_msgs::Odometry::ConstPtr& msg)
{
    mpcc::State x;
    x.X  = msg->pose.pose.position.x;
    x.Y  = msg->pose.pose.position.y;
    x.phi = tf::getYaw(msg->pose.pose.orientation);
    x.vx  = std::sqrt(std::pow(msg->twist.twist.linear.x, 2.0) + std::pow(msg->twist.twist.linear.y, 2.0));
    x.vy  = 0.0;
    x.r = msg->twist.twist.angular.z;
    return x;
}

void MPCNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    mpcc::State x0 = odomToState(msg);

    // std::cout<<"_____________________________________"<<std::endl;
    // std::cout<<"X0: "<<x0.X<<", "<<x0.Y<<std::endl;
    // std::cout<<"ODOM: "<<msg->pose.pose.position.x<<", "<<msg->pose.pose.position.y<<std::endl;
    // std::cout<<"_____________________________________"<<std::endl;

    // Run MPC only if path is set
    if (!pathSet)
        return;

    mpcc::MPCReturn mpc_sol = mpc_->runMPC(x0);
    // std::cout<<"MPC OUT: "<<mpc_sol.u0.dDelta<<", "<<mpc_sol.u0.dD<<std::endl;
    log.push_back(mpc_sol);
    

    // Extract control input
    double steering_rate = mpc_sol.u0.dDelta;   // steering rate
    double acceleration  = mpc_sol.u0.dD;   // longitudinal force -> acceleration

    // Initialize current speed and steering angle from odometry if first time
    if (!initialized_speed_) {
        current_speed_ = msg->twist.twist.linear.x;
        current_steering_ = 0.0;  // assume starting at 0 if no initial steering
        initialized_speed_ = true;
    }

    // Compute timestep dt from odom stamps
    ros::Time now = msg->header.stamp;
    if (!last_time_.isZero()) {
        double dt = (now - last_time_).toSec();
        if (dt > 0.0) {
            // Integrate acceleration
            current_speed_ += acceleration * dt;
            if (current_speed_ < 0.0) current_speed_ = 0.0;
            if (current_speed_ > 5.0) current_speed_ = 5.0;

            // Integrate steering rate to get steering angle
            current_steering_ += steering_rate * dt;

            // Optionally limit steering angle to physical limits
            double max_steering = 0.4189; // ~24 degrees in radians
            if (current_steering_ > max_steering) current_steering_ = max_steering;
            if (current_steering_ < -max_steering) current_steering_ = -max_steering;
        }
    }
    last_time_ = now;

    // Publish AckermannDrive command
    ackermann_msgs::AckermannDrive cmd;
    cmd.steering_angle = current_steering_;
    cmd.speed          = current_speed_;

    drive_pub_.publish(cmd);

}



void MPCNode::run()
{
    ros::spin();
}

void MPCNode::pathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    std::cout<<"PAAAAAATTTTTTTTTHHHHHHHHHH!!!!!!!!!!!!!!!!"<<std::endl;
    // Extract X and Y coordinates from the path
    pathSet = true;
    int n = msg->poses.size();
    Eigen::VectorXd X(n);
    Eigen::VectorXd Y(n);

    for (int i = 0; i < n; ++i) {
        X[i] = msg->poses[i].pose.position.x;
        Y[i] = msg->poses[i].pose.position.y;
    }

    // Update the MPC track
    if (mpc_) {
        mpc_->setTrack(X, Y);
        ROS_INFO("MPC track updated with %d waypoints", n);
    }
    track_xy.X = X;
    track_xy.Y = Y;
}



} // namespace mpc_ros_node
