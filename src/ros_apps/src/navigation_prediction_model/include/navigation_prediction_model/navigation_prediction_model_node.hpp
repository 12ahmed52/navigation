#pragma once

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <navigation_onxx_runtime_prediction_model/navigation_onxx_runtime_prediction_model.hpp>

#include <ackermann_msgs/AckermannDrive.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h> 

class NavigationPredictionNode
{
public:
    NavigationPredictionNode(ros::NodeHandle& nh, const std::string& model_path);
    void spin();

private:
    void ack_callback(const ackermann_msgs::AckermannDrive::ConstPtr& msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);

    ros::NodeHandle m_nh;
    ros::Subscriber m_ack_sub;
    ros::Subscriber m_odom_sub;
    ros::Publisher m_pred_pub;

    ros::Time m_last_odom_time;
    
    ackermann_msgs::AckermannDrive m_latest_ack;
    bool m_has_ack {false};
    ONNXInference m_predictor;

    float m_max_speed{15.0};

    float m_pred_x{0.0};
    float m_pred_y{0.0};
    float m_pred_theta{0.0};


};