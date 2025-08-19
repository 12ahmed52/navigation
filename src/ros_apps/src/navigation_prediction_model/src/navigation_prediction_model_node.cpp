#include "navigation_prediction_model/navigation_prediction_model_node.hpp"

NavigationPredictionNode::NavigationPredictionNode(ros::NodeHandle& nh, const std::string& model_path)
    : m_nh(nh),
      m_predictor(model_path, 15, 7, 3)   // seq_len=15, feature_dim=7, num_layers=3
{
    m_ack_sub = m_nh.subscribe("/gem/ackermann_cmd", 1, &NavigationPredictionNode::ack_callback, this);
    m_odom_sub = m_nh.subscribe("/gem/base_footprint/odom", 1, &NavigationPredictionNode::odom_callback, this);

    // Publisher
    m_pred_pub = m_nh.advertise<std_msgs::Float32MultiArray>("/navigation_prediction", 1);

    ROS_INFO("NavigationPredictionNode initialized with model: %s", model_path.c_str());
}


void NavigationPredictionNode::ack_callback(const ackermann_msgs::AckermannDrive::ConstPtr& msg)
{
    m_latest_ack = *msg;
    m_has_ack = true;
}

void NavigationPredictionNode::odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    if (!m_has_ack) return; // wait until we have at least one ackermann message

    // ----------------------------
    // Extract state
    // ----------------------------
    float vx = msg->twist.twist.linear.x;
    float vy = msg->twist.twist.linear.y;
    float omega = msg->twist.twist.angular.z;

    float throttle_fb = m_latest_ack.speed / m_max_speed; // scale
    float steering_fb = m_latest_ack.steering_angle;

    // ----------------------------
    // Extract actions
    // ----------------------------
    float throttle_cmd = m_latest_ack.speed / m_max_speed; // scale
    float steering_cmd = m_latest_ack.steering_angle;

    // ----------------------------
    // Update predictor's history
    // ----------------------------
    m_predictor.set_inputs(vx, vy, omega, throttle_fb, steering_fb, throttle_cmd, steering_cmd);

    // ----------------------------
    // Run inference
    // ----------------------------
    auto prediction = m_predictor.run_inference(1); // returns std::vector<float>: [vx, vy, omega]

    // ----------------------------
    // Calculate dt from Odometry timestamp
    // ----------------------------
    ros::Time current_time = msg->header.stamp;
    float dt = 0.0f;
    if (!m_last_odom_time.isZero())
    {
        dt = (current_time - m_last_odom_time).toSec();
    }
    m_last_odom_time = current_time;

    // ----------------------------
    // Integrate to get x, y, theta (map frame)
    // ----------------------------
    m_pred_x += prediction[0] * dt;        // vx already in map frame
    m_pred_y += prediction[1] * dt;        // vy already in map frame
    m_pred_theta += prediction[2] * dt;    // yaw rate integration

    // Wrap theta to [-pi, pi]
    if (m_pred_theta > M_PI) m_pred_theta -= 2 * M_PI;
    if (m_pred_theta < -M_PI) m_pred_theta += 2 * M_PI;

    // ----------------------------
    // Publish prediction
    // ----------------------------
    std_msgs::Float32MultiArray pred_msg;
    pred_msg.data = prediction;           // vx, vy, omega
    pred_msg.data.push_back(m_pred_x);    // x
    pred_msg.data.push_back(m_pred_y);    // y
    pred_msg.data.push_back(m_pred_theta); // theta

    m_pred_pub.publish(pred_msg);

    ROS_INFO_STREAM("Prediction published: [vx=" << prediction[0]
                    << ", vy=" << prediction[1]
                    << ", omega=" << prediction[2]
                    << ", x=" << m_pred_x
                    << ", y=" << m_pred_y
                    << ", theta=" << m_pred_theta << "]");
}


void NavigationPredictionNode::spin()
{
    ros::spin();
}