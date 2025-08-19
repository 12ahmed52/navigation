#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
import csv

class DataRecorder:
    def __init__(self):
        self.latest_ack = None

        self.csv_file = open('', 'w', newline='') #Add csv file name
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'x', 'y',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
            'vx', 'vy', 'omega',
            'input_velocity_command', 'input_steering_command'
        ])

        rospy.Subscriber('/gem/ackermann_cmd', AckermannDrive, self.ack_callback)
        rospy.Subscriber('/gem/base_footprint/odom', Odometry, self.odom_callback)

    def ack_callback(self, msg):
        self.latest_ack = msg

    def odom_callback(self, msg):
        if self.latest_ack is None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        quat_x = q.x
        quat_y = q.y
        quat_z = q.z
        quat_w = q.w

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        input_velocity_command = self.latest_ack.speed
        input_steering_command = self.latest_ack.steering_angle

        self.csv_writer.writerow([
            x, y,
            quat_x, quat_y, quat_z, quat_w,
            vx, vy, omega,
            input_velocity_command, input_steering_command
        ])
        self.csv_file.flush()

def main():
    rospy.init_node('sim_data_recorder', anonymous=True)
    recorder = DataRecorder()
    rospy.spin()
    recorder.csv_file.close()

if __name__ == '__main__':
    main()
