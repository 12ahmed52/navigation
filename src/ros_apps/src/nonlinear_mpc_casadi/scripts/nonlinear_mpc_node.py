#!/usr/bin/env python3
from casadi import *
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
import os
import time
import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nonlinear_mpc import MPC
import math 
from std_msgs.msg import Float32
import tf
# from osuf1_common.msg import MPC_metadata, MPC_trajectory, MPC_prediction

class ControllerManager:
    def __init__(self):
        rospy.init_node('controller_manager', anonymous=True)

        # Initialize ROS parameters
        try:
            self.param = {}
            self.param['dT'] = rospy.get_param('~dT', 0.2)
            self.param['N'] = rospy.get_param('~mpc_steps_N', 30)
            self.param['L'] = rospy.get_param('~vehicle_L', 1.545)  # wheelbase of M2
            self.param['theta_max'] = rospy.get_param('~mpc_max_steering', 30.0)
            self.param['v_max'] = rospy.get_param('~max_speed', 10.0)  # 5
            self.param['p_ref'] = rospy.get_param('~p_ref', 10.0)
            self.param['p_min'] = rospy.get_param('~p_min', 0.0)
            self.param['p_max'] = rospy.get_param('~p_max', 10.0)
            self.param['x_min'] = rospy.get_param('~x_min', -1500.0)
            self.param['x_max'] = rospy.get_param('~x_max', 1500.0)
            self.param['y_min'] = rospy.get_param('~y_min', -1500.0)
            self.param['y_max'] = rospy.get_param('~y_max', 1500.0)
            self.param['psi_min'] = rospy.get_param('~psi_min', -1000.0)
            self.param['psi_max'] = rospy.get_param('~psi_max', 1000.0)
            self.param['s_min'] = rospy.get_param('~s_min', 0.0)
            self.param['s_max'] = rospy.get_param('~s_max', 500000.0)
            self.param['ay_max'] = rospy.get_param('~ay_max', 8.0)
            self.param['a_min'] = rospy.get_param('~a_min', -10.5)
            self.param['a_max'] = rospy.get_param('~a_max', 8.5)
            self.param['omega_min'] = rospy.get_param('~omega_min', -0.2)
            self.param['omega_max'] = rospy.get_param('~omega_max', 0.2)
            self.param['jerk_init'] = rospy.get_param('~jerk_init', 5.0)
            self.param['mpc_w_cte'] = rospy.get_param('~mpc_w_cte', 10000.0)
            self.param['mpc_w_lag'] = rospy.get_param('~mpc_w_lag', 10000.0)
            self.param['mpc_w_s'] = rospy.get_param('~mpc_w_s', 70.0)
            self.param['mpc_w_a'] = rospy.get_param('~mpc_w_a', 2.0)
            self.param['mpc_w_omega'] = rospy.get_param('~mpc_w_omega', 0.0)
            self.param['mpc_w_p'] = rospy.get_param('~mpc_w_p', 5000.0)
            self.param['mpc_w_delta_a'] = rospy.get_param('~mpc_w_delta_a', 1000.0)
            self.param['mpc_w_delta_omega'] = rospy.get_param('~mpc_w_delta_omega', 100.0)
            self.param['mpc_w_delta_p'] = rospy.get_param('~mpc_w_delta_p', 5.0)
            self.param['INTEGRATION_MODE'] = rospy.get_param('~integration_mode', 'Euler')
            self.param['ipopt_verbose'] = rospy.get_param('~ipopt_verbose', True)
            
            self.CAR_WIDTH = rospy.get_param('~car_width', 2.3)
            self.min_speed_threshold = rospy.get_param('~min_speed_threshold', 0.7)
            self.DEBUG_MODE = rospy.get_param('~debug_mode', True)
            self.DELAY_MODE = rospy.get_param('~delay_mode', True)
            self.CONTROLLER_FREQ = rospy.get_param('~controller_freq', 15)
            self.GOAL_THRESHOLD = rospy.get_param('~goal_threshold', 0.6)
            self.INFLATION_FACTOR = rospy.get_param('~inflation_factor', 0.65)
            self.track_width = rospy.get_param('~track_width', 15.0)
            self.LAG_TIME = rospy.get_param('~lag_time', 0.15)
            self.ARC_LENGTH_MIN_DIST_TOL = rospy.get_param('~arc_length_min_dist_tol', 0.05)
            self.car_frame = rospy.get_param('~car_frame', 'base_link')
            
            cmd_vel_topic = rospy.get_param('~cmd_vel_topic_name', '/gem/ackermann_cmd')
            odom_topic = rospy.get_param('~odom_topic_name', '/gem/base_footprint/odom')
            goal_topic = rospy.get_param('~goal_topic_name', '/move_base_simple/goal')
            motor_topic = rospy.get_param('~motor_feedback_topic_name', '/feedback/motors_data')
            
            self.CENTER_TRACK_FILENAME = rospy.get_param('~x_y_waypoints_path_name', '/home/ashraf/navigation/src/ros_apps/src/POLARIS_GEM_e2/polaris_gem_drivers_sim/gem_pure_pursuit_sim/waypoints/xy.csv')
            self.CENTER_DERIVATIVE_FILENAME = rospy.get_param('~dx_dy_waypoints_path_name', '/home/ashraf/navigation/src/ros_apps/src/POLARIS_GEM_e2/polaris_gem_drivers_sim/gem_pure_pursuit_sim/waypoints/dxy.csv')
            
            rospy.loginfo("Parameters initialized successfully from ROS parameter server")
        except Exception as e:
            rospy.logwarn(f"Failed to initialize parameters: {e}, taking default values instead")
            
        self.param['theta_max'] = math.radians(self.param['theta_max'])  # convert to radians
        
        # Path related variables
        self.path_points = None
        self.center_lane = None
        self.center_point_angles = None
        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None

        # Publishers
        self.ackermann_pub = rospy.Publisher(cmd_vel_topic, AckermannDrive, queue_size=10)
        self.mpc_trajectory_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=10)
        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=10)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=10)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=10)
        self.center_tangent_pub = rospy.Publisher('/center_tangent', PoseStamped, queue_size=10)
        self.path_boundary_pub = rospy.Publisher('/boundary_marker', MarkerArray, queue_size=10)
        self.actual_trajectory_pub = rospy.Publisher('/actual_trajectory', Path, queue_size=10)
        self.track_distance_error_pub = rospy.Publisher('/track_distance_error', Float32, queue_size=10)

        # MPC related initializations
        self.p_ref_set_flag = False
        self.mpc = MPC()
        self.mpc.boundary_pub = self.path_boundary_pub
        self.initialize_MPC()
        # _, _, current_yaw = self.euler_from_quaternion((-2.1103628340227625e-06,-1.1107394429393641e-06,-0.7585388193784404, 0.6516278535255183))
        # self.current_pos_x, self.current_pos_y, self.current_yaw, self.current_s = -51.772918701171875, 44.86155319213867, current_yaw, 0.0
        # rospy.loginfo(f" yaw: {self.current_yaw}")
        self.current_pose = None
        self.current_vel_odom = 0.0
        self.previous_vel_odom = self.current_vel_odom
        self.projected_vel = 0.0
        self.steering_angle = 0.0
        self.a = 0.0
        self.omega = 0.0
        self.current_steering_feedback = 0.0
        self.current_omega_feedback = 0.0
        self.goal_pos = None
        self.goal_reached = False
        self.goal_received = False

        # Subscribers
        rospy.Subscriber(goal_topic, PoseStamped, self.goalCB)
        rospy.Subscriber(odom_topic, Odometry, self.odomCB)
        # rospy.Subscriber(motor_topic, MotorsData, self.motorfeedbackCB)

        self.time_odom1 = time.perf_counter()
        self.time_odom2 = self.time_odom1
        self.dt_odom = 0.0
        # self.time_steering1 = time.perf_counter()
        # self.time_steering2 = self.time_steering1
        # self.dt_steering = 0.0
        # Timer callback function for the control loop
        timer_period = 1.0 / self.CONTROLLER_FREQ

        # ROS 1 equivalent of create_timer
        rospy.Timer(rospy.Duration(timer_period), self.controlLoopCB)

        # Initialize actual_path
        self.actual_path = []


    def initialize_MPC(self):
        self.preprocess_track_data()
        self.param['s_max'] = self.element_arc_lengths[-1]
        self.mpc.set_initial_params(self.param)
        self.mpc.set_track_data(self.center_lut_x, self.center_lut_y, self.center_lut_dx, self.center_lut_dy,
                                self.right_lut_x, self.right_lut_y, self.left_lut_x, self.left_lut_y, 
                                self.element_arc_lengths[-1]) #was original arclength [-1]
        self.mpc.setup_MPC()


    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def find_nearest_index(self, car_pos):
        distances_array = np.linalg.norm(self.center_lane - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        return min_dist_idx, distances_array[min_dist_idx]
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w], dtype=np.float64)

    def heading(self, yaw):
        q = self.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    
    def euler_from_quaternion(self, quat):
        """
        Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw)
        using the 'sxyz' convention.
        """
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = self.euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        # if yaw < 0:
        #     yaw += 2 * math.pi

        return yaw

    def read_waypoints_array_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of numpy array'''
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            next(f) # skips first line because first line is x y lables in waypoints_xy and waypoints_derivatives
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = np.array([[float(point[0]), float(point[1])] for point in path_points])
        return path_points
        # return path_points # for carla or robot with full path

    # def pf_pose_callback(self, msg):
    #     '''acquire estimated pose of car from particle filter'''
    #     self.current_pos_x = msg.pose.position.x
    #     self.current_pos_y = msg.pose.position.y
    #     self.current_yaw = self.quaternion_to_euler_yaw(msg.pose.orientation)
    #     self.current_pose = [self.current_pos_x, self.current_pos_y, self.current_yaw]
    #     self.actual_path.append(msg)
    #     self.publish_actual_path()
    #     if self.goal_received:
    #         car2goal_x = self.goal_pos.x - self.current_pos_x
    #         car2goal_y = self.goal_pos.y - self.current_pos_y
    #         dist2goal = sqrt(car2goal_x * car2goal_x + car2goal_y * car2goal_y)
    #         if dist2goal < self.GOAL_THRESHOLD:
    #             self.goal_reached = True
    #             self.goal_received = False
    #             self.mpc.WARM_START = False
    #             self.mpc.init_mpc_start_conditions()
    #             self.get_logger().info("Goal Reached !")
    #             self.plot_data()

    # def motorfeedbackCB(self, msg):

    #     self.current_steering_feedback = math.radians((msg.front_left.steering_angle + msg.front_right.steering_angle)/2)
    #     self.current_omega_feedback =  math.radians((msg.front_left.steering_angular_velocity + msg.front_right.steering_angular_velocity)/2)

    def odomCB(self, msg):
        """Get odometry data especially velocity from the car"""
        # --- timing ---
        self.time_odom2 = time.perf_counter()
        self.dt_odom = self.time_odom2 - self.time_odom1
        self.time_odom1 = self.time_odom2

        # --- velocities ---
        self.previous_vel_odom = self.current_vel_odom
        self.current_vel_odom = msg.twist.twist.linear.x

        # --- position ---
        self.current_pos_x = msg.pose.pose.position.x
        self.current_pos_y = msg.pose.pose.position.y

        # --- yaw from quaternion ---
        self.current_yaw = self.quaternion_to_euler_yaw(msg.pose.pose.orientation)

        # --- store pose ---
        self.current_pose = [self.current_pos_x, self.current_pos_y, self.current_yaw]

        # --- path publishing ---
        pose_stamped = PoseStamped()
        pose_stamped.header = self.create_header("map")
        pose_stamped.pose = msg.pose.pose
        self.actual_path.append(pose_stamped)
        self.publish_actual_path()

        # --- TF broadcaster (map -> base_footprint) ---
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (self.current_pos_x, self.current_pos_y, 0.0),  # translation
            (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ),  # orientation as quaternion
            rospy.Time.now(),
            "base_footprint",  # child frame
            "map",             # parent frame
        )

    def goalCB(self, msg):
        '''Get goal pose from the user'''
        self.goal_pos = msg.pose.position
        self.goal_received = True
        self.goal_reached = False
        if self.DEBUG_MODE:
            print("Goal pos=", self.goal_pos)

    def publish_actual_path(self):

        '''Publish the actual path taken by the car'''
        path = Path()
        path.header = self.create_header('map')
        path.poses = self.actual_path
        self.actual_trajectory_pub.publish(path)

    def publish_path(self, waypoints, publisher):
        # Visualize path derived from the given waypoints in the path
        path = Path()
        path.header = self.create_header('map')
        path.poses = []
        for point in waypoints:
            tempPose = PoseStamped()
            tempPose.header = path.header
            tempPose.pose.position.x = point[0]
            tempPose.pose.position.y = point[1]
            tempPose.pose.orientation.w = 1.0
            path.poses.append(tempPose)
        publisher.publish(path)

    # def get_interpolated_path(self, pts, arc_lengths_arr, smooth_value=0.1, scale=2, derivative_order=0):
    #     # tck represents vector of knots, the B-spline coefficients, and the degree of the spline.
    #     tck, u = splprep(pts.T, u=arc_lengths_arr, s=smooth_value, per=1)
    #     u_new = np.linspace(u.min(), u.max(), len(pts) * scale)
    #     x_new, y_new = splev(u_new, tck, der=derivative_order)
    #     interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
    #     return interp_points, tck

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = interpolant(label_x, 'bspline', [u], V_X)
        lut_y = interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x, lut_y

    def get_arc_lengths(self, waypoints):
        d = np.diff(waypoints, axis=0)
        consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
        dists_cum = np.cumsum(consecutive_diff)
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum
    
    def create_track_boundaries(self, center_lane, track_width):
        right_bound=np.zeros((len(center_lane),2))
        left_bound=np.zeros((len(center_lane),2))
        for idx in range(len(center_lane)-1):
            width_vector = np.array([-(center_lane[idx+1, 1]-center_lane[idx, 1]), center_lane[idx+1, 0]-center_lane[idx, 0]])
            width_unit_vector = width_vector/np.linalg.norm(width_vector)
            right_bound[idx,:] = center_lane[idx,:] - width_unit_vector*track_width/2
            left_bound[idx,:] = center_lane[idx,:] + width_unit_vector*track_width/2
        right_bound[len(center_lane)-1,:] = center_lane[len(center_lane)-1,:] - width_unit_vector*track_width/2
        left_bound[len(center_lane)-1,:] = center_lane[len(center_lane)-1,:] + width_unit_vector*track_width/2
        # right_bound[len(center_lane)-1,:] = right_bound[0,:]
        # left_bound[len(center_lane)-1,:] = left_bound[0,:]
        return right_bound , left_bound


    def inflate_track_boundaries(self, center_lane, side_lane, car_width, inflation_factor):
        for idx in range(len(center_lane)):
            lane_vector = side_lane[idx, :] - center_lane[idx, :]
            side_track_width = np.linalg.norm(lane_vector)
            side_unit_vector = lane_vector / side_track_width
            side_lane[idx, :] = center_lane[idx, :] + side_unit_vector * (
                    side_track_width - car_width * inflation_factor)
        return side_lane

    def preprocess_track_data(self):
        center_lane = self.read_waypoints_array_from_csv(self.CENTER_TRACK_FILENAME)
        center_derivative_data = self.read_waypoints_array_from_csv(self.CENTER_DERIVATIVE_FILENAME)
        # right_lane = self.read_waypoints_array_from_csv(self.RIGHT_TRACK_FILENAME)
        # left_lane = self.read_waypoints_array_from_csv(self.LEFT_TRACK_FILENAME)
        print("CENTER:_LANE: ",center_lane.size)

        right_lane, left_lane = self.create_track_boundaries(center_lane, self.track_width)



        for i in range(5):
            self.publish_path(center_lane, self.center_path_pub)
            self.publish_path(right_lane, self.right_path_pub)
            self.publish_path(left_lane, self.left_path_pub)
            time.sleep(0.2)

        right_lane = self.inflate_track_boundaries(center_lane, right_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)
        left_lane = self.inflate_track_boundaries(center_lane, left_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)

        # self.center_lane = np.row_stack((center_lane, center_lane[1:int(center_lane.shape[0] / 2), :]))
        # right_lane = np.row_stack((right_lane, right_lane[1:int(center_lane.shape[0] / 2), :]))
        # left_lane = np.row_stack((left_lane, left_lane[1:int(center_lane.shape[0] / 2), :]))
        # center_derivative_data = np.row_stack(
        #     (center_derivative_data, center_derivative_data[1:int(center_lane.shape[0] / 2), :]))
        self.center_lane = center_lane


        # Interpolate center line upto desired resolution
        # self.element_arc_lengths_orig = self.get_arc_lengths(center_lane)
        self.element_arc_lengths = self.get_arc_lengths(self.center_lane)
        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y',
                                                                                 self.center_lane,
                                                                                 self.element_arc_lengths)
        self.center_lut_dx, self.center_lut_dy = self.get_interpolated_path_casadi('lut_center_dx', 'lut_center_dy',
                                                                                   center_derivative_data,
                                                                                   self.element_arc_lengths)
        self.center_point_angles = np.arctan2(center_derivative_data[:, 1], center_derivative_data[:, 0])

        # Interpolate right and left wall line
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_lane,
                                                                               self.element_arc_lengths)
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_lane,
                                                                             self.element_arc_lengths)


    def find_current_arc_length(self, car_pos):
        nearest_index, minimum_dist = self.find_nearest_index(car_pos)
        if minimum_dist > self.ARC_LENGTH_MIN_DIST_TOL:
            # if nearest_index == 0:
            #     next_idx = 1
            #     prev_idx = self.center_lane.shape[0] - 1
            # elif nearest_index == (self.center_lane.shape[0] - 1):
            #     next_idx = 0
            #     prev_idx = self.center_lane.shape[0] - 2
            if nearest_index != 0 and nearest_index != (self.center_lane.shape[0] - 1):
                next_idx = nearest_index + 1
                prev_idx = nearest_index - 1
                dot_product_value = np.dot(car_pos - self.center_lane[nearest_index, :],
                                        self.center_lane[prev_idx, :] - self.center_lane[nearest_index, :])
                if dot_product_value > 0:   # project on the vector of the previous 2 points
                    nearest_index_actual = prev_idx
                else:   # project on the vector of the next 2 points
                    nearest_index_actual = nearest_index
                    nearest_index = next_idx
            elif nearest_index == 0: # project on the vector of the next 2 points
                next_idx = nearest_index + 1
                nearest_index_actual = nearest_index
                nearest_index = next_idx
            else: # nearest_index == (self.center_lane.shape[0] - 1)
                prev_idx = nearest_index - 1
                nearest_index_actual = prev_idx # project on the vector of the previous 2 points

            new_dot_value = np.dot(car_pos - self.center_lane[nearest_index_actual, :],
                                   self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            projection = new_dot_value / np.linalg.norm(
                self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            current_s = self.element_arc_lengths[nearest_index_actual] + projection
            if current_s >  self.element_arc_lengths[-1]:
                current_s = self.element_arc_lengths[-1]
            elif current_s < 0.0:
                current_s = 0.0
            
            nearest_index = nearest_index_actual
                

        else:
            current_s = self.element_arc_lengths[nearest_index]

        # if nearest_index==0:
        #     current_s=0.0
        return current_s, nearest_index


    def controlLoopCB(self, event):
        '''Control loop for car MPC'''
        if self.goal_received and not self.goal_reached:
            control_loop_start_time = time.time()
            # Update system states: X=[x, y, psi,v, theta]
            px = self.current_pos_x
            py = self.current_pos_y
            car_pos = np.array([self.current_pos_x, self.current_pos_y])
            psi = self.current_yaw
            v = self.current_vel_odom
            dt_odom = self.dt_odom
            steering = self.steering_angle  # radian # for carla testing
            # steering = self.current_steering_feedback # for robot testing
           

            if dt_odom >= 1e-6:
                a = (self.current_vel_odom - self.previous_vel_odom)/dt_odom           # TODO: for the robot we need to take it from sensor or kalman filter
            else:
                a = self.a
            rospy.loginfo("dt_odom = %.4f", self.dt_odom)
            omega = self.omega                      # for carla testing
            
            # omega = self.current_omega_feedback # for robot testing

            # # Update system inputs: U=[a, omega]
            # v = self.current_vel_odom
            # steering = self.steering_angle  # radian
            L = self.mpc.L

            current_s, near_idx = self.find_current_arc_length(car_pos)

            # Compute current track error:
            track_error = sqrt((px-self.center_lut_x(current_s))**2 + (py-self.center_lut_y(current_s))**2)

            # print "pre",current_s,near_idx
            if self.DELAY_MODE:
                dt_lag = self.LAG_TIME
                px = px + v * np.cos(psi) * dt_lag
                py = py + v * np.sin(psi) * dt_lag
                # psi = psi + (2* v / L) * tan(steering) * dt_lag     # for robot testing
                psi = psi + (v / L) * tan(steering) * dt_lag        # For carla testing
                current_s = current_s + self.projected_vel * dt_lag
                # TODO: check if it needs update before or after the other states
                v = v + a * dt_lag
                steering = steering + omega * dt_lag

                
            current_state = np.array([px, py, psi, current_s, v, steering])

            centerPose = PoseStamped()
            centerPose.header = self.create_header('map')
            centerPose.pose.position.x = float(self.center_lane[near_idx, 0])
            centerPose.pose.position.y = float(self.center_lane[near_idx, 1])
            centerPose.pose.orientation = self.heading(self.center_point_angles[near_idx])
            self.center_tangent_pub.publish(centerPose)

            # Check distance to goal and change p_max accordingly
            distance_to_goal = abs(self.param['s_max'] - current_s)
            # if distance_to_goal <= (-0.5 * v * v / (self.param['a_min']*0.8)):
            # if abs(a)>1e-6:
            #     if distance_to_goal <= (-0.5 * v * v / a):
            #         self.get_logger().info(f"Distance to goal: {distance_to_goal} m")
            #         self.get_logger().info("Close to goal. Setting p_ref to 0.0!")
            #         self.param['p_ref'] = 0.0
            #         self.mpc.set_initial_params(self.param)
            # else:
            # if distance_to_goal <= (-0.5 * v * v / (self.param['a_min']*0.8)) and self.p_ref_set_flag != True:

            # if distance_to_goal <= (-0.5 * self.param['v_max']**2 / (self.param['a_min']*0.8)) and self.p_ref_set_flag != True:
            #     rospy.loginfo("Distance to goal: %f m and limit is %f m", 
            #                 distance_to_goal, 
            #                 (-0.5 * self.param['v_max']**2 / (self.param['a_min']*0.8)))

                # # rospy.loginfo("Close to goal (if a = 0.8*a_min). Setting p_ref to 0.0!")
                # self.param['p_ref'] = 0.0
                # self.mpc.update_p_ref(self.param)
                # self.p_ref_set_flag = True

            # Solve MPC Problem
            mpc_time = time.time()
            first_control, trajectory, control_inputs, v_theta_first = self.mpc.solve(current_state)
            mpc_compute_time = time.time() - mpc_time

            v_debug = trajectory[:, 4]
            a_debug = control_inputs[:, 0]
            a_expecteddebug = np.diff(v_debug) / 0.2
            omega_debug = control_inputs[:, 1]

            # MPC result (all described in car frame)
            # speed = float(first_control[0])  # speed
            # steering = float(first_control[1])  # radian
            # self.projected_vel = speed # TODO: check if it was correct

            # self.a = float(first_control[0])
            self.a = control_inputs[0, 0]
            self.omega = float(first_control[1])
            self.projected_vel = float(first_control[2]) # or take the speed #TODO check which is correct
            speed = trajectory[1,4]
            # speed = float(v_theta_first[0])
            steering = float(v_theta_first[1])

            #throttle calculation
            # throttle = 0.03*(speed - v)/ self.param['dT']

            # if throttle>1:
            #     throttle=1
            # elif throttle<-1:
            #     throttle=-1
            # if speed ==0:
            #     throttle=0

            if not self.mpc.WARM_START:
                # speed, steering,throttle = 0.0, 0.0, 0.0
                speed, steering, self.a, self.omega = 0.0, 0.0, 0.0, 0.0
                self.mpc.WARM_START = True
            if (speed >= self.param['v_max']): #not sure why needed since it is in constraint
                speed = self.param['v_max']
                rospy.logwarn("V is saturating and the mpc is violating its constraint!")

            elif (speed <= 0.05):
                speed = 0.0
            elif (speed > 0.05 and speed <= self.min_speed_threshold):
                speed = self.min_speed_threshold
            
            # elif (speed <= (- self.param['v_max'] / 2.0)):
            #     speed = - self.param['v_max'] / 2.0

            # Display the MPC predicted trajectory
            speed = 5.0
            mpc_traj = Path()
            mpc_traj.header = self.create_header('map')
            mpc_traj.poses = []
            for i in range(trajectory.shape[0]):
                tempPose = PoseStamped()
                tempPose.header = mpc_traj.header
                tempPose.pose.position.x = trajectory[i, 0]
                tempPose.pose.position.y = trajectory[i, 1]
                tempPose.pose.orientation = self.heading(trajectory[i, 2])
                mpc_traj.poses.append(tempPose)
            self.mpc_trajectory_pub.publish(mpc_traj)

            # Publish the track distance error
            dist_error = Float32()
            dist_error.data = float (track_error)
            self.track_distance_error_pub.publish(dist_error)


            total_time = time.time() - control_loop_start_time
            if self.DEBUG_MODE:
                # self.get_logger().info("DEBUG")
                # self.get_logger().info(f"a_debug: {a_debug}")
                # self.get_logger().info(f"a_expecteddebug: {a_expecteddebug}")
                # self.get_logger().info(f"v_debug: {v_debug}")
                # self.get_logger().info(f"current_v_lag: {current_state[4]}")
                # self.get_logger().info(f"omega_debug: {omega_debug}")
                rospy.loginfo(f"psi: {psi}")
                rospy.loginfo(f"V: {speed}")
                rospy.loginfo(f"Steering: {steering}")
                rospy.loginfo(f"Acceleration: {self.a}")
                rospy.loginfo(f"Steering rate: {self.omega}")
                rospy.loginfo(f"Control loop time mpc= {mpc_compute_time}")
                rospy.loginfo(f"Control loop time= {total_time}")

            # self.current_time += 1.0 / self.CONTROLLER_FREQ
            # # self.cte_plot.append(cte)
            # self.t_plot.append(self.current_time)
            # self.v_plot.append(speed)
            # self.steering_plot.append(np.rad2deg(steering))
            # self.time_plot.append(mpc_compute_time * 1000)
        else:
            steering = 0.0
            speed = 0.0
            self.a = 0.0
            self.omega  = 0.0
            # throttle=0.0

        # publish cmd
        # for carla testing uncomment this below
        # ackermann_cmd = AckermannDriveStamped()     
        # ackermann_cmd.header = self.create_header(self.car_frame)   
        # ackermann_cmd.drive.steering_angle = steering
        # ackermann_cmd.drive.steering_angle_velocity = self.omega
        # self.steering_angle = steering
        # ackermann_cmd.drive.speed = speed
        # ackermann_cmd.drive.acceleration = self.a
        # # if self.THROTTLE_MODE:
        # #     ackermann_cmd.drive.acceleration = throttle
        # self.ackermann_pub.publish(ackermann_cmd)

        # for robot testing uncomment this below
        ackermann_cmd = AckermannDrive()      
        ackermann_cmd.steering_angle = steering
        ackermann_cmd.steering_angle_velocity = self.omega
        self.steering_angle = steering
        ackermann_cmd.speed = speed
        ackermann_cmd.acceleration = self.a
        # if self.THROTTLE_MODE:
        #     ackermann_cmd.drive.acceleration = throttle
        self.ackermann_pub.publish(ackermann_cmd)






if __name__ == '__main__':
    mpc_node = ControllerManager()
    rospy.spin()
