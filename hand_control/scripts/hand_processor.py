#!/usr/bin/env python3 

import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped

from threading import Timer
from datetime import datetime
from collections import deque
import math
import tf
import numpy as np
import time
import csv

import cv2  

from numpy import interp

class PT():
    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

class Pose():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.r = 0.0
        self.p = 0.0
        self.y = 0.0


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
 

class gesture_processing():
    def __init__(self):
        self.pose = Pose()
        rospy.init_node('gesture_processing', anonymous=True)
        rospy.Subscriber("/gesture_realseance", PoseStamped, self.callback)
        rospy.Subscriber("/vicon/blue/blue", TransformStamped, self.orient_callback)

        self.pose_stamped_publisher = rospy.Publisher('/gesture_proc_pose', PoseStamped, queue_size=10)
        self.pose_stamped_rviz_publisher = rospy.Publisher('/gesture_rviz_proc_pose', PoseStamped, queue_size=10)        

        # self.vel_publisher = rospy.Publisher("/blue/mavros/setpoint_velocity/cmd_vel", TwistStamped , queue_size=10)
        self.vel_publisher = rospy.Publisher("/drone1/mavros/setpoint_velocity/cmd_vel", TwistStamped , queue_size=10)
        
        self.gesture_pose = PoseStamped()
        self.gesture_pose_realsence = PoseStamped()
        self.gesture_rviz = PoseStamped()
        self.gesture_pose.header.frame_id = "map"
        self.gesture_rviz.header.frame_id = "map"
        self.quaternion = [0.0, 0.0, 0.0, 0.0]

        # CV remote visualisation
        height, width = 500, 1000
        thickness = 3
        color_ch = (0, 255, 0)
        self.image = np.zeros((height, width, 3), np.uint8)

        self.th_screen = [[50, 450], 250]
        self.yaw_screen = [[50, 450], 250]
        self.pitch_screen = [[50, 450], 750]
        self.roll_screen = [[550, 950], 250]
       
        self.remote_rc =[0,0,0,0]
        self.yaw_drone = 0.0
        self.orient = None

        self.draw_remote_like_screen(color_ch, thickness)    

        self.t = PT(0.005, self.printer)
        self.t.start()

        # --- Filter ---
        self.size = 50
        self.collecter_x, self.collecter_y, self.collecter_z = deque([0]), deque([0]), deque([0])
        self.collecter_r, self.collecter_p, self.collecter_yaw = deque([0]), deque([0]), deque([0])

        for i in range(self.size  - 1):
            self.collecter_x.append(0)
            self.collecter_y.append(0)
            self.collecter_z.append(0)
            self.collecter_r.append(0)
            self.collecter_p.append(0)
            self.collecter_yaw.append(0)
        #------------

    def orient_callback(self, msg):
        x = msg.transform.rotation.x
        y = msg.transform.rotation.y
        z = msg.transform.rotation.z
        w = msg.transform.rotation.w

        def euler_from_quaternion(x, y, z, w):
            """Convert a quaternion into euler angles (roll, pitch, yaw)

            Args:
                x (_type_): _description_
                y (_type_): _description_
                z (_type_): _description_
                w (_type_): _description_

            Returns:
                roll_x, pitch_y, yaw_z: is rotation around x, y, z in radians (counterclockwise)
            """            

            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = math.atan2(t0, t1) 
            # roll_x = math.atan2(t0, t1) * (180 / math.pi)
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
            # pitch_y = math.asin(t2) * (180 / math.pi)
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = math.atan2(t3, t4)
            # yaw_z = math.atan2(t3, t4)* (180 / math.pi)
            return roll_x, pitch_y, yaw_z # in radians

        r, p, y = euler_from_quaternion(x, y, z, w)

        self.yaw_drone = y

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """Convert an Euler angle to a quaternion.

        Args:
            roll (float): The roll (rotation around x-axis) angle in radians.
            pitch (float): The pitch (rotation around y-axis) angle in radians.
            quaternion (float): The yaw (rotation around z-axis) angle in radians.

        Returns:
            quaternion([float,float,float,float]): The orientation in quaternion [x,y,z,w] format
        """        
        roll = roll * np.pi / 180.0
        pitch = pitch * np.pi / 180.0
        yaw = yaw * np.pi / 180.0
        
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        quaternion = [qx, qy, qz, qw]
        return quaternion

    def draw_remote_like_screen(self, color_ch, thickness):
        "Throttle"
        self.image = cv2.line(self.image, [self.th_screen[1], self.th_screen[0][0]], [self.th_screen[1], self.th_screen[0][1]], color_ch, thickness) 
        "Yaw"
        self.image = cv2.line(self.image, [self.yaw_screen[0][0], self.yaw_screen[1]], [self.yaw_screen[0][1], self.yaw_screen[1]], color_ch, thickness) 
        "Pitch"
        self.image = cv2.line(self.image, [self.pitch_screen[1], self.pitch_screen[0][0]], [self.pitch_screen[1], self.pitch_screen[0][1]], color_ch, thickness) 
        "Roll"
        self.image = cv2.line(self.image, [self.roll_screen[0][0], self.roll_screen[1]], [self.roll_screen[0][1], self.roll_screen[1]], color_ch, thickness)

    def printer(self):
        xx, yy, zz = self.gesture_pose_realsence.pose.position.x, self.gesture_pose_realsence.pose.position.y, self.gesture_pose_realsence.pose.position.z
        r, p, yaw = self.gesture_pose_realsence.pose.orientation.x, self.gesture_pose_realsence.pose.orientation.y, self.gesture_pose_realsence.pose.orientation.z
        xx_sm, yy_sm, zz_sm, r_sm, p_sm, yaw_sm = self.smoothing([xx,yy,zz,r,p,yaw])
        
        self.gesture_pose.pose.orientation.x = round(r_sm,2)
        self.gesture_pose.pose.orientation.y = round(p_sm, 2)
        self.gesture_pose.pose.orientation.z = -round(yaw, 2)
        self.gesture_pose.pose.position.x = -xx_sm
        self.gesture_pose.pose.position.y = -yy_sm
        self.gesture_pose.pose.position.z = zz_sm

        x,y,z,w = get_quaternion_from_euler(r_sm, p_sm, -yaw)
        self.gesture_rviz.pose.orientation.x = x
        self.gesture_rviz.pose.orientation.y = y
        self.gesture_rviz.pose.orientation.z = z
        self.gesture_rviz.pose.orientation.w = w

        th_rc = interp(self.gesture_pose.pose.position.z,[0.2,0.6],[self.th_screen[0][1],self.th_screen[0][0]])
        yaw_rc = interp(-yaw_sm,[-45,45],[self.yaw_screen[0][1],self.yaw_screen[0][0]])
        pitch_rc = interp(self.gesture_pose.pose.orientation.y,[-45,45],[self.pitch_screen[0][1],self.pitch_screen[0][0]])
        roll_rc = interp(self.gesture_pose.pose.orientation.x,[-45,45],[self.roll_screen[0][0],self.roll_screen[0][1]])

        self.remote_rc = [
            th_rc,
            yaw_rc,
            pitch_rc,
            roll_rc
            ]

        self.pose_stamped_publisher.publish(self.gesture_pose)
        self.pose_stamped_rviz_publisher.publish(self.gesture_rviz)
        self.gesture_control_callback(self.gesture_pose)

    def check_boarders(self):
        if self.remote_rc[0] < self.th_screen[0][0]:
            self.remote_rc[0] = self.th_screen[0][0]
        if self.remote_rc[0] > self.th_screen[0][1]:
            self.remote_rc[0] = self.th_screen[0][1]
        if self.remote_rc[1] < self.yaw_screen[0][0]:
            self.remote_rc[1] = self.yaw_screen[0][0]
        if self.remote_rc[1] > self.yaw_screen[0][1]:
            self.remote_rc[1] = self.yaw_screen[0][1]
        if self.remote_rc[2] < self.pitch_screen[0][0]:
            self.remote_rc[2] = self.pitch_screen[0][0]
        if self.remote_rc[2] > self.pitch_screen[0][1]:
            self.remote_rc[2] = self.pitch_screen[0][1]
    
    def visualisation_remote(self):
        self.check_boarders()
        image_show = np.copy(self.image)
        image_show = cv2.circle(image_show, (int(self.th_screen[1]), int(self.remote_rc[0])), 15, (0, 0, 255), -1)
        image_show = cv2.circle(image_show, (int(self.remote_rc[1]), int(self.yaw_screen[1])), 15, (0, 0, 255), -1)
        image_show = cv2.circle(image_show, (int(self.pitch_screen[1]), int(self.remote_rc[2])), 15, (0, 0, 255), -1)
        image_show = cv2.circle(image_show, (int(self.remote_rc[3]), int(self.roll_screen[1])), 15, (0, 0, 255), -1)
        cv2.imshow('image', image_show)


    def smoothing(self, val):
        #---average running---
        self.collecter_x.rotate(1)
        self.collecter_y.rotate(1)
        self.collecter_z.rotate(1)
        self.collecter_r.rotate(1)
        self.collecter_p.rotate(1)
        self.collecter_yaw.rotate(1)

        self.collecter_x[0], self.collecter_y[0], self.collecter_z[0] = val[0], val[1], val[2]
        self.collecter_r[0], self.collecter_p[0], self.collecter_yaw[0] = val[3], val[4], val[5]

        val_smooth = [sum(self.collecter_x)/self.size, sum(self.collecter_y)/self.size, sum(self.collecter_z)/self.size,
            sum(self.collecter_r)/self.size, sum(self.collecter_p)/self.size, sum(self.collecter_yaw)/self.size]

        return val_smooth

    def callback(self, data):
        # rospy.loginfo(data.pose)
        self.gesture_pose_realsence = data

    def gesture_control_callback(self, msg):
        roll = msg.pose.orientation.x
        pitch = msg.pose.orientation.y
        yaw = msg.pose.orientation.z

        thrust = msg.pose.position.z
        yaw += yaw * 0.01

        self.vel_from_orinent([roll, pitch, yaw], thrust)

    def vel_from_orinent(self, orient, th):

        speed = 1.5   
        y_speed = 0.1

        pitch = orient[1]
        roll = orient[0]
        yaw_rot = orient[2] * y_speed

        vx = math.sin(math.radians(roll)) * speed
        vy = math.sin(math.radians(pitch)) * speed

        if th is None:
            th = 0.0

        th_default = 0.4
        vz = round(th - th_default, 3)

        if abs(vz) < 0.03:
            vz = 0.0

        " rotate vector according to yaw of the drone "
        # vx_r = vx * math.cos(self.yaw_drone * math.pi / 180) - vy * math.sin(self.yaw_drone * math.pi / 180)
        # vy_r = vx * math.sin(self.yaw_drone * math.pi / 180) + vy * math.cos(self.yaw_drone * math.pi / 180)

        vx_r = vx * math.cos(self.yaw_drone) - vy * math.sin(self.yaw_drone) #real
        vy_r = vx * math.sin(self.yaw_drone) + vy * math.cos(self.yaw_drone)
        
        # print([round(vx,1), round(vy,1), round(vz,1)])

        # print("orient", orient[1], orient[0], self.yaw, vz)
        # print("th", th, vz)
        self.pub_vel([vx_r, vy_r, vz], yaw_rot)
    
    def pub_vel(self, velocity, yaw_rot=0.0):
        """ Publishes the goal pose to the /mavros/setpoint_velocity/cmd_vel topic to send the drone to the goal position.

        Args:
            velocity ([float, float, float]): drone speed [v_x, v_y, v_z]
        """        
        vel_cmd = TwistStamped()
        vel_cmd.header.frame_id = "map" #"base_link"
        vel_cmd.header.stamp = rospy.Time.now()
        vel_cmd.twist.linear.x = velocity[1]
        vel_cmd.twist.linear.y = -velocity[0]
        vel_cmd.twist.linear.z = velocity[2]
        vel_cmd.twist.angular.z = yaw_rot * 0.25
        self.vel_publisher.publish(vel_cmd)


if __name__ == '__main__':
    node = gesture_processing()

    while not rospy.is_shutdown():
        node.visualisation_remote()

        if cv2.waitKey(1) == ord('q'):
            break
        
    