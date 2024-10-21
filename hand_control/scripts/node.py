#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np
import pyrealsense2 as rs
import math

from landmarks_detector import gesture_points_detector

class HandDetector:
    def __init__(self):
        """ Realseance D435 init """
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.hole_filling = rs.hole_filling_filter(2)
        self.profile = self.pipe.start(self.cfg)

        self.gesture_pose = PoseStamped()
        self.gesture_pose.header.frame_id = "/map"

        self.publish_gesture = rospy.Publisher('/gesture_realseance', PoseStamped, queue_size=10) # set local position        

        rospy.loginfo("HandDetector node initialized.")

    def stream_alignment(self, color, points):

        # source: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb 
        frameset = self.pipe.wait_for_frames()
        colorizer = rs.colorizer()

        colorized_depth = np.asanyarray(colorizer.colorize(self.depth_frame).get_data())
        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        hole_filling = rs.hole_filling_filter(2)
        filled_depth = hole_filling.process(aligned_depth_frame)
        colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
        colorized_depth = cv2.flip(colorized_depth, 1)

        # Show the two frames together:
        images = np.hstack((color, colorized_depth))

        # get depth
        depth = np.asanyarray(filled_depth.get_data())
        depth = cv2.flip(depth, 1) 
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
                
        return images, colorized_depth


    def get_frame(self):
        " Get color image from Realsence camera "
        frame = self.pipe.wait_for_frames()
        color_image = np.asanyarray(frame.get_color_frame().get_data())
        color_image = cv2.flip(color_image, 1)
        
        image_detector = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # for gesture detection
        
        " Get depth image "
        align = rs.align(rs.stream.color)
        frameset = align.process(self.pipe.wait_for_frames())

        aligned_depth_frame = frameset.get_depth_frame()
        hole_filling = rs.hole_filling_filter(2)
        filled_depth = hole_filling.process(aligned_depth_frame)
        colorized_depth = np.asanyarray(rs.colorizer().colorize(filled_depth).get_data())
        colorized_depth = cv2.flip(colorized_depth, 1)

        image_visualisation = np.hstack((color_image, colorized_depth)) # for visualisation

        """ Get Depth data"""
        depth = np.asanyarray(filled_depth.get_data())
        depth = cv2.flip(depth, 1) 
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale

        return image_visualisation, depth, image_detector

    def get_yaw_hand(self, points):
        x1, y1 = points[8][0], points[8][1]
        x2, y2 = points[12][0], points[12][1]
        x_c1, y_c1 = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1 = points[5][0], points[5][1]
        x2, y2 = points[9][0], points[9][1]
        x_c2, y_c2 = (x1 + x2) / 2, (y1 + y2) / 2
        dy = y_c1 - y_c2
        dx = x_c1 - x_c2
        rads = math.atan2(dy, dx)
        degs = math.degrees(rads)
        if (degs < 0):
            degs += 90
        return round(degs, 1)

    def get_angles_and_pose(self, depth, points):
        " PITCH calculator "
        " Up and Down pulm point "
        up_point = [int((points[5, 1] + points[17, 1])/2), int((points[5, 0] + points[17, 0])/2)]
        down_point = [int(points[0, 1]), int(points[0, 0])]
        depth_up_point = round(depth[up_point[0], up_point[1]].astype(float), 3)
        depth_down_point = round(depth[down_point[0], down_point[1]].astype(float), 3)
        delta_pitch = round((depth_down_point - depth_up_point)*1000, 1)

        " ROLL calculator "
        " Left and Right pulm points "
        left_point = [int(points[5, 1]), int(points[5, 0])]
        right_point = [int(points[17, 1]), int(points[17, 0])]
        depth_left_point = round(depth[left_point[0], left_point[1]].astype(float), 3)
        depth_right_point = round(depth[right_point[0], right_point[1]].astype(float), 3)
        delta_roll = round((depth_left_point - depth_right_point)*1000, 1)
        
        " YAW calculator "
        yaw = self.get_yaw_hand(points)

        th = 15

        " Threshold values "
        if delta_roll > -th and delta_roll < th:
            delta_roll = 0
        if delta_pitch > -th and delta_pitch < th:
            delta_pitch = 0

        if delta_roll >= th and delta_roll != 0:
            delta_roll -= th
        if delta_roll < th and delta_roll != 0:
            delta_roll += th

        if delta_pitch >= th and delta_pitch != 0:
            delta_pitch -= th
        if delta_pitch < th and delta_pitch != 0:
            delta_pitch += th

        if yaw > -12 and yaw < 12:
            yaw = 0.0
        if yaw >= 12 and yaw != 0:
            yaw -= 12
        if yaw < 12 and yaw != 0:
            yaw += 12

        self.gesture_pose.pose.orientation.x = delta_roll
        self.gesture_pose.pose.orientation.y = delta_pitch
        self.gesture_pose.pose.orientation.z = yaw

        self.gesture_pose.pose.position.x = int(points[0, 1])
        self.gesture_pose.pose.position.y = int(points[0, 0])
        self.gesture_pose.pose.position.z = round(depth_down_point,2)

        self.publish_gesture.publish(self.gesture_pose)
        
        return [delta_roll, delta_pitch, yaw]


    def main(self):
        while not rospy.is_shutdown():
            
            image_visualisation, depth, image_detector = self.get_frame()

            points, image_visualisation, gesture_ml = gesture_points_detector(image_detector, image_visualisation)
            
            if points is not None:
                self.get_angles_and_pose(depth, points)

            cv2.imshow('images', image_visualisation)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                rospy.loginfo("exit()!")
                rospy.signal_shutdown("exit()!")

if __name__ == '__main__':
    try:
        rospy.init_node('hand_detector', anonymous=True)

        node = HandDetector()

        while not rospy.is_shutdown():
            node.main()

    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down gate_detection node.")

    except Exception as e:
        rospy.logerr(f"Unexpected error: {str(e)}")

    finally:
        pass