#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf
import tf2_ros
import geometry_msgs.msg
import sys
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.sep.join(dir_path.split(os.sep)[:3])
sys.path.append(path)

from lib import parameters

exit()

def init_rviz():
    rospy.init_node('tennis_court_visualization')

    # Start broadcasting the frame in a separate thread
    import threading
    frame_thread = threading.Thread(target=broadcast_frame)
    frame_thread.start()

    # Wait for the marker publisher to be ready
    rospy.sleep(2)

def broadcast_frame():
    # Create a TransformBroadcaster object
    br = tf.TransformBroadcaster()

    # Frame details
    frame_id = "world"
    child_frame_id = "map"  # You can use "map" or any other name as needed

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        br.sendTransform(
            (0, 0, 0),  # Translation (x, y, z)
            tf.transformations.quaternion_from_euler(0, 0, 0),  # Rotation (x, y, z, w)
            rospy.Time.now(),
            child_frame_id,
            frame_id
        )
        rate.sleep()

def create_line_marker(marker_id, points, color, frame_id="world", lifetime=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "tennis_court"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1

    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    
    marker.points = points
    marker.lifetime = rospy.Duration(lifetime)
    return marker

def create_cube_marker(marker_id, center, shape, color, frame_id="world", lifetime=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "tennis_court_ground"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1

    marker.scale.x = shape[0]
    marker.scale.y = shape[1]
    marker.scale.z = 0.01
    
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = -0.005

    marker.lifetime = rospy.Duration(lifetime)
    return marker


def draw_court(marker_pub):
    court_param = CourtParam()

    # Court ground
    ground_marker = create_cube_marker(0, [0,0,0], shape, [0, 1, 0, 1], frame_id="world", lifetime=0)# Green color
 
    # Court boundaries
    boundary_points = [
        Point(0, 0, 0), Point(court_width, 0, 0), Point(court_width, court_length, 0),
        Point(0, court_length, 0), Point(0, 0, 0)
    ]
    boundary_marker = create_line_marker(0, boundary_points, [0, 1, 0, 1], lifetime=0)  
    # Net
    net_points = [Point(0, court_length / 2, 0), Point(court_width, court_length / 2, 0)]
    net_marker = create_line_marker(1, net_points, [0, 0, 0, 1], lifetime=0)  # Black color

    # Service boxes
    service_line_distance = 6.40
    service_box_points_1 = [
        Point(0, service_line_distance, 0), Point(court_width, service_line_distance, 0)
    ]
    service_box_points_2 = [
        Point(0, court_length - service_line_distance, 0), Point(court_width, court_length - service_line_distance, 0)
    ]
    center_line_points = [Point(court_width / 2, 0, 0), Point(court_width / 2, court_length, 0)]
    service_box_marker_1 = create_line_marker(2, service_box_points_1, [0, 1, 0, 1], lifetime=0)  # Green color
    service_box_marker_2 = create_line_marker(3, service_box_points_2, [0, 1, 0, 1], lifetime=0)  # Green color
    center_line_marker = create_line_marker(4, center_line_points, [0, 1, 0, 1], lifetime=0)  # Green color

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        rospy.loginfo("Publishing markers...")
        marker_pub.publish(ground_marker)
        marker_pub.publish(boundary_marker)
        marker_pub.publish(net_marker)
        marker_pub.publish(service_box_marker_1)
        marker_pub.publish(service_box_marker_2)
        marker_pub.publish(center_line_marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        init_rviz()

        marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        draw_court(marker_pub)

    except rospy.ROSInterruptException:
        pass
