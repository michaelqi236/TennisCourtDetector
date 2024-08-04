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
path_layers =  dir_path.split(os.sep)[:3]
path_layers.append('lib')
path = os.sep.join(path_layers)
sys.path.append(path)

from parameters import CourtParam

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

def create_cube_marker(marker_id, center, shape, color, frame_id="world", z_offset = 0):
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
    marker.pose.position.z = z_offset

    return marker


def draw_court(marker_pub):
    court_param = CourtParam()

    # Court ground
    ground_marker_0 = create_cube_marker(0, center=court_param.net_points[5], shape=[court_param.width, court_param.length], color=[0, 1, 0, 1], frame_id="world", z_offset = -0.01)# Green color
    buffer = 5
    ground_marker_1 = create_cube_marker(1, center=court_param.net_points[5], shape=[court_param.width + buffer, court_param.length + buffer], color=[0.9, 0.5, 0, 1], frame_id="world", z_offset = -0.03)# yellow color

    # Court boundaries
    court_points = court_param.court_points
    net_points = court_param.net_points
    
    point_idx_0 = [0,1,3,2,0] # court boundary
    line_points_0 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_0]
    line_marker_0 = create_line_marker(0, line_points_0, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_1 = [4,5] # inner court boundary
    line_points_1 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_1]
    line_marker_1 = create_line_marker(1, line_points_1, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_2 = [6,7] # inner court boundary
    line_points_2 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_2]
    line_marker_2 = create_line_marker(2, line_points_2, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_3 = [8,9] # serving box boundary
    line_points_3 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_3]
    line_marker_3 = create_line_marker(3, line_points_3, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_4 = [10,11] # serving box boundary
    line_points_4 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_4]
    line_marker_4 = create_line_marker(4, line_points_4, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_5 = [12,13] # mid line
    line_points_5 = [Point(court_points[i][0], court_points[i][1], 0) for i in  point_idx_5]
    line_marker_5 = create_line_marker(5, line_points_5, [1, 1, 1, 1], lifetime=0)  # White color

    point_idx_6 = [7,9] # net line
    line_points_6 = [Point(net_points[i][0], net_points[i][1], 0) for i in  point_idx_6]
    line_marker_6 = create_line_marker(6, line_points_6, [1, 1, 1, 1], lifetime=0)  # White color

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        rospy.loginfo("Publishing markers...")
        marker_pub.publish(ground_marker_1)
        marker_pub.publish(ground_marker_0)
        marker_pub.publish(line_marker_0)
        marker_pub.publish(line_marker_1)
        marker_pub.publish(line_marker_2)
        marker_pub.publish(line_marker_3)
        marker_pub.publish(line_marker_4)
        marker_pub.publish(line_marker_5)
        marker_pub.publish(line_marker_6)
        rate.sleep()

if __name__ == '__main__':
    init_rviz()

    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    draw_court(marker_pub)
