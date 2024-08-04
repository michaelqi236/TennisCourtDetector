#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def create_marker(marker_id, points, color, frame_id="world"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "tennis_court"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.points = points
    return marker

def draw_court():
    rospy.init_node('tennis_court_visualization')

    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    # Tennis court dimensions (in meters)
    court_length = 23.77
    court_width = 8.23
    net_height = 0.91

    # Court boundaries
    boundary_points = [
        Point(0, 0, 0), Point(court_width, 0, 0), Point(court_width, court_length, 0),
        Point(0, court_length, 0), Point(0, 0, 0)
    ]
    boundary_marker = create_marker(0, boundary_points, [0, 1, 0, 1])  # Green color

    # Net
    net_points = [Point(0, court_length / 2, 0), Point(court_width, court_length / 2, 0)]
    net_marker = create_marker(1, net_points, [0, 0, 0, 1])  # Black color

    # Service boxes
    service_line_distance = 6.40
    service_box_points_1 = [
        Point(0, service_line_distance, 0), Point(court_width, service_line_distance, 0)
    ]
    service_box_points_2 = [
        Point(0, court_length - service_line_distance, 0), Point(court_width, court_length - service_line_distance, 0)
    ]
    center_line_points = [Point(court_width / 2, 0, 0), Point(court_width / 2, court_length, 0)]
    service_box_marker_1 = create_marker(2, service_box_points_1, [0, 1, 0, 1])  # Green color
    service_box_marker_2 = create_marker(3, service_box_points_2, [0, 1, 0, 1])  # Green color
    center_line_marker = create_marker(4, center_line_points, [0, 1, 0, 1])  # Green color

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(boundary_marker)
        marker_pub.publish(net_marker)
        marker_pub.publish(service_box_marker_1)
        marker_pub.publish(service_box_marker_2)
        marker_pub.publish(center_line_marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        draw_court()
    except rospy.ROSInterruptException:
        pass
