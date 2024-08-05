import subprocess
import rospy
import sys
from visualization_msgs.msg import Marker


def plot_world_points_with_rviz(
    all_world_points, camera_position, is_first_time_to_run
):
    """
    @input all_world_points: [M, N, 3]
    @input camera_position: [3]
    """

    if is_first_time_to_run:
        exit_code = subprocess.call(
            "source /root/TennisCourtDetector/visualization/devel/setup.bash \
                && roslaunch tennis_court tennis_court.launch",
            shell=True,
            executable="/bin/bash",
        )
        if exit_code:
            print("exit_code", exit_code)
            exit()

        rospy.init_node("world_points")

    marker_pub = rospy.Publisher("world_points", Marker, queue_size=10)

    # Publish
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        rospy.loginfo("Publishing world points markers...")
        marker_pub.publish(Marker())

        rate.sleep()
