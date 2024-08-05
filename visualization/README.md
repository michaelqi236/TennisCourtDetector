# Run through Docker

Due to ROS2 not being supported with Ubuntu 22, we use docker to run visualization with Rviz.

To ensure GUI is shown correctly, start docker contain with X11 forward:

```
docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=rviz_container  michaelqi236/robot:tennis_court_detection
```

# Launch visualization
```
cd /root/TennisCourtDetector
source visualization/devel/setup.bash
roslaunch tennis_court tennis_court.launch
```
