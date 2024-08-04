# Run through Docker

Due to ROS2 not being supported with Ubuntu 22, we use docker to run visualization with Rviz.

To ensure GUI is shown correctly, start docker contain with X11 forward:

```
docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=rviz_container  michaelqi236/robot:x86
```