# Real-time Drone Detection and Tracking on Jetson TX2
This code is a real-time algorithm for Visual Drone Detection and Tracking on the Nvidia Jetson TX2 using YOLOv3 and GOTURN. 
## Abstract

Today’s technology is evolving towards autonomous systems and the demand in autonomous drones,
cars, robots, etc. has increased drastically in the past years. This project presents a solution for au-
tonomous real-time visual detection and tracking of hostile drones by moving camera equipped on
surveillance drones.  The algorithm developed in this project, based on state-of-art deep learning
and computer vision methods, succeeds at autonomously detecting and tracking a single drone by
moving camera and can run at real-time on the Nvidia Jetson TX2.  The project can be divided
into two main parts:  the detection and the tracking.  The detection is based on the YOLOv3 (You
Only Look Once v3) algorithm and a sliding window method.  The tracking is based on the
GOTURN (Generic Object Tracking Using Regression Networks) algorithm, which allows to
track generic objects at high speed.  In order to allow autonomous tracking and enhance the accu-
racy, a combination of GOTURN and tracking by detection using YOLOv3 was developed.  The
developed method detects at 18 FPS and tracks at 28 FPS on the Nvidia Jetson TX2.

Keywords: autonomous, drone, detection, tracking, real-time, moving camera, YOLO,  GOTURN, Nvidia Jetson TX2


## Demo 

https://www.youtube.com/watch?v=kLVupa1nkZs

## Requirements 

#### 1. Cuda 9.0

#### 2. OpenCV 3.4.1

#### 3. Caffe 1.0.0

#### 4. Pytorch 1.0.0

#### 5. Others
All other required packages are defined in the ”requirement.txt” file. They can all be installed in your shell using the command: 
$ pip install -r requirements.txt

## Acknowledgment 
This code has been done thanks to the work available in the following github repositories:

https://github.com/davheld/GOTURN

https://github.com/nrupatunga/PY-GOTURN/

https://github.com/ultralytics/yolov3



