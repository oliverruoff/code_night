from cv2 import VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite
import cv2
import argparse
import numpy as np

FRAME_SIZE = 50

# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
imwrite("image.jpg",img) #save image