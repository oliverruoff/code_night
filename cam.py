from cv2 import VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite

FRAME_SIZE = 500

# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
imwrite("image.jpg",img) #save image