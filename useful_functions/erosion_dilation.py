import cv2
import numpy as np

img = cv2.imread('/home/CV/images/Picture1.jpg')



kernel = np.array([[0,0,1,0,0],
                  [0,0,1,1,0],
                  [0,0,1,1,1],
                  [0,0,1,1,0],
                  [0,0,1,0,0]],dtype=np.uint8)

def my_erosion(img, kernel, i):
    erosion = cv2.erode(img,kernel,iterations = i)
    return erosion

def my_dilation(img,kernel,i):
    dilation = cv2.dilate(img,kernel,iterations = i)
    return dilation
