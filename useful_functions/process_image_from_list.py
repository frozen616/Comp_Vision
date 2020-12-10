import cv2
import os
import numpy as np
import pandas as pd


#  Location and loading Face Cascade XML
OPENCV_PATH = "/usr/local/lib/python3.6/dist-packages/cv2/data"
faceXML = os.path.join(OPENCV_PATH,'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(faceXML)

baseDir = "/home/CV/FaceData_Test"
folderList = ['A']
folderPathList = [os.path.join(baseDir,folder) for folder in folderList]


def process_image_from_list(faceCascade, image_list, scaleFactor=1.07, minNeighbors=8, minSize=(100,100), maxSize=(400,400)):
    imageList = list()
    for imageFile in image_list:
        # Reads image into RGB for histogram comparison
        img = cv2.imread(imageFile)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, maxSize=maxSize)
        imageList.append((imageFile, img_gray, img_rgb, faces))
    
    return imageList
