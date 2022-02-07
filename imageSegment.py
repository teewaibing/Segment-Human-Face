# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
from matplotlib import pyplot as plt
import numpy as np


input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # background
    lower_bcg = np.array([0,0,0],np.uint8)
    upper_bcg = np.array([179,40,255],np.uint8)
    bcg = cv2.inRange(img_hsv, lower_bcg, upper_bcg)

    # hair/eyebrows
    lower_hair= np.array([0,0,0],np.uint8)
    upper_hair = np.array([179,255,103],np.uint8)
    hair = cv2.inRange(img_hsv, lower_hair, upper_hair)

    # mouth
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (100,300,150,150)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mouth = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # eyes
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,200,250,100)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    eye = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # nose
    lower_nose = np.array([7,47,72],np.uint8)
    upper_nose = np.array([13,181,163],np.uint8)
    nose = cv2.inRange(img_hsv, lower_nose, upper_nose)

    # skin
    lower_skin = np.array([0,48,80],np.uint8)
    upper_skin = np.array([20,255,255],np.uint8)
    skin = cv2.inRange(img_hsv, lower_skin, upper_skin)


    outImg = bcg + hair + mouth + eye + nose + skin

    outImg[np.where(bcg)] = [0]
    outImg[np.where(hair)] = [1]
    outImg[np.where(mouth)] = [2]
    outImg[np.where(eye)] = [3]
    outImg[np.where(nose)] = [4]
    outImg[np.where(skin)] = [5]

    # END OF YOUR CODE
    #########################################################################
    return outImg
