# IP
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:39:44 2019

@author: jackc
Edges and gradients Task

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import math
from pylab import *
from numpy import array
import sys
#-------------Notes-----------
#Edges found thrpough gradients
    #Gradient is thresholded for an edge
    #To do with derivaties, rates of change
#Digital images not continuous => Pixel intensity minus its neighbor in that direction is the derivative
    
#Noise effects intensity variation
    #Use a filter to counteract this (gaussian etc)
    #Do fitering sepratly to gradient calculating

def userSelect():
    f = easygui.fileopenbox()
    I1 = cv2.imread(f)
    return(I1)


def imageDisplay(I1):
    cv2.imshow("Final output",I1)
    cv2.waitKey(0)

def blobDetect(I):
    
    params = cv2.SimpleBlobDetector_Params()#Needed before activation of parameters

    params.filterByArea = True#Mark as false if not used
    params.minArea = (50)#sets up minimum area for the blob

    params.filterByCircularity = True#Filter by how circular a blob is
    params.minCircularity = 0.1

    params.filterByConvexity = True#How convex the blob is
    params.minConvexity = 0.1

    params.filterByInertia = True#
    params.minInertiaRatio = 0.01

    params.minDistBetweenBlobs = True#Sets a minimum distance required between blobs for them to be picked up
    params.minDistBetweenBlobs = 1
    
    detector = cv2.SimpleBlobDetector_create(params)
    #k# creates now closed mask reverse
    keypoints = detector.detect(I)
    blobs1 = len(keypoints)#Counts blobs
    
    return(blobs1)
    

def filterExample(I):
  kHP=np.array([[-1/12,-1/12,-1/12],[-1/12,1,-1/12],[-1/12,-1/12,-1/12]],dtype=float)#High pass filter
  kLP=np.array([[1/9,1/9,1/9],[1/9,1,1/9],[1/9,1/9,1/9]],dtype=float)#Low pass filter
  

  filteredImage=cv2.filter2D(blur,ddepth=-1,kernel=kHP)
  
I1=userSelect()

    
#-----------------Image gradients------------------
    #G=greyscale image 
    #dx= order of x derivates 
    #dy= order of y derivatives
#----------Sobel
ddepth=cv2.CV_64F#Allows for negative numbers

G = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY);

Ix=cv2.Sobel(G,ddepth=cv2.CV_64F,dx=1,dy=0)
Iy=cv2.Sobel(G,ddepth=cv2.CV_64F,dx=0,dy=1)

#--------Canny

    #-------Extract edges-------
    #Threshold1 = definatly edges
    #Threshold2 = Definatly not edges
    #I=original image
    
E=cv2.Canny(I1,threshold1=100,threshold2=150)#Will retuen strogest edges with gradients above 100
imageDisplay(E)
