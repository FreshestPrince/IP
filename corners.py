import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

I=cv2.imread('sudoku.jpg')
I=cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
corners =cv2.goodFeaturesToTrack(G,maxCorners=50, qualityLevel=0.1,minDistance=10)
H =cv2.cornerHarris(G,blockSize=5,ksize=3,k=0.04)

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(I,(x,y),3,(255,0,255),-1)


plt.imshow(I)
plt.show()
