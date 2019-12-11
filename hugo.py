import cv2
from PIL import  Image
import matplotlib as plt
import pandas as pd
import numpy as np
def view_image(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Convert to any image colour type
def convert(image, colour):
    if colour == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
    elif colour == "rgb":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb
    elif colour == "rgb":
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        return yuv
    elif colour == "gray":
        G = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return G
    print("Error!")
    return 0

def histogram(image):
    G = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Values = G.ravel()
    plt.hist(Values, bins=256, range=[0, 256])
    H = cv2.equalizeHist(G)
    return H

# Resize an image
def resize(image, w, h):
    S = cv2.resize(image, dsize=(w,h))
    return S

# Crop an image
def crop(image, x1, y1, x2, y2):
    C = image[x1:x2, y1:y2]
    return C

# Rotate an image
def rotate(image, cx, cy, d, s):
    M = cv2.getRotationMatrix2D(center=(cx, cy), angle=d, scale=s)
    R = cv2.warpAffine(I, M=M, dsize=(w, h))

# Add/Layer images together
def addition(I1, I2):
    A = cv2.add(I1, I2)
    return A

# Function that sharpens the image
def sharpen(image):
    kernel = np.array([[(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)]])
    output = cv2.filter2D(image, -1, kernel)
    return output

def combine(I1, I2):
    I1 = cv2.imread(I1)
    I2 = cv2.imread(I2)
    I1scaled=np.array((I1*0.3),dtype=np.uint8)# Scaled the array (image) by 0.5 and allows the image to stay as intergers
    I2scaled=np.array((I2*0.5),dtype=np.uint8)# Multiplying asjusts the 3 colour values therefore the brightness
    A= cv2.add(I1scaled,I2scaled)#Adds the two images onto the one image
    betterIm=np.array((I1*0.1),dtype=np.uint8)
    A2= cv2.add(betterIm,A)
    return A2

def kernel_sharpen(Image):
    Image=cv2.imread(Image)
    k=np.array([[-1/9,-1/9,-1/9],[-1/9,1,-1/9],[-1/9,-1/9,-1/9]],dtype=float)#High pass filter used in sharpening, done by rows in the array
    f=cv2.filter2D(Image,ddepth=-1,kernel=k)# K is the kernal, f filtered image, I image
    return f

def threshold(image, r1, g1, b1, r2, g2, b2):
    rangeLower1 = (r1, g1, b1)  # RIO values for the blue, green and red values, respectivly. Lower is the lowest value and upper is teh highest value allod
    ##NOTE: the default is BGR
    rangeUpper1 = (r2, g2, b2)  # more versitile than single threshold
    B = cv2.inRange(image, rangeLower1, rangeUpper1)  # New image is the ragees give of the last image
    G = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    B2 = cv2.adaptiveThreshold(G, maxValue=10, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               thresholdType=cv2.THRESH_BINARY, blockSize=5, C=15)
    return B

# Layer 2 images with a mask
def mask_add(image1, image2, rangeLower1, rangeUpper1, rangeLower2, rangeUpper2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    B = cv2.inRange(image1, rangeLower1, rangeUpper1)  # created the mask B using the orenge values
    B2 = cv2.inRange(image2, rangeLower2, rangeUpper2)  # Creates a mask for the image2, ultimatly not used

    BN = cv2.bitwise_not(B)  # creating the reverse mask, so that the image2 can be seen with a space cut for the image1

    ROI = cv2.bitwise_and(image1, image1, mask=B);  # Normal ROI created from the original mask
    ROI2 = cv2.bitwise_and(image2, image2, mask=B2)  # ROI created for image2, not needed

    ROI_reverse = cv2.bitwise_and(image2, image2,
                                  mask=BN)  # Cuts out the shape of the orenge from the image2, using orenge B value

    ROI_final = cv2.bitwise_or(ROI_reverse, ROI)  # combines the orenge and the cut out orenge from the image2 pictures
    ROI_final = cv2.cvtColor(ROI_final, cv2.COLOR_BGR2RGB)
    return ROI_final
