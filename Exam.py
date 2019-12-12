import cv2
import numpy as np
import easygui as ui
import matplotlib.pyplot as plt


"""
The 'showimage()' function, 
"""
def showimage(image):
    cv2.namedWindow('Debug Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
The 'blur()' function, this function blurs the image using the 'cv2.filter2D()' function in conjunction with a blurring
kernel.
"""
def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    output = cv2.filter2D(image, -1, kernel)
    return output


"""
The 'sharpen()' function, this function sharpens the image using the 'cv2.filter2D()' function in conjunction with a 
sharpening kernel.
"""
def blur(image):
    kernel = np.array([[(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)]])
    output = cv2.filter2D(image, -1, kernel)
    return output


"""
The 'rgb_separator()" function,
"""
def rgb_separator(image, low_range, high_range):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = rgb.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, low_range, high_range)
    rows, cols, layers = image.shape
    for i in range(0, rows):
        for j in range(0, cols):
            k = mask[i, j]
            if k < 200:
                original[i, j] = [255, 255, 255]
    rgb_sep = original
    return rgb_sep


"""
The 'yuv_separator()" function,
"""
def yuv_separator(image, low_range, high_range):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = rgb.copy()
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    mask = cv2.inRange(yuv, low_range, high_range)
    rows, cols, layers = image.shape
    for i in range(0, rows):
        for j in range(0, cols):
            k = mask[i, j]
            if k < 200:
                original[i, j] = [255, 255, 255]
    yuv_sep = original
    return yuv_sep


"""
The 'hsv_separator()" function,
"""
def hsv_separator(image, low_range, high_range):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = rgb.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_range, high_range)
    rows, cols, layers = image.shape
    for i in range(0, rows):
        for j in range(0, cols):
            k = mask[i, j]
            if k < 200:
                original[i, j] = [255, 255, 255]
    hsv_sep = original
    return hsv_sep


"""
The 'scale()" function,
"""
def scale(image, ratio):
    h, w, d = image.shape()
    S = cv2.resize(image, dsize=(ratio*w, ratio*h))
    return S


"""
The 'rotate()" function,
"""
def rotate(image, cx, cy, angle):
    h, w, d = image.shape()
    M = cv2.getRotationMatrix2D(center = (cx,cy), angle = angle, scale = 0.5)
    R = cv2.wrapAffine(image, M = M, dsize=(w,h))
    return R


"""
The 'inpaint()" function,
"""
def inpaint(image, low_range, high_range):
    mask = cv2.inRange(image, low_range, high_range)
    dst = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
    return dst


"""
The 'fillwhitecirles()" function, kinda shit but it uses houghcircles yoke so may get marks for innovation,
can change the colour of circle its looking for too by adjusting the blue, green and red values.
"""
def fillwhitecirles(image):
    output = image.copy()
    rows, cols, layers = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=0,
                                        maxRadius=50)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            if b < rows and a < cols:
                BGR = output[b, a]
                Blue, Green, Red = BGR
                if Blue > 150 and Green > 150 and Red > 150:
                    cv2.circle(image, (a, b), r, (255, 255, 255), -1)
                    cv2.circle(image, (a, b), r, (255, 255, 255), 10)
    return image


"""

Some useful little boizzzzz

##############################################################
						Switching colour spaces
##############################################################
RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
HSV = cv.cvtColor(image,cv2.COLOR_BGR2HSV)
YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
G = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##############################################################
##############################################################
						Shape drawing functions
##############################################################
cv2.line(img = image, pt1 = (200,200), pt2 = (500,600), color = (255,255,255), thickness = 5)
cv2.circle(img = image, centre = (800,400), radius - 50, colour = (0,0,255), thickness = -1)
cv2.rectangle(img = image, pt1 = (500,100), pt2 = (800,300), colour = (255,0,255), thickness = 10)
##############################################################
##############################################################
						 Equalising Histograms 
##############################################################
G = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
H = cv2.equalizeHist(G)
##############################################################
##############################################################
                            Cropping 
##############################################################
C = image[0:180,0:270]
##############################################################
##############################################################
                           Image Maths 
##############################################################
AddedImage = cv2.add(image1, image2)
SubtractedImage = cv2.subtract(image1, image2)
MultImage  = cv2.multiply(image1,image2,scale =0.01)
DividedImage = cv2.divide(image1,image2,scale = 100)
##############################################################
##############################################################
    Kernels (pop these into sharpen/blur function)
##############################################################
kernel = np.array([[(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)], [(1 / 9), (1 / 9), (1 / 9)]]) # Low pass filter
kernel = np.array([[(-1 / 4), (0 / 4), (1 / 4)], [(-2 / 4), (0 / 4), (2 / 4)], [(-1 / 4), (0 / 4), (1 / 4)]]) # Horizontal Gradient
kernel = np.array([[(-1 / 4), (-2 / 4), (-1 / 4)], [(0 / 4), (0 / 4), (0 / 4)], [(1 / 4), (2 / 4), (1 / 4)]]) # Vertical Gradient
kernel = np.array([[(-1 / 8), (-1 / 8), (-1 / 8)], [(-1 / 8), (8 / 8), (-1 / 8)], [(-1 / 8), (-1 / 8), (-1 / 8)]]) # High pass filter
filt = cv2.bilateralFilter(dst,9,75,75) # bilateral filter
##############################################################
"""





"""
Opening user interface 
Create a message box using 'ui.msg()' function and input a welcome message and an instruction.
Next open up a file dialog box so that the user can chose the image/images they want.
Then use the 'cv2.imread()' function to set the image selected by the user to the variable 'image'.
"""
ui.msgbox("Welcome, Please choose an image:")
f = ui.fileopenbox()
image = cv2.imread(f)

# blur = blur(image)
# rand_sep = rgb_separator(image, np.array([50, 50, 0]), np.array([255, 255, 255]))
# showimage(rand_sep)
# painted = inpaint(image, np.array([200, 200, 200]), np.array([255, 255, 255]))
circles = fillwhitecirles(image)
showimage(circles)