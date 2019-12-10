import cv2
from PIL import  Image
import matplotlib as plt
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
