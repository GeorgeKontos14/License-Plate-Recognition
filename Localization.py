import cv2
import numpy as np
import matplotlib.pyplot as plt

def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    # TODO: Replace the below lines with your code.
    masked = masked_image(image, 15,30,100,200,100,255)
    #plt.imshow(masked)
    #plt.show()
    return crop_plates(masked)

def masked_image(image, minH, maxH, minS, maxS, minV, maxV):
    """
    This function applies a color to mask to an image in order to make everything with 
    a color outside of the given range. This is used to have an image that only shows the
    color of the license plate. The parameters minH, maxH, minS, maxS, minV, maxV determine
    the desired color range in HSV format.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    colorMin = np.array([minH, minS, minV])
    colorMax = np.array([maxH, maxS, maxV])

    mask = cv2.inRange(hsv, colorMin, colorMax)
    masked = np.copy(hsv)
    masked[mask == 0] = [0,0,0]

    return masked

def crop_plates(masked):
    """
    This function, given a masked image, crops the parts of the image that contain license plates.
    It returns a list of the cropped images.
    """
    bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = np.copy(gray)
    binary[gray!=0] = 255
    images = []
    i = 0
    images = []
    while i < binary.shape[0]:
        if len(images) == 2:
            break
        j = 0
        while j < binary.shape[1]:
            if binary[i][j] != 0:
                old_i = i
                while np.count_nonzero(binary[i]) != 0:
                    i += 1
                a = 0
                while np.count_nonzero(binary[old_i:i, a]) == 0:
                    a += 1
                old_a = a
                while np.count_nonzero(binary[old_i:i, a]) != 0:
                    a += 1

                if a-old_a >= 90:
                    images.append(gray[old_i:i, old_a:a])
                    binary[old_i:i, old_a:a] = 0
                    i = old_i + 1
                    j = 0
            j += 1
        i += 1
    print(len(images))
    plotImage(images[1], "Image", cmapType="gray")
    return images

# Displays a given RGB image using matplotlib.pyplot
def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()