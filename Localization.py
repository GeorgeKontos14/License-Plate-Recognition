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
    return crop_plates(masked, 30)

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

def crop_plates(masked, threshold):
    """
    This function, given a masked image, crops the parts of the image that contain license plates.
    It returns a list of the cropped images.
    """
    bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = np.copy(gray)
    binary[gray!=0] = 255
    images = []
    """ 
    In order to identify where the plates are, we create a 3D histogram of the image, in terms of 
    x and y. We then find the bins(and their neighborhoods) that return the two maximum amount of
    ones, and then test whether they have enough ones to be considered a license plate.
    """

    hist = histogram(binary)
    highest= np.unravel_index(np.argmax(hist), hist.shape)
    oldRight = left = highest[0]-1
    oldLeft = right = highest[0]+1
    oldDown = up = highest[1]-1
    oldUp = down = highest[1]+1
    plate = binary[15*highest[0]:15*(highest[0]+1), 15*highest[1]:15*(highest[1]+1)]

    while oldRight != right or oldLeft != left or oldDown != down or oldUp != up:
        plate = binary[15*left:15*(right+2), 15*up:15*(down+2)]
        oldRight = right
        oldLeft = left
        oldDown = down
        oldUp = up
        if np.count_nonzero(hist[left-1, highest[1]]) >= 2 or np.count_nonzero(hist[left-2, highest[1]]) >= 2 or np.count_nonzero(hist[left-3, highest[1]]) >= 2:
            left -= 1
        if np.count_nonzero(hist[right+1, highest[1]]) >= 2 or np.count_nonzero(hist[right+2, highest[1]]) >= 2 or np.count_nonzero(hist[right+3, highest[1]]) >= 2:
            right += 1
        if np.count_nonzero(hist[highest[0], up-1]) >= 2 or np.count_nonzero(hist[highest[0], up-2]) >= 2 or np.count_nonzero(hist[highest[0], up-3]) >= 2:
            up -= 1
        if np.count_nonzero(hist[highest[0], down+1]) >= 2 or np.count_nonzero(hist[highest[0], down+2]) >= 2 or np.count_nonzero(hist[highest[0], down+3]) >= 2:
            down += 1
    if np.count_nonzero(plate) > threshold:
        x_min = plate.shape[0]
        y_min = plate.shape[1]
        x_max = 0
        y_max = 0
        for i in range(plate.shape[0]):
            for j in range(plate.shape[1]):
                if plate[i][j] != 0:
                    x_min = min(x_min, i)
                    y_min = min(y_min, j)
                    x_max = max(x_max, i)
                    y_max = max(y_max, j)
        images.append(plate[x_min:(x_max+1), y_min:(y_max+1)])
    hist[left:(right+2), up:(down+2)] = 0

    # Repeat for a second plate
    highest = np.unravel_index(np.argmax(hist), hist.shape)
    oldRight = left = highest[0]-1
    oldLeft = right = highest[0]+1
    oldDown = up = highest[1]-1
    oldUp = down = highest[1]+1
    plate = binary[15*highest[0]:15*(highest[0]+1), 15*highest[1]:15*(highest[1]+1)]

    while oldRight != right or oldLeft != left or oldDown != down or oldUp != up:
        plate = binary[15*left:15*(right+2), 15*up:15*(down+2)]
        oldRight = right
        oldLeft = left
        oldDown = down
        oldUp = up
        if np.count_nonzero(hist[left-1, highest[1]]) >= 2 or np.count_nonzero(hist[left-2, highest[1]]) >= 2 or np.count_nonzero(hist[left-3, highest[1]]) >= 2:
            left -= 1
        if np.count_nonzero(hist[right+1, highest[1]]) >= 2 or np.count_nonzero(hist[right+2, highest[1]]) >= 2 or np.count_nonzero(hist[right+3, highest[1]]) >= 2:
            right += 1
        if np.count_nonzero(hist[highest[0], up-1]) >= 2 or np.count_nonzero(hist[highest[0], up-2]) >= 2 or np.count_nonzero(hist[highest[0], up-3]) >= 2:
            up -= 1
        if np.count_nonzero(hist[highest[0], down+1]) >= 2 or np.count_nonzero(hist[highest[0], down+2]) >= 2 or np.count_nonzero(hist[highest[0], down+3]) >= 2:
            down += 1
    if np.count_nonzero(plate) > threshold:
        x_min = plate.shape[0]
        y_min = plate.shape[1]
        x_max = 0
        y_max = 0
        for i in range(plate.shape[0]):
            for j in range(plate.shape[1]):
                if plate[i][j] != 0:
                    x_min = min(x_min, i)
                    y_min = min(y_min, j)
                    x_max = max(x_max, i)
                    y_max = max(y_max, j)
        images.append(plate[x_min:(x_max+1), y_min:(y_max+1)])
    return images

def histogram(binary):
    binX = int(binary.shape[0]/15)
    binY = int(binary.shape[1]/15)
    res = np.zeros((binX, binY))
    for i in range(binX):
        for j in range(binY):
            res[i][j] = np.count_nonzero(binary[15*i:min(15*(i+1), binary.shape[0]), 15*j:min(15*(j+1), binary.shape[1])])

    return res 

# Displays a given RGB image using matplotlib.pyplot
def plotImage(img, title, cmapType=None):
    # Display image
    if (cmapType):
        plt.imshow(img, cmap=cmapType, vmin=0, vmax=255)
    else:
        plt.imshow(img, vmin=0, vmax=255)
    plt.title(title)
    plt.show()