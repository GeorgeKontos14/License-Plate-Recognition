import cv2
import numpy as np
import Helpers

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
    #Helpers.plotImage(image)
    #image = cv2.GaussianBlur(image, (5,5), 0)
    #masked = masked_image(image, 10,31,112,255,56,255) #75, 85, 70
    #masked = masked_image(image, 12, 29, 128, 254, 78, 251) #78, 100, 80
    #masked = masked_image(image, 14, 27,144,238, 99, 229) #78, 92, 60
    masked = masked_image(image, 13, 28, 136,246,89, 240) #78, 100, 80
    #denoise = close(masked)
    plates, boxes = crop_plates(masked, image)
    #Helpers.drawBoxes(boxes, image)
    #for plate in plates:
    #    Helpers.plotImage(plate, "")
    return plates

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

def crop_plates(masked, original):
    """
    This function, given a masked image, crops the parts of the image that contain license plates.
    It returns a list of the cropped images.
    """
    bgr = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = np.copy(gray)
    binary[gray!=0] = 255
    binary = Helpers.close(binary)
    i = 0
    images = []
    boxes = []
    while i < binary.shape[0]:
        if len(images) == 2:
            break
        j = 0
        while j < binary.shape[1]:
            if binary[i][j] != 0:
                old_i = i
                while np.count_nonzero(binary[i, max(0, j-100):min(binary.shape[1], j+100)]) != 0 and i < binary.shape[0]-1:
                    i += 1
                a = 0
                while np.count_nonzero(binary[old_i:i, a]) == 0 and a < binary.shape[1]-1:
                    a += 1
                old_a = a
                while np.count_nonzero(binary[old_i:i, a]) != 0 and a < binary.shape[1]-1:
                    a += 1
                if a-old_a >= 65: # and (a-old_a)*(i-old_i) >= 1750: #and a-old_a >= 3*(i-old_i) and a-old_a <= 6*(i-old_i):
                    #if np.count_nonzero(gray[old_i:i, old_a:a])/((i-old_i)*(a-old_a)) > 0.4:
                    images.append(original[old_i:i, old_a:a])
                    boxes.append(Helpers.BoundingBox(old_a, old_i, a-1, i-1))
                    binary[old_i:i, old_a:a] = 0
                    i = min(old_i + 1, binary.shape[0])
                    j = 0
                else:
                    i = min(old_i + 1, binary.shape[0])
                    j = min(a, binary.shape[1])
            j += 1
        i += 1
    if len(boxes) == 2:
        if boxes[0].is_close(boxes[1]):
            box = boxes[0].merge(boxes[1])
            boxes = []
            boxes.append(box)
            images = []
            images.append(original[box.y1:box.y2, box.x1:box.x2])
    return images, boxes