import cv2
import numpy as np
import Helpers

def plate_detection(image: np.ndarray) -> list:
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
    masked: np.ndarray = masked_image(image, 13, 28, 136,246,89, 240)
    plates: list = crop_plates(masked, image)
    return plates

def masked_image(image: np.ndarray, minH: int, maxH: int, minS: int, maxS: int, minV: int, maxV: int) -> np.ndarray:
    """
    This function applies a color to mask to an image in order to make everything with 
    a color outside of the given range. This is used to have an image that only shows the
    color of the license plate. The parameters minH, maxH, minS, maxS, minV, maxV determine
    the desired color range in HSV format.
    """
    hsv: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    colorMin: np.ndarray = np.array([minH, minS, minV])
    colorMax: np.ndarray = np.array([maxH, maxS, maxV])

    mask: np.ndarray = cv2.inRange(hsv, colorMin, colorMax)
    hsv[mask == 0] = [0,0,0]
    return hsv

def crop_plates(masked: np.ndarray, original:np.ndarray) -> list:
    """
    This function, given a masked image, crops the parts of the image that contain license plates.
    It returns a list of the cropped images.
    """
    bgr: np.ndarray = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
    binary: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary[binary > 50] = 255
    binary = Helpers.close(binary)
    plates: list = []
    columns: np.ndarray = np.count_nonzero(binary, axis = 0)
    start: int = 0
    while start < len(columns):
        if columns[start] == 0:
            start += 1
            if start >= len(columns)-1:
                break
            continue
        end: int = start+1
        while columns[end] > 0:
            end += 1
            if end == len(columns):
                break
        if end - start < 65:
            start = end+1
            continue
        rows: np.ndarray = np.count_nonzero(binary[:, start:end], axis = 1)
        max_start: int = 0
        max_end: int = len(rows)-1
        max_count: int = 0
        cur_start: int = 0
        cur_end: int = 0
        while cur_start < len(rows)-1:
            if rows[cur_start] == 0:
                cur_start += 1
                if cur_start >= len(rows)-1:
                    break
                continue
            cur_end = cur_start+1
            while rows[cur_end] > 0:
                cur_end += 1
                if cur_end >= len(rows):
                    break
            cur_count = np.sum(rows[cur_start:cur_end])
            if cur_count > max_count:
                max_start = cur_start
                max_end = cur_end
                max_count = cur_count
            cur_start = cur_end+1
        plates.append(original[max_start:max_end, start:end])
        start = end+1
        if len(plates) == 2:
            break
    return plates