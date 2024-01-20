import numpy as np
import cv2
import os

import Helpers

def read_image(path, filename, plot=False, gray=False, binary=False):
    """
    Reads an image from the dataset
    """
    frame = cv2.imread(path+filename)
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if binary:
            frame[frame >= 125] = 255
            frame[frame < 125] = 0
    if plot:
        if gray:
            Helpers.plotImage(frame, cmapType="gray")
        else:
            Helpers.plotImage(frame)
    return frame

def read_reference_characters(folder_path: str) -> list:
    reference_characters: list = []
    files: list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]            
    
    for file_name in files:
        reference_img: np.ndarray = cv2.imread(folder_path + '/' + file_name)
        if reference_img is not None:
            reference_characters.append((reference_img, file_name[0]))
    
    return reference_characters 