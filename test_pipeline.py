import cv2
import os
import numpy as np
import pandas as pd
import Localization
import Recognize
import plate_rotation
import Helpers

import matplotlib.pyplot as plt
from pre_processing_data import reshape_img
from character_recognition import sift_descriptor, recognise_character

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

def localize_and_rotate(frame):
    """
    Localizes plates in an image and rotates them
    """
    plates = Localization.plate_detection(frame)
    rotated = []
    for plate in plates:
        try:
            rotate = plate_rotation.rotation_pipeline(plate)
            rotated.append(rotate)
        except Exception:
            continue
    return rotated

def segment(plate):
    """
    Binarizes a plate and segments its characters
    """
    try:
        chars, dashes = Recognize.segment(plate)
        # for char in chars:
        #     Helpers.plotImage(char, cmapType="gray")
    except Exception:
        pass
    return chars, dashes

def iterate_dir(path, data=True):
    """
    Iterates a directory and runs the pipeline on it
    """
    if data:
        dataset = []
        folder_path = './dataset/Lab07-Dataset'
        files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]            
        
        for file_name in files:
            reference_img = cv2.imread(folder_path + '/' + file_name)
            if reference_img is not None:
                dataset.append((reference_img, file_name[0]))

    for filename in os.scandir(path):
        if filename.is_file():
            frame = read_image(filename.path, "")
            plates = localize_and_rotate(frame)
            print(filename.name)
            for plate in plates:
                chars, dashes = segment(plate)
                plate_num = ''

                if len(chars) != 6:
                    continue

                for char in chars:
                    plate_num += recognise_character(dataset, char)
                print(plate_num)


def adjast_size(m, n):
    while m % n != 0 and m < n:
        m += m % n
    return m

if __name__ == '__main__':
    path = "dataset/Frames/Category_II"
    iterate_dir(path)