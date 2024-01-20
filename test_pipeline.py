import cv2
import os
import numpy as np
import pandas as pd
import Localization
import Recognize
import plate_rotation
import Helpers

import matplotlib.pyplot as plt
from character_recognition import recognise_character

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

def iterate_dir(path, data=True):
    """
    Iterates a directory and runs the pipeline on it
    """
    reference_characters: list = read_reference_characters('./dataset/Lab07-Dataset')        

    for filename in os.scandir(path):
        if filename.is_file():
            frame = read_image(filename.path, "")
            plates = localize_and_rotate(frame)
            print(filename.name)
            for plate in plates:
                chars, dashes = Recognize.segment(plate, show=False)
                plate_num: str = get_license_plate_number(reference_characters, chars)
                print(plate_num)

def read_reference_characters(folder_path: str) -> list:
    reference_characters: list = []
    files: list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]            
    
    for file_name in files:
        reference_img: np.ndarray = cv2.imread(folder_path + '/' + file_name)
        if reference_img is not None:
            reference_characters.append((reference_img, file_name[0]))
    
    return reference_characters 

def get_license_plate_number(reference_characters: list, chars: list) -> str:
    plate_num: str = ''

    if len(chars) == 6:
        for char in chars:
            _, pred_char = recognise_character(reference_characters, char)
            plate_num += pred_char
    else:
        scores: list = []
        preds: list = []

        for char in chars:
            score, pred_char = recognise_character(reference_characters, char)
            scores.append(score)
            preds.append(pred_char)
        
        scores_max = np.argsort(scores)[:2].tolist()

        for i, ch in enumerate(preds):
            if i in scores_max:
                continue
            plate_num += ch

    return plate_num

if __name__ == '__main__':
    path = "dataset/Frames/Category_II"
    iterate_dir(path)