import os
import numpy as np
import pandas as pd
import Localization
import Recognize
import plate_rotation
import Helpers

from character_recognition import get_license_plate_number
from pre_processing_data import read_image, read_reference_characters

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
                scores, plate_num = Recognize.segment_and_recognize(plate, reference_characters)
                # print(scores)
                print(plate_num)
            Helpers.plotImage(frame)

if __name__ == '__main__':
    path = "dataset/Frames/Category_II"
    iterate_dir(path)