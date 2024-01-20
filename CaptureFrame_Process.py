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
from character_recognition import sift_descriptor

def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """

    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    """cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()"""

    # TODO: Implement actual algorithms for Localizing Plates
    #plates = Localization.plate_detection(frame)
    # TODO: Implement actual algorithms for Recognizing Characters

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    output.write("XS-NB-23,34,1.822\n")
    # output.write("YOUR,STUFF,HERE\n")
    # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass

def iterate_dir(path):
    dataset = []
    folder_path = './dataset/Lab07-Dataset'
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    # for file_name in files:
    #     img = 255 - cv2.imread(folder_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
    #     # img = cv2.resize(img, ())

    #     descriptor = sift_descriptor(img)
    #     dataset.append((descriptor, file_name[0]))

    for filename in os.scandir(path):
        if filename.is_file():
            print(filename.name)
            frame = cv2.imread(filename.path)
            plates = Localization.plate_detection(frame)
            
            for plate in plates:
                plt.imshow(plate)
                plt.show()

                try:
                    rotated = plate_rotation.rotation_pipeline(plate)
                except Exception:
                    continue
                #Helpers.plotImage(rotated)
                try:
                    chars, dashes = Recognize.segment(rotated)
                    for char in chars:
                        Helpers.plotImage(char, cmapType="gray")

                except Exception:
                    pass
    return plates

def adjast_size(m, n):
    while m % n != 0 and m < n:
        m += m % n
    return m