import cv2
import os
import numpy as np
import pandas as pd
import Localization
import Recognize
import plate_rotation
import Helpers

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
    path = "dataset/Frames/Category_II"
    iterate_dir(path)
    #frame = cv2.imread("dataset/Frames/Category_I/plate1.jpg")
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame[frame >= 125] = 255
    #frame[frame < 125] = 0
    #Helpers.plotImage(frame, cmapType="gray")
    #characters = Recognize.segment(frame)
    #for char in characters:
    #    Helpers.plotImage(char, cmapType="gray")
    #print(len(characters))
    #plates = Localization.plate_detection(frame)
    #for plate in plates:
        #Helpers.plotImage(plate)
    #    rotated = plate_rotation.rotation_pipeline(plate)
        #Helpers.plotImage(rotated)
    #   chars = Recognize.segment(rotated)
    #    for char in chars:
    #        Helpers.plotImage(char, cmapType="gray")
            

    #    print(len(chars))
    #    print(dashes)
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
    for filename in os.scandir(path):
        if filename.is_file():
            print(filename.name)
            if filename.name == "plate14.jpg":
                continue
            frame = cv2.imread(filename.path)
            plates = Localization.plate_detection(frame)
            for plate in plates:
                #Helpers.plotImage(plate)
                rotated = plate_rotation.rotation_pipeline(plate)
                #Recognize.segment(rotated)
                #Helpers.plotImage(rotated)
                chars, dashes = Recognize.segment(rotated)
                for char in chars:
                    Helpers.plotImage(char, cmapType="gray")
    return plates