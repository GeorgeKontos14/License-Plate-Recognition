import cv2
import time
import Localization
import plate_rotation
import Recognize
import Segment
import Helpers
import numpy as np

def CaptureFrame_Process(file_path, sample_frequency, save_path, reference_characters, show=True):
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
    start = time.time()
    data = process_video(file_path, sample_frequency, reference_characters)
    split_scenes(data, save_path)
    end = time.time()-start
    print(end)
    pass

def process_video(file_path: str, sample_frequency: int, reference_characters) -> list:
    """
    Given a video and the sampling frequency, performs the required analysis on the video's frames
    """

    # Open the video and initialize variables
    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_FPS, sample_frequency)
    if cap.isOpened()== False: 
        print("Error opening video stream or file")
    counter: int = 0
    start_time: float = time.time()
    rows: list = []

    # Iterate the frames of the video with the specified sample frequency
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # Run the pipeline for the given plate
            plates: list = Localization.plate_detection(frame)
            for plate in plates:
                if plate.shape[0]*plate.shape[1] > 0.8*frame.shape[0]*frame.shape[1]:
                    continue
                try:
                    rotated: np.ndarray = plate_rotation.rotation_pipeline(plate)
                except Exception:
                    continue
                if rotated is None:
                    continue
                
                scores, output = Recognize.segment_and_recognize(rotated, reference_characters)
                if len(scores) == 0:
                    continue

                # Add the data for the given frame on the list of rows
                end: float = time.time()-start_time
                if len(output) == 6:
                    rows.append((scores, output, counter, end))

            # Move to the next frame
            counter += sample_frequency
            cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

    # Return the video data
    return rows

def hamming_distance(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1,c2 in zip(s1, s2))

def split_scenes(data: list, save_path: str):
    """
    Given a list of data, split it to scenes and calculate the majority vote for the data of that scene 
    """
    # Open the writer to the csv file
    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")
    scene_outputs: list = []
    scene_scores: list = []
    scene_frame = data[0][2]
    scene_time = data[0][3]
    comp: str = data[0][1]
    for row in data:
        current_output: str = row[1]
        if hamming_distance(current_output, comp) > 2:
            out = Recognize.majority_characterwise(scene_outputs, scene_scores)
            to_write = out+','+str(scene_frame)+','+str(scene_time)+'\n'
            output.write(to_write)
            scene_outputs = []
            scene_scores = []
            scene_frame = row[2]
            scene_time = row[3]
        comp = current_output
        scene_outputs.append(current_output)
        scene_scores.append(row[0])
    return