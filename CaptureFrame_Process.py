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
    
    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_FPS, sample_frequency)
    if cap.isOpened()== False: 
        print("Error opening video stream or file")
    counter: int = 0
    prev: np.ndarray = None
    start_scene: int = 1
    scene_outputs: list = []
    scene_scores: list = []
    start_time: float = time.time()
    frame_no: int = 1
    min_score: float = 6
    last_out: str = ''
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if counter == 0:
                prev = frame
                # print("Set prev")
            if show:
                cv2.imshow('Frame', frame)
            if scene_change(prev, frame, start_scene, counter-sample_frequency, sample_frequency):
                # print("Change")
                # Helpers.plotImage(prev)
                # Helpers.plotImage(frame)
                pred_plate: str = Recognize.majority_characterwise(scene_outputs, scene_scores)
                start_scene = counter
                scene_outputs = []
                scene_scores = []
                min_score = 6
                if pred_plate is None or len(pred_plate) == 0 or pred_plate == last_out:
                    prev = frame
                    continue
                time_stamp: float = time.time()-start_time
                output.write(pred_plate+','+str(frame_no)+","+str(time_stamp)+"\n")
                last_out = pred_plate
            score, out = run_scene_pipeline(frame, reference_characters)
            if out != None:
                if np.sum(score) < min_score:
                    frame_no = counter
                    min_score = np.sum(score)
                scene_outputs.append(out)
                scene_scores.append(score)
            prev = frame
            counter += sample_frequency
            cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    pred_plate = Recognize.majority_characterwise(scene_outputs, scene_scores)
    if pred_plate is None:
        return
    time_stamp = time.time()-start_time
    output.write(pred_plate+','+str(frame_no)+","+str(time_stamp)+"\n")

    pass

def scene_change(before: np.ndarray, after: np.ndarray, start_scene: int, before_no: int, frequency: int) -> bool:
    """
    Returns true if the two frames are from different scenes; false otherwise
    """
    # Histogram approach:
    hist_before = cv2.calcHist([before], [0,1,2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_before, hist_before, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_after = cv2.calcHist([after], [0,1,2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_after, hist_after, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    comp: float = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CORREL)
    if before_no+frequency-start_scene > 24 and comp < 0.35:
        return True
    return False

def run_scene_pipeline(frame: np.ndarray, reference_characters):
    """
    Gets a scene as an array of frames as input and returns the output for each frame as output
    """
    output: str = ""
    plates: list = Localization.plate_detection(frame)
    if len(plates) == 0:
        return None, None
    for plate in plates:
        if plate.shape[0]*plate.shape[1] > 0.8*frame.shape[0]*frame.shape[1]:
            return None, None
        try:
            rotated: np.ndarray = plate_rotation.rotation_pipeline(plate)
        except Exception:
            return None, None
        if rotated is None:
            continue
        scores, output = Recognize.segment_and_recognize(rotated, reference_characters)
        if len(scores) == 0:
            return None, None

    return scores, output
