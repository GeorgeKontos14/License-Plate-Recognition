import cv2
import time
import Localization
import plate_rotation
import Recognize
import Segment
import Helpers

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
    counter = 0
    prev = None
    start_scene = 1
    scene_outputs = []
    scene_scores = []
    start_time = time.time()
    frame_no = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if counter == 0:
                prev = frame
            if show:
                cv2.imshow('Frame', frame)
            if scene_change(prev, frame, start_scene, counter, sample_frequency):
                # TODO: Majority vote - Also consider similarity with previous plate
                #print(scene_outputs)
                pred_plate = Recognize.majority_characterwise(scene_outputs, scene_scores)
                # print(pred_plate)
                if pred_plate is None or len(pred_plate) == 0:
                    continue
                time_stamp = time.time()-start_time
                output.write(pred_plate+','+str(counter)+","+str(time_stamp)+"\n")
                start_scene = counter+sample_frequency
                scene_outputs = []
                scene_scores = []
            score, out = run_scene_pipeline(frame, reference_characters)
            if out != None:
                frame_no = counter
                scene_outputs.append(out)
                scene_scores.append(score)
            prev = frame
            counter += sample_frequency
            #print(counter)
            cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    pred_plate = Recognize.majority_characterwise(scene_outputs, scene_scores)
    if pred_plate is None:
        return
    time_stamp = time.time()-start_time
    output.write(pred_plate+','+str(counter)+","+str(time_stamp)+"\n")

    pass

def scene_change(before, after, start_scene, before_no, frequency):
    """
    Returns true if the two frames are from different scenes; false otherwise
    """
    # Histogram approach:
    hist_before = cv2.calcHist([before], [0,1,2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_before, hist_before, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_after = cv2.calcHist([after], [0,1,2], None, [256,256,256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_after, hist_after, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    comp = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CORREL)
    # if comp < 0.1:
    #     return True
    if before_no+frequency-start_scene > 24 and comp < 0.35:
        return True
    return False

def run_scene_pipeline(frame, reference_characters):
    """
    Gets a scene as an array of frames as input and returns the output for each frame as output
    """
    output = ""
    plates = Localization.plate_detection(frame)
    if len(plates) == 0:
        return None, None
    for plate in plates:
        if plate.shape[0]*plate.shape[1] > 0.8*frame.shape[0]*frame.shape[1]:
            return None, None
        #Helpers.plotImage(plate)
        try:
            rotated = plate_rotation.rotation_pipeline(plate)
        except Exception:
            return None, None
        if rotated is None:
            continue
        #Helpers.plotImage(rotated)
        scores, output = Recognize.segment_and_recognize(rotated, reference_characters)
        #print(scores)
        # print(output)
        #print(scores)
        if len(scores) == 0:
            return None, None

    return scores, output
