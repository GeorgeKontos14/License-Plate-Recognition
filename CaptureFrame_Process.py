import cv2
import time

def CaptureFrame_Process(file_path, sample_frequency, save_path, show=True):
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
    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_FPS, sample_frequency)
    if cap.isOpened()== False: 
        print("Error opening video stream or file")
    counter = 0
    prev = None
    start_scene = 1
    scene = []
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if counter == 0:
                prev = frame
            if show:
                cv2.imshow('Frame', frame)
            if scene_change(prev, frame, start_scene, counter, sample_frequency):
                # TODO: Run Localization - Rotation - Segmentation - Recognition Pipeline for all frames of the scene
                # TODO: Majority vote - Also consider similarity with previous plate?
                time_stamp = time.time()-start_time
                output.write("XS-NB-23,"+str(counter)+","+str(time_stamp)+"\n")
                start_scene = counter+sample_frequency
                scene = [frame]
            else:
                scene.append(frame)
            prev = frame
            counter += sample_frequency
            cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    # TODO: Implement actual algorithms for Localizing Plates

    # TODO: Implement actual algorithms for Recognizing Characters

    # output = open(save_path, "w")
    # output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    # output.write("XS-NB-23,34,1.822\n")
    # output.write("YOUR,STUFF,HERE\n")
    # TODO: REMOVE THESE (above) and write the actual values in `output`

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