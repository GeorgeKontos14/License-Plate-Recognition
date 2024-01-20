import numpy as np
import cv2

def difference_score(img: np.ndarray, reference_character: np.ndarray) -> np.array:
    """ 
    """
    score: int = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (
                (img[i, j] == 0 and reference_character[i, j] > 125) or
                (img[i, j] > 0 and reference_character[i, j] < 125)
            ):
                score += 1
    
    return score

def give_label_lowest_score(img: np.ndarray, reference_characters: list) -> str:
    """
    """
    min_score: int = np.inf
    min_score_char: int = 0


    for char in reference_characters:
        reference_character = cv2.resize(char[0], (img.shape[1], img.shape[0]))
        reference_character = cv2.cvtColor(reference_character, cv2.COLOR_BGR2GRAY)
        score = difference_score(img, reference_character)

        if score < min_score:
            min_score = score 
            min_score_char = char[1]
    
    return min_score, min_score_char

def recognise_character(reference_characters: list, img: np.ndarray):
    return give_label_lowest_score(img, reference_characters)

def sift_descriptor(image_interest_patch: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.zeros(128)
    h, w = image_interest_patch.shape[:2]

    if h % 16 != 0 or w % 16 != 0:
        raise ValueError("")
    ystep: int = w//4
    xstep: int = w//4

    idx: int = 0

    for i in range(4):
        for j in range(4):
            window: np.ndarray = image_interest_patch[i*ystep:((i+1)*ystep), j*xstep:((j+1)*xstep)]

            gx: float = np.gradient(window, axis = 1)
            gy: float = np.gradient(window, axis = 0)

            magnitude = magnitude = np.sqrt(np.square(gx)+np.square(gy))

            gx[gx == 0] = 0.000001
            orientations: float = np.arctan(gy/gx)
            angles: float = np.degrees(orientations) % 360
            quant_angles: int = np.floor(angles/45)
            hist: np.ndarray = np.zeros(8)

            for k in range(angles.shape[0]):
                for l in range(angles.shape[1]):
                    hist[int(quant_angles[k][l])] += magnitude[k][l]
            result[idx:(idx+8)] = hist
            idx += 8

    return result

def get_license_plate_number(reference_characters: list, chars: list) -> str:
    plate_num: str = ''
    xor_scores: list = []

    if len(chars) == 6:
        for char in chars:
            score, pred_char = recognise_character(reference_characters, char)
            plate_num += pred_char
            xor_scores.append(score)
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
            xor_scores.append(scores[i])

    return xor_scores, plate_num
