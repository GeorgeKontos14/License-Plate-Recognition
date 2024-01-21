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
    min_score_char: str = 0

    img[img < 125] = 0
    img[img >= 125] = 255 

    for char in reference_characters:
        reference_character = cv2.resize(char[0], (img.shape[1], img.shape[0]))
        reference_character = cv2.cvtColor(reference_character, cv2.COLOR_BGR2GRAY)
        reference_character[reference_character < 125] = 0
        reference_character[reference_character >= 125] = 255

        xor_diff = cv2.bitwise_xor(img, reference_character)
        score = np.sum(xor_diff) / (img.shape[0] * img.shape[1] * 255.0)


        if score < min_score:
            min_score = score 
            min_score_char = char[1]
    
    return min_score, min_score_char

def recognise_character(reference_characters: list, img: np.ndarray):
    return give_label_lowest_score(img, reference_characters)

def get_license_plate_number(reference_characters: list, chars: list) -> str:
    plate_num: str = ''
    xor_scores: list = []

    if len(chars) == 6:
        for char in chars:
            score, pred_char = recognise_character(reference_characters, char[0])
            plate_num += pred_char
            xor_scores.append(score)
    # else:
        # scores: list = []
        # preds: list = []

        # for char in chars:
        #     score, pred_char = recognise_character(reference_characters, char)
        #     scores.append(score)
        #     preds.append(pred_char)
        
        # scores_max = np.argsort(scores)[:2].tolist()

        # for i, ch in enumerate(preds):
        #     if i in scores_max:
        #         continue
        #     plate_num += ch
        #     xor_scores.append(scores[i])

    return xor_scores, plate_num

