import numpy as np
import cv2
from kd_tree import KDTree

def difference_score(img: np.ndarray, reference_character: np.ndarray) -> np.array:
    """ 
    """
    score: int = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (
                (img[i, j] == 0 and reference_character[i, j] > 0) or
                (img[i, j] > 0 and reference_character[i, j] == 0)
            ):
                score += 1
    
    return score

def give_label_lowest_score(img: np.ndarray, reference_characters: list) -> str:
    """
    """
    min_score: int = np.inf
    min_score_char: int = 0

    for entry in reference_characters:
        score = difference_score(img, entry.img)
        if score < min_score:
            min_score = score 
            min_score_char = entry.label
    
    return min_score_char

def calculate_perimeter_area_vector(img: np.ndarray) -> np.ndarray:
    """
    """
    binary_image: np.ndarray = (img != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    perimeter: float = cv2.arcLength(contours[0], closed=True)
    area: float = cv2.contourArea(contours[0])

    return np.array([perimeter, area])

# def recognise_character(kd_tree: KDTree, img: np.ndarray, k: int):
#     """
#     """
#     perimeter_area_vector: np.ndarray = calculate_perimeter_area_vector(img)
#     k_nearest_points: list = kd_tree.get_k_nearest_points(perimeter_area_vector, k)
#     pred: str = give_label_lowest_score(img, k_nearest_points)

#     return pred

def recognise_character(data_instances: list, img: np.ndarray):
    pass