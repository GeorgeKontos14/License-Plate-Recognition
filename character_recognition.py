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

def recognise_character(data_references: list, img: np.ndarray):
    descriptor: np.ndarray = sift_descriptor(img)

    min_dist: float = 10000000000000
    pred: str = None

    for reference in data_references:
        dist = euclidean_distance(reference[0], descriptor)

        if dist < min_dist:
            min_dist = dist
            pred = reference[1] 

    return pred

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

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Calculates the euclidean distance between two points

        Parameters:
        p1 (np.ndarray) 
        p2 (np.ndarray)

        Returns:
        float: The distance between the two given points
    """
    return np.sqrt(np.sum(np.square(p1 - p2)))