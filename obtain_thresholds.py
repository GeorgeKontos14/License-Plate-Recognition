import numpy as np
import cv2

import os

def k_fold_cross_validation(X: np.ndarray, k: int, l: int = 2) -> None:
    """ Performs k-fold cross-validation for Gaussian Distributed HSI bounds

        Parameters:
        X (np.ndarray): all image data as a numpy matrix
        k (int): number of folds
        l (int): extend of standard deviation

        Returns:
        None
    """
    fold_size: int = X.shape[0] // k
    start_idx: int = 0

    for i in range(1, k + 1):
        input_arrangement: np.ndarray = np.ones((X.shape[0]), dtype=bool)
        
        for j in range(start_idx, start_idx + fold_size):
            input_arrangement[j] = False
        
        train_data: np.ndarray = X[input_arrangement]
        test_data = np.ndarray = X[input_arrangement == False]

        hsi_bounds: np.ndarray = get_hsi_bounds(train_data, l)
        acc: float = 0

        for x in test_data:
            if (
                x[0] >= hsi_bounds[0] and x[0] <= hsi_bounds[1]
                and x[1] >= hsi_bounds[2] and x[1] <= hsi_bounds[3]
                and x[2] >= hsi_bounds[4] and x[2] <= hsi_bounds[5]
            ):
                acc += 1

        print(f'Fold #{i}') 
        print('For bounds:')
        print(hsi_bounds)
        print(f'Accuracy: {acc / test_data.shape[0]}')

def read_img_data_from_disk(folder_path: str) -> np.ndarray:
    """ Reads image data from disk

        Parameters:
        folder_path (str): The folder path where all the data is saved
        
        Returns:
        np.ndarray: All image data as a numpy matrix
    """
    X: list = []
    
    file_names: list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_name in file_names:
        img: np.ndarray = cv2.imread(folder_path + '/' + file_name)
        img_hsi: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        X.append(img_hsi.reshape(img_hsi.shape[0] * img_hsi.shape[1], 3))

    X_train: list = []

    for x in X:
        for i in range(x.shape[0]):
            X_train.append(x[i])

    return np.array(X_train)

def get_hsi_bounds(X: np.ndarray, l: int=2) -> np.ndarray:
    """ Gets the HSI bound value for hue, saturation, and intensity

        Parameters:
        X (np.ndarray): All image data as a numpy matrix
        l (int): extend of standard deviation

        Returns:
        np.ndarray: contains all HSI bounding values [minH, maxH, minS, maxS, minV, maxV]
    """
    hue_mean: float = np.mean(X[:, 0])
    saturation_mean:float = np.mean(X[:, 1])
    intensity_mean: float = np.mean(X[:, 2])

    hue_sd: float = get_sd_of_data(X[:, 0], hue_mean)
    saturation_sd: float = get_sd_of_data(X[:, 1], saturation_mean)
    intensity_sd: float = get_sd_of_data(X[:, 2], intensity_mean)

    hsi_bounds: np.ndarray = np.array([
        np.ceil(hue_mean - l * hue_sd), np.ceil(hue_mean + l * hue_sd),
        np.ceil(saturation_mean - l * saturation_sd), np.ceil(saturation_mean + l * saturation_sd),
        np.ceil(intensity_mean - l * intensity_sd), np.ceil(intensity_mean + l * intensity_sd)
    ])

    hsi_bounds[hsi_bounds > 255] = 255
    hsi_bounds[hsi_bounds < 0] = 0

    return hsi_bounds

def get_sd_of_data(x: np.ndarray, x_mean: float) -> float:
    """ Calculates the standard deviation of the given array of numerical elements

        Parameters:
        x (np.ndarray): data array
        x_mean (float): the mean value of the data array

        Returns:
        float: The standard deviation of the given data array
    """
    return np.sqrt(np.sum(np.square(x - x_mean)) / x.shape[0])

