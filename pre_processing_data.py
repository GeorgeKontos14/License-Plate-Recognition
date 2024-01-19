import numpy as np
import cv2
import os

from character_recognition import calculate_perimeter_area_vector

def reshape_img(img: np.ndarray) -> np.ndarray:
  """
  """
  minX = 1000000
  minY = 1000000
  maxX = -1
  maxY = -1

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i, j] > 0:
        if minX > j:
          minX = j
        if minY > i:
          minY = i
        if maxX < j:
          maxX = j
        if maxY < i:
          maxY = i

  margin_img: np.ndarray = put_margin(img, minX, minY, maxX, maxY)

  return margin_img

def put_margin(img: np.ndarray, minX: int, minY: int, maxX: int, maxY: int) -> np.ndarray:
  reshaped_size: np.ndarray = np.zeros((64, 64))

  central_x: int = (maxX - minX) // 2
  central_y: int = (maxY - minY) // 2

  top_x: int = 32 - central_x
  top_y: int = 32 - central_y

  for i in range(minY, maxY+1):
    for j in range(minX, maxX+1):
      reshaped_size[i-minY+top_y, j-minX+top_x] = img[i, j]

  return reshaped_size

def process_raw_data_for_characters(directory_path: str) -> list:
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    dataset = []

    for folder in folders:
        folder_path = directory_path + '/' + folder
        files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

        for file_name in files:
            img = 255 - cv2.imread(folder_path + '/' + file_name, cv2.IMREAD_GRAYSCALE)
            reshaped_img = reshape_img(img)

            p = calculate_perimeter_area_vector(reshaped_img)

            dataset.append((folder[0], p, reshaped_img))
    
    return dataset