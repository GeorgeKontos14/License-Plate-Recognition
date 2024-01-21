import cv2
import numpy as np
import os
import Segment
import Helpers

from character_recognition import get_license_plate_number

def segment_and_recognize(plate_image: np.ndarray, reference_characters: list):
	"""
	In this file, you will define your own segment_and_recognize function.
	To do:
		1. Segment the plates character by character
		2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
		3. Recognize the character by comparing the distances
	Inputs:(One)
		1. plate_imgs: cropped plate images by Localization.plate_detection function
		type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
	Outputs:(One)
		1. recognized_plates: recognized plate characters
		type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
	Hints:
		You may need to define other functions.
	"""
	# chars, dashes = Segment.segment(plate_image, show=False)
	# if chars is None:
	# 	#print('None')
	# 	return [], ''
	
	bounding_boxes: list = character_segmentation(plate_image)

	return get_license_plate_number(reference_characters, bounding_boxes)

def majority_characterwise(scene_outputs: list, scene_scores: list) -> str:
	votes: list = [{} for i in range(6)]
	#print(scene_outputs, scene_scores)
	if len(scene_outputs) == 0:
		return None
	for characters, scores in zip(scene_outputs, scene_scores):
		i = 0
		for score, character in zip(scores, characters):
			if character in votes[i]:
				votes[i][character] += score
			else:
				votes[i][character] = score 

			i += 1
	
	return add_dashes(''.join(min(vote, key=vote.get) for vote in votes))



def add_dashes(output: str) -> str:
	"""
	Adds dashes to a given license plate
	"""
	prev: bool = output[0].isdigit()
	res: str = ''
	res += output[0]
	dashes: int = 0
	for i in range(1, 6):
		if dashes == 2:
			res += output[i]
			continue
		char: str = output[i]
		cur: bool = char.isdigit()
		if cur == prev:
			res += char
		elif cur != prev:
			res += '-' + char
			dashes += 1
		prev = cur
	if dashes == 1:
		if res[4] == '-':
			res = res[:2]+'-'+res[2:]
		else:
			res = res[:5]+'-'+res[5:]
	return res


def character_segmentation(plate_image: np.ndarray) -> list:
	plate: np.ndarray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
	plate = Helpers.isodata_thresholding(plate)
	plate = Segment.clear_top_bottom(plate)
	plate = np.array(plate, np.uint8)
	contours, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return []

	mean_area: float = 0
	plate_area: float = plate.shape[0] * plate.shape[1]
	for contour in contours:
		contour_area = cv2.contourArea(contour)
		if contour_area < plate_area * 0.5:
			mean_area += contour_area

	mean_area /= len(contours)
	threshold_area: float = mean_area * 0.5

	bounding_boxes: list = []

	for contour in contours:
		contour_area: float = cv2.contourArea(contour)
		if contour_area >= plate_area * 0.7:
			continue

		if contour_area > threshold_area:
			x, y, w, h = cv2.boundingRect(contour)
			bounding_boxes.append((plate[y:y+h, x:x+w], x))
	
	bounding_boxes.sort(key=lambda x: x[1])

	return bounding_boxes