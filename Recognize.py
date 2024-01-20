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
	chars, dashes = Segment.segment(plate_image, show=True)
	if chars is None:
		#print('None')
		return [], ''
	return get_license_plate_number(reference_characters, chars)

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
	
	return ''.join(min(vote, key=vote.get) for vote in votes)