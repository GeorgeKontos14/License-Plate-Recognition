import cv2
import numpy as np
import os
import Segment
import Helpers

def segment_and_recognize(plate_image: np.ndarray) -> list:
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
	recognized_characters: list = []
	chars, dashes = Segment.segment(plate_image)
	
	return recognized_characters


def add_dashes(output):
	"""
	Adds dashes to a given license plate
	"""
	prev = output[0].isdigit()
	res = ''
	res += output[0]
	dashes = 0
	for i in range(1, 6):
		if dashes == 2:
			res += output[i]
			continue
		char = output[i]
		cur = char.isdigit()
		if cur == prev:
			res += char
		elif cur != prev:
			res += '-' + char
			dashes += 1
		prev = cur
	if dashes == 1:
		res = res[:5]+'-'+res[5:]
	return res