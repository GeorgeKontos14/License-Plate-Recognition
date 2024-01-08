import cv2
import numpy as np
import os
import Helpers

def segment_and_recognize(plate_images):
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

	recognized_plates = [None, None, None]
	return recognized_plates

def segment(plate, out=None, binary = False):
	"""
	Given an image of a plate, it segments eacg character
	"""
	cleared = np.copy(plate)
	if not binary:
		gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
		background, _ = Helpers.isodata_thresholding(gray)
		#background, _ = Helpers.adaptive_thresholding(gray, 25, 30) - Does not work well; perhaps try better parameters
		#Helpers.plotImage(background, "Background", cmapType="gray")
		cleared = np.copy(background)
	
	if (out):
		cv2.imwrite(out, cleared)
	
	height, length = cleared.shape
	top = int(0.15*height)
	bottom = int(0.85*height)
	left = int(0.05*length)
	right = int(0.95*length)
	cleared = cleared[top:bottom, left:right]
	Helpers.plotImage(cleared, cmapType="gray")
	middle = int(plate.shape[0]/2)
	upper = middle+int(0.25*cleared.shape[0])
	lower = middle-int(0.25*cleared.shape[0])
	characters = []
	i = 0
	while i < cleared.shape[1]: # and len(characters) < 6:
		column = cleared[:, i]
		# Discard columns without enough white pixels
		if np.count_nonzero(column) <= 8:
			i += 1
			continue
		# Find the area of the continuous white pixels
		old_i = i
		while np.count_nonzero(cleared[:, i]) > 5:
			i += 1
			if i == cleared.shape[1]:
				break
		letter = cleared[:, old_i:i]
		#Helpers.plotImage(letter, cmapType="gray")
		whites = np.count_nonzero(letter)

		# Check if the character is a dot or a dash;
		# Dots appear in the beginning of the plates
		# due to shadows. We check by examining how many 
		# of the white pixels are near the middle.
		if np.count_nonzero(letter[lower:upper, :]) >= 0.8*whites:
			continue
		else:
			characters.append(letter)
		i += 1

	return characters


def can_be_dash(chars_length, dashes_length):
	"""
	Returns true if it is possible that the next letter
	detected in the plate can be a dash, according 
	to all the possible formats for the dutch license
	plates.
	"""
	if dashes_length >= 2:
		return False
	if chars_length == 2 and dashes_length == 0:
		return True
	if chars_length == 1 and dashes_length == 0:
		return True
	if chars_length == 3 and dashes_length == 0:
		return True
	if chars_length == 4 and dashes_length == 1:
		return True
	if chars_length == 5 and dashes_length == 1:
		return True
	if chars_length == 3 and dashes_length == 1:
		return True
	return False