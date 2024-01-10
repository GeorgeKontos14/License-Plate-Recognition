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
		return
	
	height, length = cleared.shape
	top = int(0.15*height)
	bottom = int(0.85*height)
	left = int(0.05*length)
	right = int(0.95*length)
	cleared = cleared[top:bottom, left:right]
	Helpers.plotImage(cleared, cmapType="gray")
	characters = []
	limits = []
	i = 0
	while i < cleared.shape[1]:
		column = cleared[:, i]
		# Discard columns without enough white pixels
		if np.count_nonzero(column) <= 8:
			i += 1
			continue
		# Find the area of the continuous white pixels
		old_i = i
		while np.count_nonzero(cleared[:, i:i+3]) > 10:
			i += 1
			if i == cleared.shape[1]:
				break
		j = old_i
		while np.count_nonzero(cleared[:, j-3:j]) > 10:
			j -= 1
			if j == 0:
				break
		letter = cleared[:, j:i]
		
		# Check if the character is a dot or a dash;
		# Dots appear in the beginning of the plates
		# due to shadows. We check by examining how many 
		# of the white pixels are near the middle.
		if is_dash(letter):
			continue
		else:
			if letter.shape[1] >= 3:
				characters.append(letter)
				limits.append((j, i))
		i += 1
	return merge_or_split(characters, limits, cleared)

def is_dash(letter):
	whites = np.count_nonzero(letter)
	middle = int(letter.shape[0]/2)
	upper_mid = middle+int(0.1*letter.shape[0])
	lower_mid = middle-int(0.1*letter.shape[0])
	upper = middle+int(0.2*letter.shape[0])
	lower = middle-int(0.2*letter.shape[0])
	return np.count_nonzero(letter[lower:middle]) > 0.7*whites or np.count_nonzero(letter[lower_mid:upper_mid]) > 0.7*whites or np.count_nonzero(letter[middle:upper]) > 0.7*whites

def clear_top_bottom(binary):
	height, length = binary.shape
	top_white = i = 0
	bottom_white = j = height-1
	while i < j:
		if np.count_nonzero(binary[i]) > 0.9*length:
			top_white = i
		if np.count_nonzero(binary[j]) > 0.9*length:
			bottom_white = j
		i += 1
		j -= 1
	return binary[top_white:bottom_white]

def merge_or_split(characters, limits, plate):
	avg = 0
	res = []
	for char in characters:
		avg += char.shape[1]
	avg = avg/len(characters)
	i = 0
	while i < len(characters):
		char = characters[i]
		if char.shape[1] > 1.6* avg:
			mid = int(char.shape[1]/2)
			res.append(char[:, :mid])
			res.append(char[:, mid:])
		elif char.shape[1] < 0.4* avg:
			if i == len(characters)-1:
				continue
			next = characters[i+1]
			if next.shape[1] < 0.4*avg:
				res.append(plate[limits[i][0]:limits[i+1][1]])
		else:
			res.append(char)
		i += 1
	return res
