import argparse
import os
import CaptureFrame_Process
import numpy
from pre_processing_data import read_reference_characters


# define the required arguments: video path(file_path), sample frequency(second), saving path for final result table
# for more information of 'argparse' module, see https://docs.python.org/3/library/argparse.html
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='dataset/trainingvideo.avi')
	parser.add_argument('--output_path', type=str, default='dataset/Output.csv')
	parser.add_argument('--sample_frequency', type=int, default=2)
	args = parser.parse_args()
	return args


# In this file, you need to pass three arguments into CaptureFrame_Process function.
if __name__ == '__main__':
	args = get_args()
	if args.output_path is None:
		output_path = os.getcwd()
	else:
		output_path = args.output_path
	file_path = args.file_path
	#file_path = 'dataset\TrainingSet\Categorie II\Video225.avi'
	file_path = 'dataset\\trainingvideo.avi'
	# sample_frequency = args.sample_frequency
	sample_frequency = 5
	reference_characters: list = read_reference_characters('./dataset/Lab07-Dataset')
	CaptureFrame_Process.CaptureFrame_Process(file_path, sample_frequency, output_path, reference_characters, show=False)
