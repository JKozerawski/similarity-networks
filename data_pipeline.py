import os.path
from os import walk
import random
import re
import numpy as np
from glob import glob
import argparse

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
FLAGS = []

def get_images(image_dir, sub_dir, extensions):
	file_list = []
	dir_name = os.path.basename(sub_dir)
   	
	for extension in extensions:
		file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
		file_list.extend(glob(file_glob))
	if not file_list:
		print 'No files found'
		return
	if len(file_list) < 20:
		print 'WARNING: Folder has less than 20 images, which may cause issues.'
	elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
		print 'WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS)
	return file_list

def get_random_image_tuples(sim_label, list_pos, list_neg, no_of_pos = 1, no_of_neg = 0):

	neg_images = []

	# If the query image is of the same class as the positive examples:
	if(sim_label == 1):
		pos_images = np.random.choice(list_pos, no_of_pos+1, replace=False)
		for i in xrange(no_of_neg):
			neg_images.append(np.random.choice(list_neg[i]))
	
	# If the query image is from different class than the positive examples:
	else:
		pos_images = np.random.choice(list_pos, no_of_pos ,replace=False)
		for i in xrange(no_of_neg+1):
			neg_images.append(np.random.choice(list_neg[i]))

	
	images = [str(i) for i in pos_images]+[str(i) for i in neg_images]
	return [ "/".join(img.split("/")[-2:]) for img in images]

def create_image_lists(image_dir, testing_percentage, validation_percentage, positive_examples_per_category, positives_to_negatives_ratio, no_of_positive_inputs, no_of_negative_inputs):
	"""Builds a list of training images from the file system.


	Args:
	describe args

	Returns:
	describe output
	"""
	negative_examples_per_category = int(positive_examples_per_category/positives_to_negatives_ratio)		# how many image pairs per category

	# Check if the directory exists:
	if not os.path.isdir(image_dir):
		print "Image directory '" + image_dir + "' not found."
		return None

	results = []	# empty list for the image tuples
	sub_dirs = [x[0] for x in walk(image_dir)]

	# The root directory comes first, so skip it.
	sub_dirs = sub_dirs[1:]

	# Enumerate through all the categories:
	for sub_dir_i, sub_dir in enumerate(sub_dirs):
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']	# allowed image extensions
		file_list = get_images(image_dir, sub_dir, extensions)
		dir_name = os.path.basename(sub_dir)
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())	# what is the label name

		# get positive examples (same class) with label "1":
	    	for k in xrange(positive_examples_per_category):
			neg_indices = np.random.choice([x for x in xrange(len(sub_dirs)) if x != sub_dir_i], no_of_negative_inputs, replace=False)
			image_tuples = get_random_image_tuples(1, file_list,  [get_images(image_dir, sub_dirs[j], extensions) for j in neg_indices], no_of_pos = no_of_positive_inputs, no_of_neg = no_of_negative_inputs)
			results.append(image_tuples+["1"])
		# get negative examples (different class) with label "0":
		for k in xrange(negative_examples_per_category):
			neg_indices = np.random.choice([x for x in xrange(len(sub_dirs)) if x != sub_dir_i], no_of_negative_inputs+1, replace=False)
			image_tuples = get_random_image_tuples(0, file_list,  [get_images(image_dir, sub_dirs[j], extensions) for j in neg_indices], no_of_pos = no_of_positive_inputs, no_of_neg = no_of_negative_inputs)
			results.append(image_tuples+["0"])

	# Shuffle the list randomly:
	random.shuffle(results)	

	# Split the tuples into training, validation and testing:
	test_index = int(testing_percentage/100*len(results))
	val_index = test_index+int(validation_percentage/100*len(results))
	testing_data = results[:test_index]
	validation_data = results[test_index:val_index]
	training_data = results[val_index:]
	    
	return training_data, validation_data, testing_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--image_dir',
		type=str,
		default='',
		help='Path to folders of labeled images.'
	)
	parser.add_argument(
		'--testing_percentage',
		type=int,
		default=10,
		help='What percentage of images to use as a test set.'
	)
	parser.add_argument(
		'--validation_percentage',
		type=int,
		default=27,
		help='What percentage of images to use as a validation set.'
	)
	parser.add_argument(
		'--pos_examples',
		type=int,
		default=100,
		help='How many positive examples per category.'
	)
	parser.add_argument(
		'--pos_to_neg_ratio',
		type=int,
		default=1.0,
		help='Ratio of positive to negative examples (per category).'
	)
	parser.add_argument(
		'--pos_inputs',
		type=int,
		default=1,
		help='Number of positive inputs to the network (except query).'
	)
	parser.add_argument(
		'--neg_inputs',
		type=int,
		default=0,
		help='Number of negative inputs to the network (except query).'
	)

	FLAGS = parser.parse_args()
	training_data, validation_data, testing_data = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage, FLAGS.pos_examples, FLAGS.pos_to_neg_ratio,
										FLAGS.pos_inputs, FLAGS.neg_inputs)
