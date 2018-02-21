import os.path
import random
import re
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def get_images(image_dir, sub_dir, extensions):
	file_list = []
	dir_name = os.path.basename(sub_dir)
   	
	tf.logging.info("Looking for images in '" + dir_name + "'")		# look into a subfolder
	for extension in extensions:
		file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
		file_list.extend(gfile.Glob(file_glob))
	if not file_list:
		tf.logging.warning('No files found')
		return
	if len(file_list) < 20:
		tf.logging.warning(
	  	'WARNING: Folder has less than 20 images, which may cause issues.')
	elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
		tf.logging.warning(
	  	'WARNING: Folder {} has more than {} images. Some images will '
	  	'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
	return file_list

def get_random_image_tuples(sim_label, list_pos, list_neg, no_of_pos = 1, no_of_neg = 0):
	neg_images = []
	if(sim_label == 1):
		pos_images = np.random.choice(list_pos, no_of_pos+1, replace=False)
		for i in xrange(no_of_neg):
			neg_images.append(np.random.choice(list_neg[i]))
	
	else:
		pos_images = np.random.choice(list_pos, no_of_pos ,replace=False)
		for i in xrange(no_of_neg+1):
			neg_images.append(np.random.choice(list_neg[i]))

	
	images = [str(i) for i in pos_images]+[str(i) for i in neg_images]
	#print os.path.basename(images[0]), images[0].split("/")[-2:]
	return [ "/".join(img.split("/")[-2:]) for img in images]

def create_image_lists(image_dir, testing_percentage=30, validation_percentage=21, positive_examples_per_category=100, positives_to_negatives_ratio = 1., no_of_positive_inputs = 1, no_of_negative_inputs = 0):
	"""Builds a list of training images from the file system.


	Args:
	describe args

	Returns:
	describe output
	"""
	negative_examples_per_category = int(positive_examples_per_category/positives_to_negatives_ratio)		# how many image pairs per category

	if not gfile.Exists(image_dir):
		tf.logging.error("Image directory '" + image_dir + "' not found.")
		return None
	result = []
	sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

	# The root directory comes first, so skip it.
	sub_dirs = sub_dirs[1:]

	for sub_dir_i, sub_dir in enumerate(sub_dirs):				# go through the subdirectories (categories)
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']		# allowed image extensions
		file_list = get_images(image_dir, sub_dir, extensions)
		dir_name = os.path.basename(sub_dir)
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())		# what is the label name

		# get "noOfPositiveExamples" of pairs with label "1":
	    	for k in xrange(positive_examples_per_category):
			neg_indices = np.random.choice([x for x in xrange(len(sub_dirs)) if x != sub_dir_i], no_of_negative_inputs, replace=False)
			image_tuples = get_random_image_tuples(1, file_list,  [get_images(image_dir, sub_dirs[j], extensions) for j in neg_indices], no_of_pos = no_of_positive_inputs, no_of_neg = no_of_negative_inputs)
			result.append(image_tuples+["1"])

		for k in xrange(negative_examples_per_category):
			neg_indices = np.random.choice([x for x in xrange(len(sub_dirs)) if x != sub_dir_i], no_of_negative_inputs+1, replace=False)
			image_tuples = get_random_image_tuples(0, file_list,  [get_images(image_dir, sub_dirs[j], extensions) for j in neg_indices], no_of_pos = no_of_positive_inputs, no_of_neg = no_of_negative_inputs)
			result.append(image_tuples+["0"])

	   	print len(result)

	random.shuffle(result)
	test_index = int(testing_percentage/100*len(result))
	val_index = test_index+int(validation_percentage/100*len(result))

	testing_data = results[:test_index]
	validation_data = results[test_index:val_index]
	training_data = results[val_index:]
	    
	return training_data, validation_data, testing_data


if __name__ == '__main__':
	training_data, validation_data, testing_data = create_image_lists("/media/jedrzej/Seagate/DATA/ILSVRC2012/TRAIN/",positive_examples_per_category=2)
