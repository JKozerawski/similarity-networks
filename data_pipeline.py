import os.path
from os import walk
import random
import re
import numpy as np
from glob import glob
import argparse
import pickle
import caffe
from scipy.spatial.distance import cdist

from caffe_feature_extractor import CaffeFeatureExtractor

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
FLAGS = []

def get_all_features(image_dir, sub_dirs):
	# Initialize Extraction Net (for pre-processing):
	extractionNet = CaffeFeatureExtractor(
		model_path = "/media/jedrzej/Seagate/Python/siamese-network/model/googlenet_deploy.prototxt",
		pretrained_path = "/media/jedrzej/Seagate/Python/siamese-network/model/bvlc_googlenet.caffemodel",
		blob = "pool5/7x7_s1",
		crop_size = 224,
		mean_values = [104.0, 117.0, 123.0]
		)
	all_info = []
	for sub_dir_i, sub_dir in enumerate(sub_dirs):
		print sub_dir_i
		allVal = []
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']	# allowed image extensions
		file_list = get_images(image_dir, sub_dir, extensions)

		n = len(file_list)
		for i in xrange(n):
			feat = np.asarray(extractionNet.extract_feature(caffe.io.load_image(file_list[i])).reshape(1024))
			allVal.append([file_list[i], feat.copy()])
		all_info.append(allVal)
	pickle.dump( all_info, open( "./train_features_70.p", "wb" ) )
	print "File saved"
	return all_info

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

def find_hard_query(all_info, curr_category_number, positive_indices, positive = True):
	positive_features = np.asarray([all_info[curr_category_number][i][1] for i in positive_indices])
	positive_names = [all_info[curr_category_number][i][0] for i in positive_indices]
	query_name = ""
	if(positive):
		names = []
		distances = []
		for i in xrange(len(all_info[curr_category_number])):
			if i not in positive_features:
				feat = np.asarray(all_info[curr_category_number][i][1]).reshape(1,1024)
				Y = np.mean(cdist(positive_features,feat,'euclidean'))
				distances.append(Y)
				names.append(all_info[curr_category_number][i][0])

		distances = np.asarray(distances)
		best_indices = np.argsort(distances)[::-1][:5]
		chosen_index = np.random.choice(best_indices, 1, replace=False)[0]
		query_name = names[chosen_index]
	else:
		names = []
		distances = []
		for j in xrange(len(all_info)):
			if(j != curr_category_number):
				for i in xrange(len(all_info[j])):
					feat = np.asarray(all_info[j][i][1]).reshape(1,1024)
					Y = np.mean(cdist(positive_features,feat,'euclidean'))
					distances.append(Y)
					names.append(all_info[j][i][0])
		distances = np.asarray(distances)
		best_indices = np.argsort(distances)[:5]
		chosen_index = np.random.choice(best_indices, 1, replace=False)[0]
		query_name = names[chosen_index]
	positive_names.append(query_name)
	return [ "/".join(name.split("/")[-2:]) for name in positive_names]

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



def create_image_lists(image_dir, testing_percentage, validation_percentage, positive_examples_per_category, positives_to_negatives_ratio, no_of_positive_inputs, no_of_negative_inputs, hierarchy_info, categories_file):
	"""Builds a list of training images from the file system.


	Args:
	describe args

	Returns:
	describe output
	"""
	
	negative_examples_per_category = int(positive_examples_per_category/positives_to_negatives_ratio)		# how many image pairs per category

	# Check if the directory exists:
	if not os.path.isdir(image_dir):
		print "Image directory " + image_dir + " not found."
		return None

	results = []	# empty list for the image tuples
	sub_dirs = [x[0] for x in walk(image_dir)]
	# The root directory comes first, so skip it.
	sub_dirs = sub_dirs[1:]
	sub_dirs.sort()	# sort the directories

	if(categories_file!=None):
		categories_to_use = pickle.load( open( categories_file, "rb" ) )	# use part of categories
		sub_dirs = [sub_dirs[i] for i in categories_to_use]

	
	# Enumerate through all the categories:
	for sub_dir_i, sub_dir in enumerate(sub_dirs):
		print sub_dir_i
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
	print len(results)
	# Split the tuples into training, validation and testing:
	test_index = int((testing_percentage/100.)*len(results))
	val_index = test_index+int((validation_percentage/100.)*len(results))
	testing_data = results[:test_index]
	validation_data = results[test_index:val_index]
	training_data = results[val_index:]
	#print training_data, validation_data, testing_data
	return training_data, validation_data, testing_data


def save_list(pathList, posNumber=3, negNumber=0, prefix = "train"):
		#print pathList
		n = len(pathList[0])-1
		txtFiles = []
		for i in xrange(n):
			txtFiles.append("")

		for line_i, line in enumerate(pathList):
			for i in xrange(n):
				txtFiles[i]+=line[i]+" "+line[n]+"\n"	# image path + label

		# Save all files:
		for i in xrange(n):
			if (i==(n-1)):
				f = open('./txtfiles/'+prefix+'_query.txt','w')
			else:
				f = open('./txtfiles/'+prefix+'_'+str(i+1)+'.txt','w')
			f.write(txtFiles[i])
			f.close()


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
		default=0,
		help='What percentage of images to use as a test set.'
	)
	parser.add_argument(
		'--validation_percentage',
		type=int,
		default=30,
		help='What percentage of images to use as a validation set.'
	)
	parser.add_argument(
		'--pos_examples',
		type=int,
		default=30,
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
		default=2,
		help='Number of positive inputs to the network (except query).'
	)
	parser.add_argument(
		'--neg_inputs',
		type=int,
		default=0,
		help='Number of negative inputs to the network (except query).'
	)
	parser.add_argument(
		'--hierarchy',
		type=bool,
		default=False,
		help='Add ILSVRC 2012 hierarchy information to training.'
	)
	parser.add_argument(
		'--categories',
		type=str,
		default=None,
		help='Path to pickle file with categories to process.'
	)
	FLAGS = parser.parse_args()
	#training_data, validation_data, testing_data = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage, FLAGS.pos_examples, FLAGS.pos_to_neg_ratio, FLAGS.pos_inputs, FLAGS.neg_inputs, FLAGS.hierarchy, FLAGS.categories)
	training_data, validation_data, testing_data = create_image_lists("/media/jedrzej/Seagate/DATA/102flowers/images/", 0, 30, 300, 1.0, 8, 0, False, "/media/jedrzej/Seagate/Python/similarity-networks/train_categories_70.p")
	save_list(training_data, posNumber=8, prefix ="train")
	save_list(validation_data, posNumber=8, prefix ="val")
