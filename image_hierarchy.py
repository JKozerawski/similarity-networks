from glob import glob
from nltk.corpus import wordnet as wn
import numpy as np
import pickle

IMAGENET_DIR = "/media/jedrzej/Seagate/DATA/ILSVRC2012/TRAIN/"

folders_list = glob(IMAGENET_DIR+"*")
categories_list = []
all_parents = []
for category in folders_list:
	category_no = category.split("/")[-1][1:]
	categories_list.append(category_no)

	s = wn._synset_from_pos_and_offset('n',int(category_no))
	parents = s.hypernyms()
	unique_parents = []
	while(parents):
		if( parents[0] not in unique_parents): unique_parents.append(parents[0])
		if( parents[0] not in all_parents): all_parents.append(parents[0])
		parents.extend(parents[0].hypernyms())
		parents.remove(parents[0])
	
print len(all_parents)
##########################################################
frequency = np.zeros(len(all_parents))
for category in folders_list:
	category_no = category.split("/")[-1][1:]
	categories_list.append(category_no)

	s = wn._synset_from_pos_and_offset('n',int(category_no))

	parents = s.hypernyms()
	unique_parents = []
	while(parents):
		if( parents[0] not in unique_parents): unique_parents.append(parents[0])
		parents.extend(parents[0].hypernyms())
		parents.remove(parents[0])

	for u_p in unique_parents:
		idx = all_parents.index(u_p)
		frequency[idx] +=1
f1 = np.where(frequency >1)[0]
f2 = np.where(frequency <1000)[0]
f_tot = np.intersect1d(f1,f2)
#print np.max(frequency)
new_parents = [all_parents[i] for i in f_tot]
print len(new_parents)

##########################################################
positive_wordnets = dict()
for p in new_parents:
	positive_wordnets[str(p.lemmas()[0].name())] = []
for category in folders_list:
	category_no = category.split("/")[-1][1:]
	categories_list.append(category_no)

	s = wn._synset_from_pos_and_offset('n',int(category_no))
	parents = s.hypernyms()
	unique_parents = []
	while(parents):
		if( parents[0] not in unique_parents): unique_parents.append(parents[0])
		parents.extend(parents[0].hypernyms())
		parents.remove(parents[0])
	for u_p in unique_parents:
		if(u_p in new_parents):
			positive_wordnets[str(u_p.lemmas()[0].name())].append('n'+category_no)
			#positive_wordnets[str(u_p.lemmas()[0].name())].append(str(s.lemmas()[0].name()))

print positive_wordnets
pickle.dump(positive_wordnets, open( "./hierarchy.p", "wb"))


#print categories_list



