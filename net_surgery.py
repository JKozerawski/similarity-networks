import sys
CAFFE_ROOT = '/home/jedrzej/work/caffe/python'
sys.path.insert(0, CAFFE_ROOT) 
import caffe

def perform_net_surgery(new_tuple_arm_names):
	NETWORK_NAME = "Similarity Network"
	baseNet = caffe.Net('./general_model_files/inception_v1/googlenet_deploy.prototxt', './general_model_files/inception_v1/bvlc_googlenet.caffemodel', caffe.TEST)
	newNet = caffe.Net('./general_model_files/deploy.prototxt', './general_model_files/init_iter_1.caffemodel', caffe.TEST)
	for layerName in baseNet._layer_names:
	    	print "Copying parameters for layer:",layerName
	
		try:
			W = baseNet.params[str(layerName)][0].data[...]
			b = baseNet.params[str(layerName)][1].data[...]

			for arm_name in new_tuple_arm_names:
				newNet.params[str(layerName)+"_"+arm_name][0].data[...] = W
				newNet.params[str(layerName)+"_"+arm_name][1].data[...] = b
			#print "Layer",layerName,"parameters copied"

		except:	
			try:
				W = baseNet.params[str(layerName)][0].data[...]
				for arm_name in new_tuple_arm_names:
					newNet.params[str(layerName)+"_"+arm_name][0].data[...] = W
				#print "Layer",layerName,"parameters copied"
			except:
				print "Layer",layerName,"with no weights"
	newNet.save('./general_model_files/similarityNet_copied.caffemodel')



#new_tuple_arm_names = ["query","positive_1","positive_2","positive_3"]
#perform_net_surgery(new_tuple_arm_names)
