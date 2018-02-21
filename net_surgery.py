import sys
CAFFE_ROOT = '/home/jedrzej/work/caffe/python'
sys.path.insert(0, CAFFE_ROOT) 
import caffe

NETWORK_NAME = "siamese_mobilenet"

baseNet = caffe.Net('./model/mobilenet_deploy.prototxt', './model/mobilenet.caffemodel', caffe.TEST)
newNet = caffe.Net('./model/deploy.prototxt', './model/'+NETWORK_NAME+'_initialized.caffemodel', caffe.TEST)

new_tuple_arm_names = ["left","right"]

for layerName in baseNet._layer_names:
    	#print "Copying parameters for layer:",layerName
	try:
		W = baseNet.params[str(layerName)][0].data[...]
		b = baseNet.params[str(layerName)][1].data[...]

		for arm_name in new_tuple_arm_names:
			newNet.params[str(layerName)+"_"+arm_name][0].data[...] = W
			newNet.params[str(layerName)+"_"+arm_name][1].data[...] = b
		print "Layer",layerName,"parameters copied"
		
	except:	
		try:
			W = baseNet.params[str(layerName)][0].data[...]
			for arm_name in new_tuple_arm_names:
				newNet.params[str(layerName)+"_"+arm_name][0].data[...] = W
			
		except:
			print "Layer",layerName,"with no weights"
newNet.save('./model/'+NETWORK_NAME+'_copied.caffemodel')

