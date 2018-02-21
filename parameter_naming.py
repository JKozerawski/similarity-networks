import re

BASE_PROTOTXT_FILE_PATH = "./model/mobilenet_siamese_train_val_base.prototxt"
NEW_PROTOTXT_FILE_PATH = "./model/mobilenet_siamese_train_val_updated.prototxt"

#############################################
def create_tuple_network_arms():

	new_tuple_arm_names = ["left","right"]
	which_arms_have_shared_weights = [True, True]
	txt = ""
	name_list = ["bottom","top","name"]

	for arm_i, new_tuple_arm_name in enumerate(new_tuple_arm_names):
		# creating an arm
		with open(BASE_PROTOTXT_FILE_PATH) as f:
		    	lines = f.readlines()
		
		counter = 0
		name = ""
		txt+="#" * 50+"\n"
		txt+="#### "+new_tuple_arm_name+" arm ####\n"
		txt+="#" * 50+"\n"
		for line in lines:
			if(any(substring in line for substring in name_list)):
				txt+=line.rsplit("\"", 1)[0]+"_"+new_tuple_arm_name+"\"\n"
				
			else:
				txt+=line
			if ("layer" in line):
				counter = 0
				name = ""
			if("name" in line):
				name = str(re.findall('"([^"]*)"', line)[0])
				print name
				#name+="_right"
				#if("right" in name): name = name.replace("right", "")
				#else: name = name.replace("left", "")
			if("  param {" in line):
				counter +=1
				if(which_arms_have_shared_weights[arm_i]):	# check if this arm should have shared weights
					if(counter==1):
						txt+= "    name: \""+name+"_w\"\n"
					elif(counter==2):
						txt+= "    name: \""+name+"_b\"\n"
	txt+="#" * 50+"\n"
	txt+="#### loss function below ####\n"
	txt+="#" * 50+"\n"
	#print txt
	f = open(NEW_PROTOTXT_FILE_PATH,'w')
	f.write(txt)
	f.close()


		
create_tuple_network_arms()
