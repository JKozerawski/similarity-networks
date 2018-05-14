import re
DATA_PROTOTXT_FILE_PATH = "./general_model_files/data_input.prototxt"
LOSS_PART_1_PROTOTXT_FILE_PATH = "./general_model_files/loss_part1.prototxt"
LOSS_PART_2_PROTOTXT_FILE_PATH = "./general_model_files/loss_part2.prototxt"
LOSS_PART_2_DEPLOY_PROTOTXT_FILE_PATH = "./general_model_files/loss_part2_deploy.prototxt"
BASE_PROTOTXT_FILE_PATH = "./general_model_files/inception_v1/googlenet_base.prototxt"
NEW_PROTOTXT_FILE_PATH = "./general_model_files/train_val.prototxt"
NEW_DEPLOY_PROTOTXT_FILE_PATH = "./general_model_files/deploy.prototxt"


name_list = ["bottom","top","name"]
loss_layers = ["loss1/classifier", "loss2/classifier","pool5/7x7_s1"]
loss_weights = ["0.1","0.1","1.0"]


#############################################
def create_similarity_network_train_val(new_tuple_arm_names,which_arms_have_shared_weights,PATH_TO_DATA_DIR,TRAIN_BATCH_SIZE,VAL_BATCH_SIZE):
	txt = ""
	# Add data reading part:
	txt+= "name: \"Similarity Network\"\n"
	for i in xrange(len(new_tuple_arm_names)):
		with open(DATA_PROTOTXT_FILE_PATH) as f:
	    		lines = f.readlines()
		for line in lines:
			if(i == 0): line = line.replace("_nameF_special", "")
			else: line = line.replace("nameF_special", new_tuple_arm_names[i])
			line = line.replace("nameF", new_tuple_arm_names[i])
			line = line.replace("namePath", PATH_TO_DATA_DIR)
			line = line.replace("train_batch_size",TRAIN_BATCH_SIZE)
			line = line.replace("val_batch_size",VAL_BATCH_SIZE)
			if(i == 0 ): app1 = "query"
			else: app1 = str(i)
			line = line.replace("name1", app1)
			txt+=line
	txt+= "layer {\n"
	txt+= "  name: \"silence_layer\"\n"
	txt+= "  type: \"Silence\"\n"
	for i in xrange(len(new_tuple_arm_names)-1):
		txt+= "  bottom: \"label_"+new_tuple_arm_names[i+1]+"\"\n"
	txt+= "}\n"

	curr_loss = 0
	# Create and add main body of network:
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
				
			elif("use_global_stats:" in line):
				txt+= line.replace("true", "false")		# change batchNorm layers
			else:
				txt+=line
			if ("layer" in line):
				counter = 0
				name = ""
			if("name" in line):
				name = str(re.findall('"([^"]*)"', line)[0])
				#print name
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
					elif(counter==3):
						txt+= "    name: \""+name+"_c\"\n"


			if(arm_i == len(new_tuple_arm_names)-1):
				if("here" in line): 
					txt+="#" * 50+"\n"
					txt+="#### loss function below ####\n"
					txt+="#" * 50+"\n"
	
					# Add loss function part:
					# TO DO: Create loss function in here:
					for i in xrange(len(new_tuple_arm_names)):
						for j in xrange(len(new_tuple_arm_names)):
							if j>i:
								if(i == 0 ): app1 = "q"
								else: app1 = str(i)
								app2 = str(j)
				 				
								with open(LOSS_PART_1_PROTOTXT_FILE_PATH) as f:
				    					lines = f.readlines()
								for line in lines:
									line = line.replace("lossname", loss_layers[curr_loss])
									line = line.replace("nameF", str(app1)+str(app2))
									line = line.replace("number", str(curr_loss+1))
									line = line.replace("name1", new_tuple_arm_names[i])
									line = line.replace("name2", new_tuple_arm_names[j])
									txt += line
					with open(LOSS_PART_2_PROTOTXT_FILE_PATH) as f:
    						lines = f.readlines()
						for line in lines:
							line = line.replace("loss_no", loss_weights[curr_loss])
							line = line.replace("number", str(curr_loss+1))
							if ("concatenating_prev_layers" in line):
								for i in xrange(len(new_tuple_arm_names)):
									for j in xrange(len(new_tuple_arm_names)):
										if j>i:
											if(i == 0 ): app1 = "q"
											else: app1 = str(i)
											app2 = str(j)
											txt += "  bottom: \"dist_"+str(app1)+str(app2)+"_"+str(curr_loss+1)+"\""+"\n"
							else: txt+=line
					curr_loss += 1
		
	#print txt
	f = open(NEW_PROTOTXT_FILE_PATH,'w')
	f.write(txt)
	f.close()

#############################################
def create_similarity_network_deploy(new_tuple_arm_names,which_arms_have_shared_weights):
	txt = ""
	# Add data reading part:
	txt+= "name: \"Similarity Network\"\n"

	txt+= "layer {\n"
  	txt+= "  name: \"data\"\n"
  	txt+= "  type: \"Input\"\n"
  	txt+= "  top: \"data\"\n"
  	txt+= "  input_param { shape: { dim: 1 dim: "+str(3*len(new_tuple_arm_names))+" dim: 224 dim: 224 } }\n"
	txt+= "}\n"

	txt+= "layer {\n"
  	txt+= "  name: \"slicer_data\"\n"
  	txt+= "  type: \"Slice\"\n"
  	txt+= "  bottom: \"data\"\n"
	for i in xrange(len(new_tuple_arm_names)):
  		txt+= "  top: \"data_"+new_tuple_arm_names[i]+"\"\n"
  	txt+= "  slice_param {\n"
    	txt+= "    axis: 1\n"
	for i in xrange(len(new_tuple_arm_names)-1):
    		txt+= "    slice_point: "+str(3*(i+1))+"\n"
  	txt+= "  }\n"
	txt+= "}\n"

	# Create and add main body of network:
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
			# TO DO: WORK HERE ON DELETING UNNECESSARY LAYERS (from loss1 and loss2)
			if(any(substring in line for substring in name_list)):
				txt+=line.rsplit("\"", 1)[0]+"_"+new_tuple_arm_name+"\"\n"
				
			elif("use_global_stats:" in line):
				txt+= line.replace("true", "false")		# change batchNorm layers
			else:
				txt+=line
			if ("layer" in line):
				counter = 0
				name = ""
			if("name" in line):
				name = str(re.findall('"([^"]*)"', line)[0])
				#print name
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
					elif(counter==3):
						txt+= "    name: \""+name+"_c\"\n"


	# Adding final loss function:
	txt+="#" * 50+"\n"
	txt+="#### loss function below ####\n"
	txt+="#" * 50+"\n"
	for i in xrange(len(new_tuple_arm_names)):
		for j in xrange(len(new_tuple_arm_names)):
			if j>i:
				if(i == 0 ): app1 = "q"
				else: app1 = str(i)
				app2 = str(j)
 				
				with open(LOSS_PART_1_PROTOTXT_FILE_PATH) as f:
    					lines = f.readlines()
				for line in lines:
					line = line.replace("lossname", loss_layers[2])
					line = line.replace("nameF", str(app1)+str(app2))
					line = line.replace("number", "3")
					line = line.replace("name1", new_tuple_arm_names[i])
					line = line.replace("name2", new_tuple_arm_names[j])
					txt += line
	with open(LOSS_PART_2_DEPLOY_PROTOTXT_FILE_PATH) as f:
		lines = f.readlines()
		for line in lines:
			line = line.replace("number", "3")
			if ("concatenating_prev_layers" in line):
				for i in xrange(len(new_tuple_arm_names)):
					for j in xrange(len(new_tuple_arm_names)):
						if j>i:
							if(i == 0 ): app1 = "q"
							else: app1 = str(i)
							app2 = str(j)
							txt += "  bottom: \"dist_"+str(app1)+str(app2)+"_3\""+"\n"
			else: txt+=line
		
	#print txt
	f = open(NEW_DEPLOY_PROTOTXT_FILE_PATH,'w')
	f.write(txt)
	f.close()


#if __name__ == '__main__':		
	#create_tuple_network_arms()
	#create_similarity_network_deploy()
