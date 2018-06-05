import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os, threading, time,random
from tensorflow.python.client import device_lib
import data


USE_MY_MODEL = False#False



def get_use_my_model_flag():
	return USE_MY_MODEL


'''

'''
def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num



"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class EDSR(object):
	'''
	Build the EDSR model in the mult GPU mode
	'''
	def EDSR_model(self, img=None, layers=1, feature=1, scale = 2, reuse=False):
		scaling_factor = self.factor_scale
		print('scaling_factor = %f'%(scaling_factor))

		with tf.variable_scope('L1', reuse=reuse):
			#One convolution before res blocks and to convert to required feature depth
			x = slim.conv2d(img,feature,[3,3],weights_regularizer=slim.l2_regularizer(0.0005),activation_fn=None)
			#Store the output of the first convolution to add later
			conv_1 = x	

		with tf.variable_scope('Resblock', reuse=reuse):
			for i in range(layers):
				x = utils.resBlock(x=x,channels=feature,scale=scaling_factor, weights_regularizer=slim.l2_regularizer(0.0005))	

		with tf.variable_scope('Add', reuse=reuse):
			#One more convolution, and then we add the output of our first conv layer
			x = slim.conv2d(x,feature,[3,3],weights_regularizer=slim.l2_regularizer(0.0005),activation_fn=None)
			x += conv_1

		with tf.variable_scope('Upsample', reuse=reuse):
			#Upsample output of the convolution		
			x = utils.upsample(x,scale,feature,None, weights_regularizer=slim.l2_regularizer(0.0005))
			x = slim.conv2d(x,self.output_channels,[3,3],weights_regularizer=slim.l2_regularizer(0.0005),activation_fn=None)
		return x


	##############################################################
	def __init__(self,img_size=32,num_layers=32,feature_size=256,scale=2,output_channels=3, batch_size=32, lr = 0.001, factor_scale = 0.1, step_in_epo = 0 ,use_mult_gpu = False, is_test = False, use_queue = True):

		self.img_size = img_size
		self.scale = scale
		self.output_channels = output_channels
		self.mult_gpu = use_mult_gpu
		self.use_queue = use_queue
		self.lr = lr
		self.factor_scale = factor_scale
		self.step_in_epo = step_in_epo
		self.batch_size = batch_size
		self.gpu_num = check_available_gpus()

		if USE_MY_MODEL is True:
			print('USE MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF MYSELF model')


		
		if self.use_queue is not True:
			#Placeholder for image inputs
			self.input1 = x1 = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
			self.input2 = x2 = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
			#Placeholder for upscaled image ground-truth
			self.target = y = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])

			if USE_MY_MODEL is True:
				self.bicubic = bicubic = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])
				image_bicubic = bicubic - 127

		else:
			self.input_single = tf.placeholder(tf.float32, [img_size,img_size,output_channels])
			self.target_single = tf.placeholder(tf.float32, [img_size*scale,img_size*scale,output_channels])
			if USE_MY_MODEL is True:
				self.bicubic_single = tf.placeholder(tf.float32, [img_size*scale,img_size*scale,output_channels])				


			if USE_MY_MODEL is not True:					
				q = tf.FIFOQueue((step_in_epo ), [tf.float32, tf.float32], [[img_size,img_size,output_channels], [img_size*scale,img_size*scale,output_channels]])
				self.enqueue_op = q.enqueue([self.input_single, self.target_single])
			
				self.input, self.target	= q.dequeue_many(self.batch_size)
				x=self.input
				y=self.target

			else:
				#FIFOQueue
				q = tf.FIFOQueue((step_in_epo), 
								[tf.float32, tf.float32, tf.float32], 
								[[img_size,img_size,output_channels], 
								[img_size*scale,img_size*scale,output_channels], 
								[img_size*scale,img_size*scale,output_channels]])
				self.enqueue_op = q.enqueue([self.input_single, self.target_single, self.bicubic_single])
				
				self.input, self.target, self.bicubic = q.dequeue_many(self.batch_size)
				x=self.input
				y=self.target
				bicubic = self.bicubic
				image_bicubic = bicubic - 127
				
				
			
		"""
		Preprocessing as mentioned in the paper, by subtracting the mean
		However, the subtract the mean of the entire dataset they use. As of
		now, I am subtracting the mean of each batch
		"""
		mean_x = 127#tf.reduce_mean(self.input)
		image_input_0 =x1- mean_x
		image_input_1 =x2- mean_x
		mean_y = 127#tf.reduce_mean(self.target)
		image_target =y- mean_y

		print("Building EDSR in mult GPU mode...")
		gpu_num = check_available_gpus()
		names = globals()

		if gpu_num == 1:
			print('error: only one gpu in PC, Can NOT use the mult GPU mode and sys exit')
			os._exit(0)	
		
		total_loss = []	

		#the batch set has been split			
		#img_input = tf.split(image_input, int(gpu_num))		
		#img_target = tf.split(image_target, int(gpu_num))

		img_input = [None,None]
		img_input[0] = image_input_0
		img_input[1] = image_input_1
		img_target = image_target
		
		if USE_MY_MODEL is True:
			img_bicubic = tf.split(image_bicubic, int(gpu_num))


		for gpu_id in range(int(gpu_num)):
			with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
			    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
					print('device_index --> gpu%2d'%(gpu_id))							
					if USE_MY_MODEL is not True:
						names['self.output_%d'%(gpu_id)] = output = self.EDSR_model(img_input[gpu_id], num_layers, feature_size, scale,reuse = (gpu_id > 0))
						names['self.out_%d'%(gpu_id)] = tf.clip_by_value(output+mean_x,0.0,255.0)
					else:
						names['self.output_%d'%(gpu_id)] = output = img_bicubic[gpu_id] - self.EDSR_model(img_input, num_layers, feature_size, scale,reuse = (gpu_id > 0))
						#output image in tensorboard								
						names['self.out_%d'%(gpu_id)] = tf.clip_by_value(output+mean_x,0.0,255.0)
					#names['self.loss_%d'%(gpu_id)] = loss = tf.reduce_mean(tf.losses.absolute_difference(img_target[gpu_id],output))

		#Tensorflow graph setup... session, saver, etc.
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		print("Done MULT GPU model building!")





	### WITH ASYNCHRONOUS DATA LOADING ###
	def load_and_enqueue(self, coord, file_list, args, idx=0, num_thread=1):
		count = 0;
		length = len(file_list)
		shrunk_size = args.imgsize//args.scale
		try:
			while not coord.should_stop():
				i = count % length;
				gt_img = data.get_image(file_list[i],args.imgsize)
				input_img = scipy.misc.imresize(gt_img,(shrunk_size,shrunk_size),'bicubic')

				if USE_MY_MODEL is not True:
					self.sess.run(self.enqueue_op, feed_dict={self.input_single:input_img, self.target_single:gt_img})
				else:
					bicubic_img = scipy.misc.imresize(input_img,(args.imgsize,args.imgsize),'bicubic')
					self.sess.run(self.enqueue_op, feed_dict={self.input_single:input_img, self.target_single:gt_img,self.bicubic_single:bicubic_img})
					
				count+=1
		except Exception as e:
			print "stopping...", idx, e


	#######################################	
	def load_dataset_queue(self, _args):		
		train_list, test_list = data.get_global_train_set()
		threads = []

		# create threads
		num_thread = 12
		coord = tf.train.Coordinator()
		self.coord = coord

		for i in range(num_thread):
			length = len(train_list)//num_thread
			t = threading.Thread(target=self.load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length], _args, i, num_thread))
			threads.append(t)
			t.start()
		print "num thread:" , len(threads)

	"""
	Function to setup your input data pipeline
	"""
	def set_test_data_fn(self,test_set_fn=None,test_set_args=None):
		self.test_data = test_set_fn
		self.test_args = test_set_args


	"""
	Save the current state of the network to file
	"""
	def save(self,savedir='saved_models'):
		print("Saving...")
		self.saver.save(self.sess,savedir+"/model")
		print("Saved!")
		
	"""
	Resume network from previously saved weights
	"""
	def resume(self,savedir='saved_models'):
		print("Restoring...")
		self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
		print("Restored!")	

	"""
	Compute the output of this network given a specific input

	x: either one of these things:
		1. A numpy array of shape [image_width,image_height,3]
		2. A numpy array of shape [n,input_size,input_size,3]

	return: 	For the first case, we go over the entire image and run super-resolution over windows of the image
			that are of size [input_size,input_size,3]. We then stitch the output of these back together into the
			new super-resolution image and return that

	return  	For the second case, we return a numpy array of shape [n,input_size*scale,input_size*scale,3]
	"""
	def predict(self,x, y,gpu_id = 0):
		names = globals()
		'''
		if self.mult_gpu == False:
			print("ONLY one GPU Predicting...")
		else:
			print("Mult GPU Predicting...")

		print('W = %d ,H=%d'%(x.shape[0],x.shape[1]))
		'''

		if USE_MY_MODEL is True:
			bic = scipy.misc.imresize(x,(x.shape[0]*self.scale,x.shape[1]*self.scale),'bicubic')

		if (len(x.shape) == 3) and not(x.shape[0] == self.img_size and x.shape[1] == self.img_size):
			num_across = x.shape[0]//self.img_size
			num_down = x.shape[1]//self.img_size
			tmp_image1 = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,3])
			tmp_image2 = np.zeros([y.shape[0]*self.scale,y.shape[1]*self.scale,3])
			
			for i in range(num_across):
				for j in range(num_down):				
					if self.mult_gpu == False:
						tmp = self.sess.run(self.out,feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})[0]
					else:
						if USE_MY_MODEL is True:
							bicubic = bic[i*self.img_size*self.scale:(i+1)*self.img_size*self.scale,j*self.img_size*self.scale:(j+1)*self.img_size*self.scale]	
							tmp = self.sess.run(names['self.out_%d'%(gpu_id)],feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]], self.bicubic:[bicubic]})[0]
						else:
							tmp1,tmp2 = self.sess.run([names['self.out_0'],names['self.out_1']],feed_dict={self.input1:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]],self.input2:[y[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})

					tmp1 = tmp1[0]
					tmp2 = tmp2[0]
					tmp_image1[i*tmp1.shape[0]:(i+1)*tmp1.shape[0],j*tmp1.shape[1]:(j+1)*tmp1.shape[1]] = tmp1		
					tmp_image2[i*tmp2.shape[0]:(i+1)*tmp2.shape[0],j*tmp2.shape[1]:(j+1)*tmp2.shape[1]] = tmp2
			
			#this added section fixes bottom right corner when testing
			if (x.shape[0]%self.img_size != 0 and  x.shape[1]%self.img_size != 0):
				
				if self.mult_gpu == False:
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]]})[0]
				else:
					if USE_MY_MODEL is True:
						bicubic = bic[-1*self.img_size *self.scale :,-1*self.img_size * self.scale:]
						tmp = self.sess.run(names['self.out_%d'%(gpu_id)],feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]], self.bicubic:[bicubic]})[0]
					else:
						tmp1,tmp2 = self.sess.run([names['self.out_0'],names['self.out_1']],feed_dict={self.input1:[x[-1*self.img_size:,-1*self.img_size:]],self.input2:[y[-1*self.img_size:,-1*self.img_size:]]})
				tmp1 = tmp1[0]
				tmp2 = tmp2[0]
				tmp_image1[-1*tmp1.shape[0]:,-1*tmp1.shape[1]:] = tmp1
				tmp_image2[-1*tmp2.shape[0]:,-1*tmp2.shape[1]:] = tmp1
				
			#
			if x.shape[0]%self.img_size != 0:
				for j in range(num_down):
					if self.mult_gpu == False:
						tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})[0]
					else:
						if USE_MY_MODEL is True:
							bicubic = bic[-1*self.img_size * self.scale:,j*self.img_size * self.scale:(j+1)*self.img_size * self.scale]
							tmp = self.sess.run(names['self.out_%d'%(gpu_id)],feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]], self.bicubic:[bicubic]})[0]
						else:
							tmp1,tmp2 = self.sess.run([names['self.out_0'],names['self.out_1']],feed_dict={self.input1:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]],self.input2:[y[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})
					tmp1 = tmp1[0]
					tmp2 = tmp2[0]
					tmp_image1[-1*tmp1.shape[0]:,j*tmp1.shape[1]:(j+1)*tmp1.shape[1]] = tmp1
					tmp_image2[-1*tmp2.shape[0]:,j*tmp2.shape[1]:(j+1)*tmp2.shape[1]] = tmp2



			#
			if x.shape[1]%self.img_size != 0:
				for j in range(num_across):
					if self.mult_gpu == False:
						tmp = self.sess.run(self.out,feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})[0]
					else:
						if USE_MY_MODEL is True:
							bicubic = bic[j*self.img_size * self.scale:(j+1)*self.img_size * self.scale,-1*self.img_size * self.scale:]
							tmp = self.sess.run(names['self.out_%d'%(gpu_id)],feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]], self.bicubic:[bicubic]})[0]
						else:
							tmp1,tmp2 = self.sess.run([names['self.out_0'],names['self.out_1']],feed_dict={self.input1:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]],self.input2:[y[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})
					tmp1 = tmp1[0]
					tmp2 = tmp2[0]
					tmp_image1[j*tmp1.shape[0]:(j+1)*tmp1.shape[0],-1*tmp1.shape[1]:] = tmp1
					tmp_image2[j*tmp2.shape[0]:(j+1)*tmp2.shape[0],-1*tmp2.shape[1]:] = tmp2
					
			return tmp_image1,tmp_image2
		else:
			tmp_image1 = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,3])
			tmp_image2 = np.zeros([y.shape[0]*self.scale,y.shape[1]*self.scale,3])

			if USE_MY_MODEL is not True:
				tmp_image1,tmp_image2 = self.sess.run([names['self.out_0'],names['self.out_1']],feed_dict={self.input1:[x],self.input2:[y]})
			else:
				tmp_image = self.sess.run(names['self.out_%d'%(gpu_id)],feed_dict={self.input:[x], self.bicubic:[bic]})[0]

			tmp_image1 = scipy.misc.imresize(tmp_image1[0],(x.shape[0]*self.scale,x.shape[1]*self.scale,3))
			tmp_image2 = scipy.misc.imresize(tmp_image2[0],(y.shape[0]*self.scale,y.shape[1]*self.scale,3))
			return tmp_image1,tmp_image2

			 

	'''
	'''
	def movie_predict(self,x_):
		names = globals()
		result_list = []

		if self.gpu_num > 1:
			print('mult GPU to SR the movie')
			if len(x_) <2 :
				print('the input is not movie')
				os._exit(0)			
		else:
			print('only one Gpu to work')

		for i in range(0,int(len(x_)),self.gpu_num):
			tmp1,tmp2 = self.predict(x_[i],x_[i+1])
			result_list.append(tmp1)
			result_list.append(tmp2)
			print(i,tmp1.shape,tmp2.shape)
		print(len(result_list))
		return result_list



						



	"""
	Function to setup your input data pipeline
	"""
	def set_data_fn(self,fn,args,test_set_fn=None,test_set_args=None):
		self.data = fn
		self.args = args
		self.test_data = test_set_fn
		self.test_args = test_set_args

	"""
	Train the neural network
	"""
	def train(self,iterations=1000,save_dir="saved_models",pre_train_model ='',  step_in_epoch = 0):
		if 	step_in_epoch == 0:
			print('step_in_epoch in the train() has problem...and sys is break')
			os._exit(0)

		#Removing previous save directory if there is one
		if pre_train_model == '':		
			if os.path.exists(save_dir):
				shutil.rmtree(save_dir)
			#Make new save directory
			os.mkdir(save_dir)
			print('the model is new!')
		else:
			if os.path.exists(save_dir+'/test'):
				shutil.rmtree(save_dir+'/test')
				shutil.rmtree(save_dir+'/train')
				print('old scalars has been delted and restore model is going on')
		
		#the lr is change
		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = self.lr
		print('the strater learning rate is %f'%(starter_learning_rate))
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,(self.step_in_epo * 1), 0.96, staircase=True)#
		#tf.summary.scalar("learning rate", learning_rate)
		#tf.summary.scalar("global_step", global_step)

		#Using adam optimizer as mentioned in the paper
		optimizer = tf.train.AdamOptimizer(learning_rate)
		if True:			
			#This is the train operation for our objective
			train_op = optimizer.minimize(self.loss, global_step = global_step, colocate_gradients_with_ops = True)	
		else:
			#use the grads clip
			grads = optimizer.compute_gradients(self.loss, colocate_gradients_with_ops = True)
			for i, (g,v) in enumerate(grads):
				if g is not None:
					grads[i] = (tf.clip_by_norm(g,5),v)
			train_op = optimizer.apply_gradients(grads, global_step=global_step)		

		#the mult_gpu info
		if self.mult_gpu == False:
			print('the optimizer ONLY one GPU mode')
		else:
			print('the optimizer MULT GPU mode')	
	
		#Just a tf thing, to merge all summaries into one
		merged = tf.summary.merge_all()
		
		#Operation to initialize all variables
		init = tf.global_variables_initializer()
		print("Begin training...")

		loss_max = 1e8
		loss_init = 0
		lr_before = starter_learning_rate

		if step_in_epoch > 1000:
			every_batch_test = step_in_epoch/10
		elif step_in_epoch > 100:
			every_batch_test = min(20,step_in_epoch)
		else:
			every_batch_test = step_in_epoch /30 

		error_scale = 20.0

		with self.sess as sess:
			#Initialize all variables
			sess.run(init)
			test_exists = self.test_data
			#create summary writer for train
			train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)

			#if we want to refine the EDSR model, we restore the pre-train model
			if pre_train_model != '':
				print('get pre-train model ...\r\n')
				self.resume(pre_train_model)

			#If we're using a test set, include another summary writer for that
			if test_exists:
				test_writer = tf.summary.FileWriter(save_dir+"/test",sess.graph)
				if USE_MY_MODEL is not True:
					test_x,test_y = self.test_data(*self.test_args)
					test_feed = {self.input:test_x,self.target:test_y}
				else:
					test_x,test_y,test_bic = self.test_data(*self.test_args)
					test_feed = {self.input:test_x,self.target:test_y,self.bicubic:test_bic}

			#This is our training loop, the loop is the batch data.NOT epoch
			print('total step is %d, and each epoch has %d step , epoch is %d'%(iterations*step_in_epoch, step_in_epoch, iterations))
			for i in tqdm(range(iterations*step_in_epoch)):

				if self.mult_gpu is not True:
					#Use the data function we were passed to get a batch every iteration
					x,y = self.data(*self.args)
					#Create feed dictionary for the batch
					feed = {
						self.input:x,
						self.target:y
					}
					#Run the train op and calculate the train summary
					if (i+1) % (step_in_epoch/every_batch_test) == 0:  #save the time
						summary,_,lr_now,_ = sess.run([merged,train_op, learning_rate, global_step],feed)
					else:
						_,lr_now,_ = sess.run([train_op, learning_rate, global_step],feed)
				else:	
					if (i+1) % (step_in_epoch/every_batch_test) == 0:  #save the time				
						summary,_,lr_now,_ = sess.run([merged,train_op, learning_rate, global_step])
					else:
						_,lr_now,_ = sess.run([train_op, learning_rate, global_step])

				if lr_now != lr_before:
					lr_before = lr_now
					print('learning rate is change and the value is %.10f'%(lr_now))					

				#If we're testing, don't train on test set. But do calculate summary				 
				save_flag = False
				if test_exists and (i+1)%(step_in_epoch/every_batch_test) == 0:
					t_summary,loss_now = sess.run([merged,self.loss],test_feed)
					#Write test summary
					test_writer.add_summary(t_summary,i)
					
					if loss_now < loss_max:
						if (i+1-(step_in_epoch/every_batch_test)) == 0:
							loss_init = loss_now
							print('loss_init = %f'%(loss_init))
						loss_max = loss_now
						print('the loss is lower and the value is %f'%(loss_now))
						save_flag = True						
					elif loss_now > loss_init * error_scale or loss_now > 3.0 * loss_max:
						print('sys is stop\r\nthe loss is boom... and the value is %f'%(loss_now))
						os.system('python train.py --mult_gpu --load_model saved_models --lr %f'%(starter_learning_rate/2.0))		
						os._exit(0)
					else:
						save_flag = False

				#Write train summary for this step
				if (i+1) % (step_in_epoch/every_batch_test) == 0:  #save the memory
					train_writer.add_summary(summary,i)

				#Save our trained model
				#if (i+1) % step_in_epoch == 0 or (i+1) % (step_in_epoch/5) == 0 :		
				if save_flag == True:
					self.save()	

			#close the queue
			self.coord.request_stop()	