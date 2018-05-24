import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os
from tensorflow.python.client import device_lib


'''

'''
def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num


'''
Build the EDSR model in the mult GPU mode
'''
def EDSR_model(img=None, layers=1, feature=1, scale = 2, reuse=False):
	scaling_factor = 0.1
	
	with tf.variable_scope('L1', reuse=reuse):
		#One convolution before res blocks and to convert to required feature depth
		x = slim.conv2d(img,feature,[3,3])
		#Store the output of the first convolution to add later
		conv_1 = x	

	with tf.variable_scope('Resblock', reuse=reuse):
		for i in range(layers):
			x = utils.resBlock(x=x,channels=feature,scale=scaling_factor)	

	with tf.variable_scope('Add', reuse=reuse):
		#One more convolution, and then we add the output of our first conv layer
		x = slim.conv2d(x,feature,[3,3])
		x += conv_1

	with tf.variable_scope('Upsample', reuse=reuse):
		#Upsample output of the convolution		
		x = utils.upsample(x,scale,feature,None)

	return x

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class EDSR(object):

	def __init__(self,img_size=32,num_layers=32,feature_size=256,scale=2,output_channels=3, use_mult_gpu = False):
		self.img_size = img_size
		self.scale = scale
		self.output_channels = output_channels
		self.mult_gpu = use_mult_gpu

		#Placeholder for image inputs
		self.input = x = tf.placeholder(tf.float32,[None,img_size,img_size,output_channels])
		#Placeholder for upscaled image ground-truth
		self.target = y = tf.placeholder(tf.float32,[None,img_size*scale,img_size*scale,output_channels])
	
		"""
		Preprocessing as mentioned in the paper, by subtracting the mean
		However, the subtract the mean of the entire dataset they use. As of
		now, I am subtracting the mean of each batch
		"""
		mean_x = 127#tf.reduce_mean(self.input)
		image_input =x- mean_x
		mean_y = 127#tf.reduce_mean(self.target)
		image_target =y- mean_y

		if self.mult_gpu == False:
			print("Building EDSR in one GPU mode...")			

			#One convolution before res blocks and to convert to required feature depth
			x = slim.conv2d(image_input,feature_size,[3,3])
	
			#Store the output of the first convolution to add later
			conv_1 = x	

			"""
			This creates `num_layers` number of resBlocks
			a resBlock is defined in the paper as
			(excuse the ugly ASCII graph)
			x
			|\
			| \
			|  conv2d
			|  relu
			|  conv2d
			| /
			|/
			+ (addition here)
			|
			result
			"""

			"""
			Doing scaling here as mentioned in the paper:

			`we found that increasing the number of feature
			maps above a certain level would make the training procedure
			numerically unstable. A similar phenomenon was
			reported by Szegedy et al. We resolve this issue by
			adopting the residual scaling with factor 0.1. In each
			residual block, constant scaling layers are placed after the
			last convolution layers. These modules stabilize the training
			procedure greatly when using a large number of filters.
			In the test phase, this layer can be integrated into the previous
			convolution layer for the computational efficiency.'

			"""
			scaling_factor = 0.1
		
			#Add the residual blocks to the model
			for i in range(num_layers):
				x = utils.resBlock(x,feature_size,scale=scaling_factor)

			#One more convolution, and then we add the output of our first conv layer
			x = slim.conv2d(x,feature_size,[3,3])
			x += conv_1
		
			#Upsample output of the convolution		
			x = utils.upsample(x,scale,feature_size,None)

			#One final convolution on the upsampling output
			output = x#slim.conv2d(x,output_channels,[3,3])
			self.out = tf.clip_by_value(output+mean_x,0.0,255.0)

			self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,output))
	
			#Calculating Peak Signal-to-noise-ratio
			#Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
			mse = tf.reduce_mean(tf.squared_difference(image_target,output))	
			PSNR = tf.constant(255**2,dtype=tf.float32)/mse
			PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
	
			#Scalar to keep track for loss
			tf.summary.scalar("loss",self.loss)
			tf.summary.scalar("PSNR",PSNR)
			#Image summaries for input, target, and output
			tf.summary.image("input_image",tf.cast(self.input,tf.uint8))
			tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
			tf.summary.image("output_image",tf.cast(self.out,tf.uint8))
		
			#Tensorflow graph setup... session, saver, etc.
			self.sess = tf.Session()
			self.saver = tf.train.Saver()
			print("Done ONLY one GPU model building!")
		else:



			print("Building EDSR in mult GPU mode...")
			gpu_num = check_available_gpus()
			names = locals()

			if gpu_num == 1:
				print('error: only one gpu in PC, Can NOT use the mult GPU mode and sys exit')
				os._exit(0)	
			
			total_loss = []	

			#the batch set has been split
			img_input = tf.split(image_input, int(gpu_num))
			img_target = tf.split(image_target, int(gpu_num))

			for gpu_id in range(int(gpu_num)):
				with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
				    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
						print('device_index --> gpu%2d'%(gpu_id))
						names['self.output_%d'%(gpu_id)] = output = EDSR_model(img_input[gpu_id], num_layers, feature_size, scale,reuse = (gpu_id > 0))
						names['self.out_%d'%(gpu_id)] = tf.clip_by_value(output+mean_x,0.0,255.0)
						names['self.loss_%d'%(gpu_id)] = loss = tf.reduce_mean(tf.losses.absolute_difference(img_target[gpu_id],output))				        
						total_loss.append(loss)
						#print('total_loss shape:',total_loss)
						
			self.loss = tf.reduce_mean(tf.stack(total_loss, axis=0))

			for gpu_id in range(int(gpu_num)):
				#Calculating Peak Signal-to-noise-ratio
				#Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
				mse = tf.reduce_mean(tf.squared_difference(img_target[gpu_id],names['self.output_%d'%(gpu_id)]))	
				PSNR = tf.constant(255**2,dtype=tf.float32)/mse
				PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
	
				#Scalar to keep track for loss
				tf.summary.scalar("loss_gpu %d"%(gpu_id),names['self.loss_%d'%(gpu_id)])
				tf.summary.scalar("PSNR_gpu %d"%(gpu_id),PSNR)

				#Image summaries for input, target, and output
				tf.summary.image("input_image %d" %(gpu_id) ,tf.cast(img_input[gpu_id],tf.uint8))
				tf.summary.image("target_image %d"%(gpu_id) ,tf.cast(img_target[gpu_id],tf.uint8))
				tf.summary.image("output_image %d"%(gpu_id) ,tf.cast(names['self.out_%d'%(gpu_id)],tf.uint8))
			
			#Scalar to keep track for loss
			tf.summary.scalar("total_loss",self.loss)
			#Tensorflow graph setup... session, saver, etc.
			self.sess = tf.Session()
			self.saver = tf.train.Saver()
			print("Done MULT GPU model building!")




	





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
	def predict(self,x):
		print("Predicting...")
		if (len(x.shape) == 3) and not(x.shape[0] == self.img_size and x.shape[1] == self.img_size):
			num_across = x.shape[0]//self.img_size
			num_down = x.shape[1]//self.img_size
			tmp_image = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,3])
			for i in range(num_across):
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[i*tmp.shape[0]:(i+1)*tmp.shape[0],j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			#this added section fixes bottom right corner when testing
			if (x.shape[0]%self.img_size != 0 and  x.shape[1]%self.img_size != 0):
				tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]]})[0]
				tmp_image[-1*tmp.shape[0]:,-1*tmp.shape[1]:] = tmp
					
			if x.shape[0]%self.img_size != 0:
				for j in range(num_down):
					tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})[0]
					tmp_image[-1*tmp.shape[0]:,j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
			if x.shape[1]%self.img_size != 0:
				for j in range(num_across):
                                        tmp = self.sess.run(self.out,feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})[0]
                                        tmp_image[j*tmp.shape[0]:(j+1)*tmp.shape[0],-1*tmp.shape[1]:] = tmp
			return tmp_image
		else:
			return self.sess.run(self.out,feed_dict={self.input:x})

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
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)
		#Make new save directory
		os.mkdir(save_dir)
		#Just a tf thing, to merge all summaries into one
		merged = tf.summary.merge_all()
		#Using adam optimizer as mentioned in the paper
		optimizer = tf.train.AdamOptimizer()

		#This is the train operation for our objective
		train_op = optimizer.minimize(self.loss, colocate_gradients_with_ops = True)
		if self.mult_gpu == False:
			print('the optimizer ONLY one GPU mode')
		else:
			print('the optimizer MULT GPU mode')	
			
		#Operation to initialize all variables
		init = tf.global_variables_initializer()
		print("Begin training...")

		loss_max = 1e8
		loss_init = 0

		every_batch_test = min(20,step_in_epoch)
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
				test_x,test_y = self.test_data(*self.test_args)
				test_feed = {self.input:test_x,self.target:test_y}

			#This is our training loop, the loop is the batch data.NOT epoch
			print('total step is %d, and each epoch has %d step , epoch is %d'%(iterations*step_in_epoch, step_in_epoch, iterations))
			for i in tqdm(range(iterations*step_in_epoch)):
				#Use the data function we were passed to get a batch every iteration
				x,y = self.data(*self.args)
				#Create feed dictionary for the batch
				feed = {
					self.input:x,
					self.target:y
				}
				#Run the train op and calculate the train summary
				summary,_ = sess.run([merged,train_op],feed)

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
					elif loss_now > loss_init * error_scale:
						print('sys is stop\r\nthe loss is boom... and the value is %f'%(loss_now))
						os._exit(0)

				#Write train summary for this step
				train_writer.add_summary(summary,i)
				
				#Save our trained model
				#if (i+1) % step_in_epoch == 0 or (i+1) % (step_in_epoch/5) == 0 :		
				if save_flag == True:
					self.save()		
