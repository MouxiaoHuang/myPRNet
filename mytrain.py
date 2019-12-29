import numpy as np
import os 
import argparse
import tensorflow as tf
import cv2
import random
from predictor import resfcn256
import math

# Read & obtain TrainData
class TrainData(object):

	def __init__(self,train_data_file):
		super(TrainData,self).__init__()
		self.train_data_file = train_data_file
		self.train_data_list = []
		self.readTrainData()
		self.index = 0
		self.num_data = len(self.train_data_list)
	
	def readTrainData(self):
		with open(self.train_data_file) as fp:
			temp = fp.readlines()
			for item in temp:
				item = item.strip().split()
				self.train_data_list.append(item)
			random.shuffle(self.train_data_list)

	def getBatch(self,batch_list):
		batch = []
		imgs = []
		labels = []
		for item in batch_list:
			if len(item)==2 :
		        	img_name = item[0]
		        	label_name = item[1]
			else :
		        	img_name = item[0] + ' ' + item[1]
		        	label_name = item[2] + ' ' + item[3]
			img = cv2.imread(img_name)
			label = np.load(label_name)
			
			img_array = np.array(img,dtype=np.float32)
			#imgs.append(img_array/255.0)
			imgs.append(img_array/255.0/1.1)#test

			label_array = np.array(label,dtype=np.float32)
			#labels.append(label_array/255.0)
			#labels.append((label_array)/(255.0*1.1))				# optimize it, no reason	因为除以255有点偏移，所以归一化的数值稍加改进（虽然理论不是如此）
			#labels_array_norm = (label_array-label_array.min())/(label_array.max()-label_array.min())
			#labels.append(labels_array_norm)
            #labels_array_norm = (label_array)/(255.0*1.1)
            #labels.append(labels_array_norm)
			labels.append(label_array/255.0/1.1)#test

		batch.append(imgs)
		batch.append(labels)

		return batch

	def __call__(self,batch_num):
		if (self.index+batch_num) <= self.num_data:
			batch_list = self.train_data_list[self.index:(self.index+batch_num)]
			batch_data = self.getBatch(batch_list)
			self.index += batch_num

			return batch_data
		elif self.index < self.num_data:
			batch_list = self.train_data_list[self.index:self.num_data]
			batch_data = self.getBatch(batch_list)
			self.index = 0
			return batch_data
		else:
			self.index = 0
			batch_list = self.train_data_list[self.index:(self.index+batch_num)]
			batch_data = self.getBatch(batch_list)
			self.index += batch_num
			return batch_data


def main(args):
	# Some arguments
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	batch_size = args.batch_size
	epochs = args.epochs
	train_data_file = args.train_data_file							# Obtain TrainData.txt, which contains their paths
	#o_learning_rate = args.learning_rate
	learning_rate = args.learning_rate
	model_path = args.model_path									# Pre-trained model's path

	save_dir = args.checkpoint										# Path of checkpoint
	if not os.path.exists(save_dir):
		print("****************************************************************************************")
		print("dir /checkpoint doesn't exit, so now it is created")
		print("****************************************************************************************")
		os.makedirs(save_dir)

	# Mask
	weight_mask_path = 'Data/uv-data/uv_weight_mask.png'
	face_mask_path = 'Data/uv-data/uv_face_mask.png'
    
	weight_mask = cv2.imread(weight_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
	weight_mask = weight_mask / 255.0								# Normalize it to [0-1]
	face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
	face_mask = face_mask / 255.0
	final_mask = np.multiply(face_mask, weight_mask)
	final_weight_mask = np.zeros(shape=(1,256,256,3)).astype('float32')
	final_weight_mask[0,:,:,0]=final_mask
	final_weight_mask[0,:,:,1]=final_mask
	final_weight_mask[0,:,:,2]=final_mask

	
	# Training data
	data = TrainData(train_data_file)
	
	x = tf.placeholder(tf.float32,shape=[None,256,256,3])
	label = tf.placeholder(tf.float32,shape=[None,256,256,3])
	#weight = tf.placeholder(tf.float32,shape=[None,256,256,3])

	# Train net
	net = resfcn256(256,256)
	print("****************************************************************************************")
	print("the net is printed below:")
	print(net)
	print("****************************************************************************************")
	x_op = net(x,is_training=True)									# Set it to train mode
	
	# Loss
	#loss = tf.losses.mean_squared_error(label,x_op)
	loss = tf.losses.mean_squared_error(label,x_op,weights=final_weight_mask)
	print("****************************************************************************************")
	print("the loss is initialized:")
	print(loss)
	print("****************************************************************************************")


	# This is for batch norm layer
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	global_step = tf.Variable(0, name='global_step', trainable=False)
#	learning_rate = tf.train.exponential_decay(o_learning_rate, global_step, 40000, 0.5, staircase=True)
#	with tf.control_dependencies(update_ops):
#		train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
	sess.run(tf.global_variables_initializer())						# Init the model
	sess.run(global_step.initializer)

#	if os.path.exists(model_path):
#		tf.train.Saver(net.vars).restore(sess.model_path)
#		print("load pre-trained model finished")

#	saver = tf.train.import_meta_graph('/home/mxhuang/projects/myPRNet/checkpoint/256_256_resfcn256_weight.meta')
#	saver.restore(sess, tf.train.latest_checkpoint('/home/mxhuang/projects/myPRNet/checkpoint'))
	# whether to continue training, if not please annotate the line below
#	tf.train.Saver(net.vars).restore(sess, tf.train.latest_checkpoint('./checkpoint'))

#	print("****************************************************************************************")
#	print("pre-trained model loaded, continue training")
#	print("****************************************************************************************")

	saver = tf.train.Saver(var_list = tf.global_variables())
	save_path = model_path
	
	# Begining train
	for epoch in range(epochs):
		for iters in range(int(math.ceil(1.0*data.num_data/batch_size))):
			batch = data(batch_size)			
			loss_res = sess.run(loss,feed_dict={x:batch[0],label:batch[1]})
			sess.run(train_step,feed_dict={x:batch[0],label:batch[1]})

			print('iters:%d/epoch:%d,learning rate:%f,loss:%f'%(iters,epoch,learning_rate,loss_res))
			#print('global_step: %s, iters:%d/epoch:%d, loss:%f'%(tf.train.global_step(sess,global_step),iters,epoch,loss_res))

		saver.save(sess=sess,save_path=save_path)
		if (epoch!=0) and (epoch%5==0):								# Decays half after each 5 epochs
			learning_rate = learning_rate / 2
		
		#if learning_rate <= 0.000001:
		#	break
		#if loss_res <0.00002:
		#	break


if __name__ == '__main__':

	par = argparse.ArgumentParser(description='Training code of PRNet based on tensorflow')

	par.add_argument('--train_data_file',default='Data/trainData/trainDataLabel_all.txt',type=str,help='The txt-file which contains the paths of trainData')
	par.add_argument('--learning_rate',default=0.0001,type=float,help='The learning rate')
	par.add_argument('--epochs',default=1,type=int,help='Total epochs')
	par.add_argument('--batch_size',default=16,type=int,help='Batch sizes')
	par.add_argument('--checkpoint',default='checkpoint/',type=str,help='The path of checkpoint')
	par.add_argument('--model_path',default='checkpoint/256_256_resfcn256_weight',type=str,help='The path of pretrained model')
	par.add_argument('--gpu',default='0',type=str,help='The GPU ID')

	main(par.parse_args())
