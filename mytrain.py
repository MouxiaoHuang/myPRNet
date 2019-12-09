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
    def __init__(self, train_data_file):
        super(TrainData, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self, train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            if len(item) == 2:
                img_name = item[0]
                label_name = item[1]
            else:
                img_name = item[0] + ' ' + item[1]
                label_name = item[2] + ' ' + item[3]
            img = cv2.imread(img_name)
            label = np.load(label_name)

            img_array = np.array(img, dtype=np.float32)
            imgs.append(img_array/255.0)

            label_array = np.array(img, dtype=np.float32)
            labels.append(label_array/255.0)
        
        batch.append(imgs)
        batch.append(lables)

        return batch

    def __call__(self, batch_num):
        if(self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data

        elif self.index < self.num_data:
            batch_list = self.train_data_list[self.index + batch_num]
            batch_data = self.getBatch(batch_list)
            self.index = 0
            return batch_data

        else:
            self.index = 0
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data


def main(args):
    # Set some arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu               # Choose the ID of GPU
    batch_size = args.batch_size
    epochs = args.epochs
    train_data_file = args.train_data_file                      # Obtain TrainData.txt, which contains their paths
    learning_rate = args.learning_rate
    model_path = args.model_path                                # Pre-trained model's path

    save_dir = args.checkpoint                                  # Path of checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Mask
    weight_mask_path = 'Data/uv-data/uv_weight_mask.png'
    face_mask_path = 'Data/uv-data/uv_face_mask.png'

    weight_mask = cv2.imread(weight_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
    weight_mask = weight_mask / 255.0                           # Normalize the mask to [0 - 1]
    face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
    face_mask = face_mask / 255.0
    final_mask = np.multiply(face_mask, weight_mask)
    final_weight_mask = np.zeros(shape=(1,256,256,3)).astype('float32')
    final_weight_mask[0,:,:,0]=final_mask
	final_weight_mask[0,:,:,1]=final_mask
	final_weight_mask[0,:,:,2]=final_mask

    # Load TrainData
    data = TrainData(train_data_file)

    x = tf.placeholder(tf.float32, shape=[None,256,256,3])
    label = tf.placeholder(tf.float32, shape=[None,256,256,3])

    # Load net
    net = resfcn256(256, 256)
    x_op = net(x, is_training=True)                             # Set it to train mode

    # Loss
    loss = tf.losses.mean_squared_error(label, x_op, weights=final_weight_mask)

    # For batch norm layer
    update_ops = tf.get_collection(tf.GraphKes.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    if os.path.exists(model_path):
        tf.train.Saver(net.vars).restore(sess.model_path)

    saver = tf.train.Saver(var_list=tf.global_variables())
    save_path = model_path

    # Begin training
    for epoch in range(epochs):
        for iters in range(int(math.ceil(1.0*data.num_data/batch_size))):
            batch = data(batch_size)
            loss_res = sess.run(loss,feed_dict={x:batch[0],label:batch[1]})
			sess.run(train_step,feed_dict={x:batch[0],label:batch[1]})

            print('iters:%d/epoch:%d,learning rate:%f,loss:%f'%(iters,epoch,learning_rate,loss_res))
        
        saver.save(sess=sess, save_path=save_path)
        if (epoch!=0) and (epoch%5==0):                         # Decays half after each 5 epochs
		    learning_rate = learning_rate / 2

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Training code of PRNet based on tensorflow')

    par.add_argument('--train_data_file', default='Data/trainData/trainDataLabel.txt', type=str, help='The txt-file which contains the paths of trainData')
    par.add_argument('--learning_rate', default=0.0001, type=float, help='Learing rate, default=0.0001')
    par.add_argument('--epochs', default=10, type=int, help='The total epochs')     # 10 is just for test
    par.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
    par.add_argument('--checkpoint',default='checkpoint/',type=str,help='The path of checkpoint')
	par.add_argument('--model_path',default='checkpoint/256_256_resfcn256_weight',type=str,help='The path of pretrained model')
	par.add_argument('--gpu',default='0',type=str,help='The GPU ID')