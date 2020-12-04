import sys
import time
import tensorflow as tf
import numpy as np
import os
import loader as ldr
import trainer as tr
import model as md
import utils as ut
#print('set up configs')
# Config options here
LIST=''
WTS_DIR=''
sourcePath = ''
#DISP_FREQ=10
BSZ=80
WEIGHT_DECAY=0.
LR = 0.001
MOM = 0.9

MAX_ITER = int(1.2e7)

# numbers for validation
VAL_numBatches = 64

# Set up data prep
data = ldr.trainload(BSZ)
labels = tf.placeholder(shape=(BSZ,),dtype=tf.int32)
# Load model-def
net = md.model(data.batch,train=True)
# Load trainer-def
opt = tr.train(net,labels,LR,MOM,WEIGHT_DECAY)
# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def valRes(weightName, iter):
	net.load(weightName,sess)
	batcher = ut.batcher(LIST,sourcePath,BSZ,iter)
	val_loss = []
	val_accuracy = []
	imgs, nlbls = batcher.get_batch()
	_=sess.run(data.fetchOp,feed_dict=data.getfeed(imgs))
	for i in range(VAL_numBatches):
		_=sess.run(data.swapOp)
		clbls=nlbls
		imgs,nlbls = batcher.get_batch()
		fdict=data.getfeed(imgs)
		fdict[labels]=clbls
		outs = sess.run([opt.accuracy,opt.loss,data.fetchOp],feed_dict=fdict)
		val_loss.append(outs[1])
		val_accuracy.append(outs[0])
	loss_val = np.mean(val_loss)
	accuracy_val = np.mean(val_accuracy)
	sys.stdout.write(" [%09d] Val loss = %.6f, Val accuracy = %.6f\n"
					% (iter,loss_val,accuracy_val))

weight_names = os.listdir(WTS_DIR)
weight_names.sort()
weight_names = weight_names[1:]
for weight_name in weight_names:
	itr = weight_name.split('.')
	iter = int((itr[0].split('_'))[1])
	valRes(WTS_DIR + weight_name,iter)









