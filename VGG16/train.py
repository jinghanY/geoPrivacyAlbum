import sys
import time
import tensorflow as tf
import numpy as np
import loader as ldr
import trainer as tr
import model as md
import utils as ut
from config import *

# Config options here
KEEPLAST = 50
SAVE_FREQ=1000
DISP_FREQ=100
BSZ=64
WEIGHT_DECAY=0.
LR = 0.0001
MOM = 0.9

MAX_ITER = int(1.2e7)
# val
VAL_FREQ = 100
VAL_numBatches = 20
# Check for saved weights
saved = ut.ckpter(WTS_DIR + 'iter_*.model.npz')
iter = saved.iter
# Set up batching
batcher = ut.batcher(LIST,sourcePath,BSZ,iter)
batcher_val = ut.batcher(VAL,sourcePath,BSZ,iter)

# Set up data prep
data = ldr.trainload(BSZ)

labels = tf.placeholder(shape=(BSZ,),dtype=tf.int32)
labels_val = tf.placeholder(shape=(BSZ,),dtype=tf.int32)

# Load model-def
iftrain=tf.placeholder(tf.bool, shape=[])
net = md.model(data.batch,iftrain)

# Load trainer-def
opt = tr.train(net,labels,LR,MOM,WEIGHT_DECAY)
# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

init =False
if init:
	sys.stdout.write("Initializing from the pre-trained model.\n")
	net.loadInitial(WTS_DIR+'init.model.npz',sess)

# Load saved weights if any
if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest,sess)
    saved.clean(last=KEEPLAST)

train_loss = []
train_accuracy1 = []
train_accuracy5 = []
val_loss = []
val_accuracy1 = []
val_accuracy5 = []

stop=False
try:
	imgs, nlbls = batcher.get_batch()
	_=sess.run(data.fetchOp, feed_dict= data.getfeed(imgs))
	while iter < MAX_ITER and not stop:
		_=sess.run(data.swapOp)
		clbls = nlbls
		
		# Run training step & fetch for next batch
		imgs,nlbls = batcher.get_batch()
		fdict = data.getfeed(imgs)
		fdict[labels] = clbls
		fdict[iftrain] = True
		outs =sess.run([opt.accuracy1,opt.accuracy5,opt.loss, opt.tstep, data.fetchOp],feed_dict=fdict) 
		train_accuracy1.append(outs[0])
		train_accuracy5.append(outs[1])
		train_loss.append(outs[2])
		

		if iter % VAL_FREQ == 0:
			imgs_val, nlbls_val = batcher_val.get_batch()
			_=sess.run(data.fetchOp, feed_dict=data.getfeed(imgs_val))
			for i in range(VAL_numBatches):
				_=sess.run(data.swapOp)
				clbls_val = nlbls_val
				imgs_val, nlbls_val = batcher_val.get_batch()
				fdict_val = data.getfeed(imgs_val)
				fdict_val[labels] = clbls_val
				fdict_val[iftrain] = False
				outs_val = sess.run([opt.accuracy1, opt.accuracy5, opt.loss, data.fetchOp], feed_dict = fdict_val)
				val_accuracy1.append(outs_val[0])
				val_accuracy5.append(outs_val[1])
				val_loss.append(outs_val[2])
			loss_val = np.mean(val_loss)
			accuracy1_val = np.mean(val_accuracy1)
			accuracy5_val = np.mean(val_accuracy5)
			val_loss = []
			val_accuracy1 = []
			val_accuracy5 = []
			tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
			sys.stdout.write(tmstr + " [%09d] lr=%.2e Val.loss = %.6f,  Val.accuracy1 = %.6f, Val.accuracy5 = %.6f\n" % (iter,LR,loss_val,accuracy1_val,accuracy5_val))
			sys.stdout.flush()
			imgs, nlbls = batcher.get_batch()
			_=sess.run(data.fetchOp,feed_dict=data.getfeed(imgs))

		# Display frequently
		if iter % DISP_FREQ == 0:
			loss = np.mean(train_loss)
			accuracy1 = np.mean(train_accuracy1)
			accuracy5 = np.mean(train_accuracy5)
			train_loss = []
			train_accuracy1 = []
			train_accuracy5 = []
			
			tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
			sys.stdout.write(tmstr + " [%09d] lr=%.2e Train.loss = %.6f, Train.accuracy1 = %.6f,Train.accuracy5 = %.6f\n" % (iter,LR,loss,accuracy1,accuracy5))
			sys.stdout.flush()
		
		iter = iter+1

		if iter % SAVE_FREQ == 0:
			fname = WTS_DIR + "iter_%d.model.npz" % iter
			net.save(fname,sess)
			saved.clean(last=KEEPLAST)
			sys.stdout.write("Saved weights to " + fname + "\n")
			sys.stdout.flush()

except KeyboardInterrupt: # Catch ctrl+c
    sys.stderr.write("Stopped!\n")
    sys.stderr.flush()
    stop = True
    pass

if saved.iter < iter:    
    fname = WTS_DIR + "iter_%d.model.npz" % iter
    net.save(fname,sess)
    saved.clean(last=KEEPLAST)
    sys.stdout.write("Saved weights to " + fname + "\n")
    sys.stdout.flush()
