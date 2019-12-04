import time 
import tensorflow as tf
import numpy as np

class trainload:
	def __init__(self,bsz,scales=[256,512]):
		self.names = []
		
		# Create placeholders
		for i in range(bsz):
			self.names.append(tf.placeholder(tf.string))

		batch = []
		sizes = tf.constant(np.float32(scales))
		for i in range(bsz):
			img = tf.read_file(self.names[i])
			code = tf.decode_raw(img, tf.uint8)[0] 
			img = tf.cond(tf.equal(code,137),
						lambda: tf.image.decode_png(img,channels=3),
						lambda: tf.image.decode_jpeg(img,channels=3))  
			
			# Resize image to a random scale in scales
			in_s = tf.to_float(tf.shape(img)[:2])
			min_s = tf.minimum(in_s[0],in_s[1])
			size = tf.random_shuffle(sizes)[0]  
			new_s = tf.to_int32((size/min_s)*in_s)
			img = tf.image.resize_images(img,new_s)
			# Randomly flip image
			img = tf.image.random_flip_left_right(img)
			
			# Randomly crop image
			img = tf.random_crop(img,[224,224,3])

			batch.append(tf.reshape(img,[1,224,224,3]))

		batch = tf.to_float(tf.concat(batch,0))

		# Fetching logic
		nBuf = tf.Variable(tf.zeros([bsz,224,224,3],dtype=tf.float32),trainable=False)
		self.batch = tf.Variable(tf.zeros([bsz,224,224,3],dtype=tf.float32),trainable=False)
		self.fetchOp = tf.assign(nBuf,batch)
		self.swapOp = tf.assign(self.batch,nBuf)
	
	def getfeed(self,imgs):
		dict = {}
		for i in range(len(self.names)):
			dict[self.names[i]] = imgs[i]
		return dict

def testload(name,size):
		# load image
		img = tf.image.decode_jpeg(tf.read_file(name),channels=3)

		# Resize image to specific scale.
		in_s = tf.to_float(tf.shape(img)[:2])
		min_s = tf.minimum(in_s[0],in_s[1])
		new_s = tf.to_int32((size/min_s)*in_s)
		img = tf.image.resize_images(img,new_s)
		return tf.expand_dims(tf.to_float(img),0)
			



