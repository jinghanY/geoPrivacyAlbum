import numpy as np

def load(fname):
	photoAlbumNum = []
	label = []
	photoNames = []
	for line in open(fname).readlines():
		l = line.strip().strip(',').split(' ')
		photoAlbumNum.append(int(l[0]))
		label.append(int(l[1])-1)
		photoName_this = l[2].strip().strip(',').split(',')
		photoName_this = [photoName_this[i] for i in range(len(photoName_this))]
		photoNames.append(photoName_this)
	return photoAlbumNum, photoNames, np.array(label,dtype=np.int32)
