import argparse
import pickle 
import time
import numpy as np
import readList
import os
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model","-m", help="model: 1 or 2")
parser.add_argument("--album_size","-s", help="album size")
args = parser.parse_args()

modelNo = int(args.model)
albumSize = int(args.album_size)


pt = "../../dataset/imgs_features/model"+str(modelNo)+"/"
f = pt + "imgsValFea.pickle"
with open(f, "rb") as handle:
	d = pickle.load(handle)

VAL= "../../dataset/albumlists/albumlist"+str(albumSize)+".txt"
model_path = "../../dataset/features_npz/model"+str(modelNo)+"/"
npz_path = model_path + "val"+str(albumSize)+"/"

DISP_FREQ = 10

if os.path.exists(npz_path):
	shutil.rmtree(npz_path)
os.makedirs(npz_path)
albums_id, photoNames, albums_label = readList.load(VAL)
num_albums = len(albums_id)
accuracy1 = []
accuracy5 = []

for i in range(num_albums):
	album_photos_scores = []
	album_label = albums_label[i]
	album_id = albums_id[i]
	if not os.path.isdir(npz_path+str(album_label)+"/"):
		os.mkdir(npz_path+str(album_label)+"/")
	npz_path_this = npz_path+str(album_label)+"/"
	
	imgs = photoNames[i]
	for j in range(len(imgs)):
		img = imgs[j]
		album_photos_scores.append(d[img])
	album_photos_scores = np.array(album_photos_scores)
	np.savez(npz_path_this+'iter_%s.npz'%album_id, albumFea = album_photos_scores, label = album_label, album_id = album_id)
	album_score = np.mean(album_photos_scores, axis = 0)
	ind_descend = np.argsort(album_score)[::-1]
	top1 = ind_descend[0]
	top5 = ind_descend[:5]
	if top1 == album_label:
		accuracy1.append(True)
	else:
		accuracy1.append(False)
	if album_label in top5:
		accuracy5.append(True)
	else:
		accuracy5.append(False)
	if i % DISP_FREQ == 0:
		top1_acc = np.mean(accuracy1)
		top5_acc = np.mean(accuracy5)
		tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
		sys.stdout.write(tmstr + " [%09d] Val.accuracy1 = %.6f, Val.accuracy5 = %.6f\n" % (i, top1_acc, top5_acc))
		sys.stdout.flush()






















