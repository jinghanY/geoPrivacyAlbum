import argparse
from tqdm import tqdm
import itertools
import time
from docplex.mp.model import Model
import cplex
import numpy as np
import glob
import sys
import os
import shutil
import math
"""
Bounds
"""

def softmax(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div  = div[:, np.newaxis]
	return e_x /div

def h_p(F, true_label, p):
	S = softmax(F)
	idx_sort = np.argsort(S[:,true_label])[::-1]
	n, m = np.shape(S)
	F = F[idx_sort,:]
	F_r = F[p:,:]
	label_scores = np.sum(F_r, axis=0)
	predictions = np.argsort(label_scores)[::-1]
	res = np.where(predictions==true_label)[0][0]
	return res

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--delete_prop","-p", help="Deleting proportion")
	args = parser.parse_args()
	fp = float(args.delete_prop)
	
	albumSize_max = 400
	
	hm_path = "../dataset/"
	locations_path = hm_path + "Flickr_album/"
	locations = os.listdir(locations_path)

	ct = 0
	ct_whole = 0
	
	for location in locations:
		location_path = locations_path + location
		albums_path = glob.glob(location_path+"/"+"iter_*.npz")
		num_winner_list = []
		for album_path in albums_path:
			ct_whole += 1
			f = np.load(album_path)
			album_this = f["albumFea"]
			true_winner = f["label"]
			album_id = f["album_id"]
			album_size,_ = np.shape(album_this)
			if album_size < 16: continue
			p = int(np.round(fp*album_size))
			num_winners_p = h_p(album_this, true_winner,p)
			sys.stdout.write("winner_rank=%d\n"%num_winners_p)
			sys.stdout.flush()
			

if __name__ == "__main__":
	main()
