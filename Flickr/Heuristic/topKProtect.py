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
import pickle

def softmax(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div  = div[:, np.newaxis]
	return e_x /div

def h(F, true_label, k):
	S = softmax(F)
	idx_sort = np.argsort(S[:,true_label])[::-1]
	n, m = np.shape(S)
	F = F[idx_sort,:]
	for i in range(0,n):
		F_r = F[i:,:]
		f_v = np.sum(F_r, axis=0)
		topk = np.argsort(f_v)[::-1][:k]
		if true_label not in topk:
			return i
	return n


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--top_k","-k", help="specify the top k to defeat")
	args = parser.parse_args()
	
	k = int(args.top_k)
	
	albumSize_max = 400
	rank_range = 16
	hm_path = "../dataset/"
	locations_path = hm_path + "Flickr_album/"
	locations = os.listdir(locations_path)
	
	albums_all_locs = []
	album_size_all_locs = []	
	
	for location in locations:
		location_path = locations_path + location
		albums_path = glob.glob(location_path+"/"+"iter_*.npz")
		num_albums = len(albums_path)
		for album_path in albums_path:
			f = np.load(album_path)
			album_this = f["albumFea"]
			true_winner = f["label"]
			album_id = f["album_id"]
			album_size,_ = np.shape(album_this)

			if album_size < rank_range: continue
			num_deletion = h(album_this, true_winner, k)
			num_deletion = np.float(num_deletion/album_size)
			sys.stdout.write("%.4f\n"%num_deletion)
			sys.stdout.flush()


if __name__ == "__main__":
	main()
