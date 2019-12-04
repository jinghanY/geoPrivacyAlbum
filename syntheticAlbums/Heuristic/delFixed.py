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
import math


def get_winner_rank(album_this, ind_delete, label):
	label_scores_prev = np.sum(album_this,axis=0)
	predictions_prev = np.argsort(label_scores_prev)[::-1]

	ind_delete_new = [i for i, x in enumerate(ind_delete) if x ==1]
	album_new = np.delete(album_this, ind_delete_new, axis=0)
	label_scores = np.sum(album_new,axis=0)
	predictions = np.argsort(label_scores)[::-1]
	res = np.where(predictions==label)[0][0]+1
	return res


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
	parser.add_argument("--delete_num","-p", help="level of robustness")
	parser.add_argument("--album_size","-s", help="album size")
	args = parser.parse_args()
	p = int(args.delete_num)
	albumSize = int(args.album_size)
	hm_path = "../dataset/features_npz/model1/"
	locations_path = hm_path + "val"+str(albumSize)+"/"
	locations = os.listdir(locations_path)
	
	num_winner_total_list = []
	ct = 0
	ct_whole = 0
	for location in locations:
		location_path = locations_path + location + "/"
		albums_path = os.listdir(location_path)

		success_rate = []
		num_winner_list = []
		num_deletion_list = []
		num_albums = len(albums_path)
		for i in range(num_albums):
			ct_whole += 1
			album_path = location_path + albums_path[i]
			f = np.load(album_path)
			album_this = f["albumFea"]
			true_winner = f["label"]
			album_id = f["album_id"]
			num_winners_p = h_p(album_this, true_winner,p)
			sys.stdout.write("winner_rank=%d\n"%num_winners_p)
			sys.stdout.flush()

if __name__ == "__main__":
	main()
