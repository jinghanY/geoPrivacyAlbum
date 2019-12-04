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

# Integer Programming
def IP(X, p, albumSize_max):
	time1 = time.time()
	md = Model(name='photoAlbum')
	context = md.context
	context.cplex_parameters.threads = 1
	num_photos, num_labels = X.shape
	M = np.abs((albumSize_max)*np.max(X))*100
	photos_range = range(num_photos)
	labels_range = range(num_labels)
	z = [md.binary_var(name='z_{0}'.format(i)) for i in photos_range]
	z = np.array(z)
	w = [md.continuous_var(name='w_{0}'.format(j)) for j in labels_range]
	w = np.array(w)
	g = [md.binary_var(name='g_{0}'.format(j)) for j in labels_range]
	g = np.array(g)


	# write constraints
	for j in range(num_labels):
		md.add_constraint(w[j] >= 0)
		md.add_constraint(w[j] >=-M*g[j])
		md.add_constraint(w[j]<=M*g[j])
		md.add_constraint(w[j]>=md.sum(X[:,j]*(1-z)) - (1-g[j])*M)
		md.add_constraint(w[j]<=md.sum(X[:,j]*(1-z)) + (1-g[j])*M)

	md.add_constraint(md.sum(z)<=p)
	obj = md.sum(g)
	md.maximize(obj)
	mds = md.solve(agent='local')
	ind_deletion = np.array(mds.get_values(z))
	num_deletion = int(sum(ind_deletion))
	num_winners = int(sum(mds.get_values(g)))

	md.clear()
	return num_winners, num_deletion, ind_deletion 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--delete_num","-p", help="level of robustness") # Deletion Budget
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
			diff = np.transpose(np.transpose(album_this)-album_this[:,true_winner])
			diff = np.delete(diff, true_winner, axis=1)
			num_winners,num_deletion, ind_delete = IP(diff, p, albumSize)
			if math.isnan(num_deletion): 
				ct += 1
				continue
			
			rank_true_winner = get_winner_rank(album_this, ind_delete, true_winner)
			sys.stdout.write("winner_rank=%d\n"%rank_true_winner)  
			sys.stdout.flush()
	fail_rate = ct/ct_whole
	sys.stdout.write("fail_rate=%.4f\n"%fail_rate)
	sys.stdout.flush()

if __name__ == "__main__":
	main()
