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
	M = np.abs((albumSize_max)*np.max(X))
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

	#md.add_constraint(md.sum(z)>=p)
	md.add_constraint(md.sum(z)<=p)
	#md.add_constraint(md.sum(g)>=0)
	# write objective
	obj = md.sum(g)
	md.maximize(obj)
	#md.minimize(md.sum(z))
	time2= time.time()
	mds = md.solve(agent='local')
	try:
		ind_deletion = np.array(mds.get_values(z))
		num_deletion = int(sum(ind_deletion))
		num_winners = int(sum(mds.get_values(g)))
	except:
		ind_deletion = float("nan")
		num_deletion = float("nan")
		num_winners = float("nan")

	md.clear()
	return num_winners, num_deletion, ind_deletion 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--delete_num_1","-p1", help="level of robustness")
	parser.add_argument("--delete_num_2","-p2", help="level of robustness")
	parser.add_argument("--album_size","-size", help="album size")
	args = parser.parse_args()
	p1 = int(args.delete_num_1)
	p2 = int(args.delete_num_2)
	albumSize = int(args.album_size)
	hm_path = "../dataset/features_npz/"
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
			
			num_winners1,num_deletion1, ind_delete1 = IP(diff, p1, albumSize)
			if math.isnan(num_deletion1): 
				ct += 1
				continue
			no_total = np.sum(ind_delete1)

			num_winners2,num_deletion2, ind_delete2 = IP(diff, p2, albumSize)
			no_same = np.sum(ind_delete1[ind_delete1 == ind_delete2])
			ratio = (no_total - no_same)/no_total
			sys.stdout.write("ratio=%.4f\n"%ratio)
			sys.stdout.flush()
			

if __name__ == "__main__":
	main()
