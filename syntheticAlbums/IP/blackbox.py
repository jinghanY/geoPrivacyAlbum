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

# Integer Programming
def IP(X, k, albumSize_max):
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

	md.add_constraint(md.sum(g)>=k)
	md.minimize(md.sum(z))
	time2= time.time()
	mds = md.solve(agent='local')
	num_deletion = mds.get_values(z)
	num_winners = mds.get_values(g)
	md.clear()
	time3 = time.time()
	return int(sum(num_deletion)), int(sum(num_winners)), time3 - time1, time3 - time2, np.array(num_deletion)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--eps","-e", help="level of robustness")
	parser.add_argument("--album_size","-s", help="album size")
	parser.add_argument("--top_k","-k", help="specify the top k to defeat")
	args = parser.parse_args()
	eps = float(args.eps)
	albumSize = int(args.album_size)
	k = int(args.top_k)
	
	hm_path = "../dataset/features_npz/model1/"
	locations_path = hm_path + "val"+str(albumSize)+"/"
	locations = os.listdir(locations_path)
	
	hm_pt_model2 = "../dataset/features_npz/model2/"
	locations_pt_model2 = hm_pt_model2 + "val"+str(albumSize)+"/"

	for location in locations:
		location_path = locations_path + location + "/"
		albums_path = os.listdir(location_path)
		location_path_model2 = locations_pt_model2 + location + "/"

		success_rate = []
		num_deletion_list = []
		num_albums = len(albums_path)
		for i in range(num_albums):
			album_path = location_path + albums_path[i]
			f = np.load(album_path)
			album_this = f["albumFea"]
			true_winner = f["label"]
			album_id = f["album_id"]
			album_this[:,true_winner] = album_this[:,true_winner]+eps
			diff = np.transpose(np.transpose(album_this)-album_this[:,true_winner])
			diff = np.delete(diff, true_winner, axis=1)
			num_deletion, num_winners, time_cost_whole, time_cost_solver, ind_delete = IP(diff, k, albumSize)
			num_deletion_list.append(num_deletion)	
			album_path_model2 = location_path_model2 + albums_path[i]
			f2 = np.load(album_path_model2)
			album_model2 = f2["albumFea"]
			true_winner_model2 = f2["label"]
			album_id_model2 = f2["album_id"]
			ind_delete_model1 = np.where(ind_delete==1)[0]
			album_delete_model2 = np.delete(album_model2, ind_delete_model1, axis=0)
			album_delete_sum_model2 = np.sum(album_delete_model2, axis = 0)
			predictions_delete_model2 = np.argsort(album_delete_sum_model2)[::-1][:k]
			if true_winner_model2 in predictions_delete_model2:
				success_rate.append(0)
			else:
				success_rate.append(1)
		ave_deletion = np.mean(np.array(num_deletion_list))
		ave_success_rate = np.mean(np.array(success_rate))
		num_deletion_list = []
		sys.stdout.write("location=%d,num_albums=%d,num_deletion=%.4f,success_rate=%.4f\n" % (true_winner, num_albums,ave_deletion,ave_success_rate))
		sys.stdout.flush()
	

if __name__ == "__main__":
	main()
