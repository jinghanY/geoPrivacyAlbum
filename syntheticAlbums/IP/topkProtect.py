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
	div = div[:, np.newaxis]
	return e_x/div

# Integer Programming
def IP(X, k, albumSize_max):
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
	mds = md.solve(agent='local')
	try:
		num_deletion = mds.get_values(z)
		num_winners = mds.get_values(g)
		md.clear()
		return int(sum(num_deletion)), int(sum(num_winners))
	except:
		num_deletion = np.ones(num_photos)
		num_winners = np.ones(num_labels)
		md.clear()
		return np.float("NaN"), np.float("NaN")

def get_conf_photo(score_matrix, label):
	# get the most confident photo
	score_matrix = softmax(score_matrix)
	photos_label_scores = score_matrix[:, label]
	keep_ids_rank = np.argsort(photos_label_scores)[::-1]
	return keep_ids_rank

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--top_k","-k", help="specify the top k to defeat")
	parser.add_argument("--album_size","-s", help="specify the album size")
	
	args = parser.parse_args()
	albumSize = int(args.album_size)
	k = int(args.top_k)

	hm_path = "../dataset/features_npz/model1/"
	locations_path = hm_path + "val"+str(albumSize)+"/"
	locations = os.listdir(locations_path)
	
	for location in locations:
		location_path = locations_path + location
		albums_path = glob.glob(location_path+"/"+"iter_*.npz")
		num_deletion_list = []
		num_albums = len(albums_path)
		for i in range(len(albums_path)):
			album_path = albums_path[i]
			f = np.load(album_path)
			album_this = f["albumFea"]
			true_winner = f["label"]
			album_id = f["album_id"]
			diff = np.transpose(np.transpose(album_this)-album_this[:,true_winner])
			diff = np.delete(diff, true_winner, axis=1)
			num_deletion, num_winners = IP(diff, k, albumSize)
			sys.stdout.write("%.4f,%.4f\n"%(num_deletion,num_winners))
			sys.stdout.flush()


if __name__ == "__main__":
	main()
