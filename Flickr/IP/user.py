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
"""
Bounds
"""

def softmax(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis]
	return e_x/div

# Integer Programming
def IP(X, k, albumSize_max, keep_id):
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

	md.add_constraint(z[keep_id] == 0)
	md.add_constraint(md.sum(g)>=k)
	# write objective
	md.minimize(md.sum(z))
	time2= time.time()
	mds = md.solve(agent='local')
	try:
		num_deletion = mds.get_values(z)
		num_winners = mds.get_values(g)
		md.clear()
		return int(sum(num_deletion))
	except:
		md.clear()
		return np.float("NaN")

def get_conf_photo(score_matrix, label, rank_range):
	# get the most confident photo
	score_matrix = softmax(score_matrix)
	photos_label_scores = score_matrix[:, label]
	keep_ids_rank = np.argsort(photos_label_scores)[::-1][:rank_range]
	return keep_ids_rank

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--top_k","-k", help="specify the top k to defeat")
	k = int(args.top_k)
	
	albumSize_max = 400
	rank_range = 16
	hm_path = "../dataset/"
	locations_path = hm_path + "Flickr_album/"
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
			album_size,_ = np.shape(album_this)
			if album_size < rank_range: continue
			album_score = np.mean(album_this, axis=0)
			diff = np.transpose(np.transpose(album_this)-album_this[:,true_winner])
			diff = np.delete(diff, true_winner, axis=1)
			keep_ids_rank = get_conf_photo(album_this, true_winner, rank_range)
			rank_deletion_list = []
			for j in range(len(keep_ids_rank)):
				keep_id = keep_ids_rank[j]
				num_deletion = IP(diff, k, albumSize_max, keep_id)
				frac_deletion = num_deletion/np.float(album_size)
				rank_deletion_list.append(frac_deletion)
			for ele in rank_deletion_list:
				sys.stdout.write("%.4f,"%ele)
			sys.stdout.write("\n")
			sys.stdout.flush()
	

if __name__ == "__main__":
	main()
