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

	#md.add_constraint(md.sum(z)<=k)
	md.add_constraint(md.sum(g)>=k)
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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--top_k","-k",help="specify the top k to defeat")
	
	args = parser.parse_args()
	k = int(args.top_k)
	
	rank_range = 16
	albumSize_max = 400 
	
	hm_path = "../dataset/" 
	locations_path = hm_path + "Flickr_album/"
	locations = os.listdir(locations_path)
	
	rank_deletion_whole = []
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
			diff = np.transpose(np.transpose(album_this)-album_this[:,true_winner])
			diff = np.delete(diff, true_winner, axis=1)
			num_deletion = IP(diff, k, albumSize_max)
			frac_deletion = num_deletion/np.float(album_size)
			sys.stdout.write("%.4f\n"%frac_deletion)	
			sys.stdout.flush()
	

if __name__ == "__main__":
	main()
