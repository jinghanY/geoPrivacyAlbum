# Ayan Chakrabarti <ayanc@ttic.edu>
import re
import os
from glob import glob
import numpy as np


# Find all indices of labels with class cls
def find(labels,cls):
    return np.array(range(len(labels)))[labels == cls]

# Raw load text file
def load(fname,sourcePath):
    data = []
    labels = []
    for line in open(fname).readlines():
        l = line.strip().split(' ')
        data.append(sourcePath + l[0])
        labels.append(int(l[1])-1)
    return data,np.array(labels,dtype=np.int32)

# Reading in batches
# repeatable random shuffling
# Initialize by calling a filename where each line is a string
# (typically a filename) followed by a numerical label.
class batcher:
    def __init__(self,fname,sourcePath,bsz,niter=0):

        # Load from file
        d,l = load(fname,sourcePath)
        self.data = d
        self.labels = l

        # Setup batching
        self.bsz = bsz
        
        self.rand = np.random.RandomState(0)
        idx = self.rand.permutation(len(self.labels))
        for i in range(niter*bsz // len(idx)):
            idx = self.rand.permutation(len(idx))

        self.idx = np.int32(idx)
        self.pos = niter*bsz % len(self.idx)

    def get_batch(self):
        if self.pos+self.bsz >= len(self.idx):
            bidx = self.idx[self.pos:]

            idx = self.rand.permutation(len(self.idx))
            self.idx = np.int32(idx)

            self.pos = 0
            if len(bidx) < self.bsz:
                self.pos = self.bsz-len(bidx)
                bidx2 = self.idx[0:self.pos]
                bidx = np.concatenate((bidx,bidx2))
        else:
            bidx = self.idx[self.pos:self.pos+self.bsz]
            self.pos = self.pos+self.bsz

        return [self.data[bidx[i]]
                for i in range(len(bidx))], self.labels[bidx,...]

# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])
