import numpy as np
import sys

path = sys.argv[1]

F = np.load(path)
print("load TF matrix of USER-FRIENDS")
print(F)

# get IDF(f) where f from friends

# get the number of documents
N = F.shape[0]
# tag - friend
TF = F / np.sum(F, axis=1, keepdims=True)
TF = np.nan_to_num(TF)
print(N)

# get log(N+1)/(N(x)+1) + 1
# N_x = np.sum((TF > 0), axis=0)
# print(N_x)
# IDF = np.log10((N+1)/(N_x+1)) + 1
# print(IDF)

# get log c(j)/c(h',j)
cj = np.sum(F, axis=0, keepdims=True)
ch_j = cj - F
IDF = np.log10((cj+1)/(ch_j+1)) + 1

np.save(path.replace(".npy", "_tfidf.npy"), TF*IDF)
print(TF*IDF)

# tag v.s. friend