"""
This code was used to optimize the partitioning and hierarchical clustering 
algorithms and parameters for HW2. See code comments for details.
"""

import numpy as np
import scipy.cluster.hierarchy as hac
from hw2skeleton import *
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

# For silhouette
from sklearn import metrics

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Read in active sites    
active_sites = io.read_active_sites('data')

# Compute similarity metric and build pairwise distance matrix
active_site_score_dict = {}
active_site_score_list = []
active_site_list = []
matrix = np.zeros([len(active_sites), len(active_sites)])
for i in range(len(active_sites)):
    for j in range(len(active_sites)):
        active_site_scores, matrix[i,j] = cluster.compute_similarity(active_sites[i],active_sites[j])
        active_site_score_dict.update(active_site_scores)
        
for key, value in active_site_score_dict.items():
    active_site_list.append(key)
    active_site_score_list.append(value)

# From http://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
# This converts the n*n matrix to condense nC2 array for scipy
distArray = ssd.squareform(matrix)

# Hierarchical clustering: try lots of different types of agglomerative clustering
z_single = hac.linkage(distArray, method="single")
plt.figure(1)
d_single = hac.dendrogram(z_single)
plt.title("single linkage")
plt.savefig('single linkage')

plt.figure(2)
z_complete = hac.complete(distArray)
d_complete = hac.dendrogram(z_complete)
plt.title("complete linkage")
plt.savefig('complete linkage')

plt.figure(3)
z_centroid = hac.centroid(distArray)
d_centroid = hac.dendrogram(z_centroid)
plt.title("centroid linkage")
plt.savefig('centroid linkage')

plt.figure(4)
z_weighted = hac.weighted(distArray)
d_weighted = hac.dendrogram(z_weighted)
plt.title("weighted linkage")
plt.savefig('weighted linkage')

plt.figure(5)
z_ward = hac.ward(distArray)
d_ward = hac.dendrogram(z_ward)
plt.title("ward linkage")
plt.savefig('ward linkage')

names = [z_single,z_complete,z_centroid,z_weighted,z_ward]
label_names = ['single linkage','complete linkage','centroid linkage','weighted linkage','ward linkage']
# Go through each kind of agglomerative clustering, and calculate silhouette score at
# various cutpoints in the dendrogram. The cutpoints define the number of clusters.
# Plot each clustering method with its silhouette score over range of cutpoints to
# determine the best performing one
for index,name in enumerate(names):
    sil_cuttree_list = []
    for num in range(2,20):
        sil_cuttree = 0
        cuttree = hac.cut_tree(name, n_clusters = num)
        labels_cuttree = []
        for value in cuttree:
            labels_cuttree.append(int(value))
        sil_cuttree = metrics.silhouette_score(matrix, labels_cuttree, metric='euclidean')
        sil_cuttree_list.append(sil_cuttree)
    # print number of clusters and max silhouette score for each type of hac for writeup
    print((sil_cuttree_list.index(max(sil_cuttree_list))+2),max(sil_cuttree_list))

    plt.figure(6)
    plt.plot(range(2,20), sil_cuttree_list, label=label_names[index]) 
    plt.legend(loc='lower right')
    plt.xlabel('Number of clusters by cutting dendrogram')
    plt.ylabel('Silhouette score')
    plt.show

# Partition clustering: Kmediods. Run 100 times over a range of k values and calculate 
# mean and stdev silhouette scores at each k value to determine optimum k for ultimate
# implementation
sil_list = []
mean_sil_list = []
std_sil_list = []
for n in range(2, 50):
    mean_sil=float
    std_sil = float
    for x in range(100):
        sil = 0
        M, C = cluster.kMedoids(matrix,n)
        # Convert dictionary output from kMedoids to list of labels indexed by active site
        labels = np.zeros(len(active_sites))
        for c, value in C.items():
            labels[value] = c
        # Compute silhoutte score
        sil = metrics.silhouette_score(matrix, labels, metric='euclidean')
        sil_list.append(sil)
    mean_sil = np.mean(sil_list)
    std_sil = np.std(sil_list)
    mean_sil_list.append(mean_sil)
    std_sil_list.append(std_sil)
    print(mean_sil, n) # print mean silhouette values so can find max for writeup
plt.figure(7)
plt.errorbar(range(2,50),mean_sil_list,std_sil_list) 
plt.ylim([0,1])
plt.title("K_medoid silhouette plot")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.savefig('K_medoid silhouette plot')
plt.show