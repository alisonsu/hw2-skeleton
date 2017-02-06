from hw2skeleton import cluster
from hw2skeleton import io
import os
import numpy as np
import math
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hac

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    # update this assertion
    x, y = cluster.compute_similarity(activesite_a, activesite_b)
    assert int(y) == 106

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))
    
    M, C = cluster.cluster_by_partitioning(active_sites,3)
    
    labels = np.zeros(3, dtype=np.int64)
    for c, value in C.items():
        labels[value] = (c)

    answer = np.array([0, 1, 2])
    
    # update this assertion
    assert np.all(labels==answer)
  
def test_partition_clustering_simple():


    X = np.array([[0,0,0], [0,1,0], [0,0,1], [100,100,100], [100,110,100]])
    matrix = np.zeros([len(X), len(X)])    
    for i in range(len(X)):
        for j in range(len(X)):
            matrix[i,j] = math.sqrt(math.pow(X[i][0]-X[j][0],2)+math.pow(X[i][1]-X[j][1],2)+math.pow(X[i][2]-X[j][2],2))
    M, C = cluster.kMedoids(matrix,2)
    
    labels = np.zeros(len(X), dtype=np.int64)
    for c, value in C.items():
        labels[value] = (c)

    
    answer = np.array([0, 0, 0, 1, 1])
        
    # update this assertion
    assert np.all(labels == answer)

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    # assert cluster.cluster_hierarchically(active_sites,3) == []
   
def test_hierarchical_clustering_simple():


    X = np.array([[0,0,0], [0,1,0], [0,0,1], [100,100,100], [100,110,100]])
    matrix = np.zeros([len(X), len(X)])    
    for i in range(len(X)):
        for j in range(len(X)):
            matrix[i,j] = math.sqrt(math.pow(X[i][0]-X[j][0],2)+math.pow(X[i][1]-X[j][1],2)+math.pow(X[i][2]-X[j][2],2))
    
    distArray = ssd.squareform(matrix)
    
    z_centroid = hac.centroid(distArray)
    cuttree = hac.cut_tree(z_centroid, n_clusters = 2)
    labels_cuttree = []
    for value in cuttree:
        labels_cuttree.append(int(value))
    
    answer = np.array([0, 0, 0, 1, 1])
        
    # update this assertion
    assert np.all(labels_cuttree == answer)

