from .utils import Atom, Residue, ActiveSite
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.
    
    The similarity metric is related to the biochemical properties of the residues
    that make up each active site. Each active site gets a score for the number 
    of charged, polar, and nonpolar residues contained, respectively. The contribution
    of each residue to the score is weighted by the distance of its nearest atom to
    the active site centroid. The Euclidean distance between the 3 scores of each
    active site is computed, and this distance is what is stored in the similarity
    matrix.

    Input: two ActiveSite instances
    Output: the 3D coordinates that define the active site biochemically (for plotting),
    and the euclidean distance between the two active sites (as input to similarity matrix)
    """

    charged_residues = ['ARG','LYS','ASP','GLU']
    polar_residues = ['GLN','ASN','HIS','SER','THR','TYR','CYS']
    nonpolar_residues = ['ALA','ILE','LEU','PHE','VAL','PRO','GLY','MET','TRP']
        
    euc_dist = 0
    sites = [site_a, site_b]
    active_site_score = {}
    all_asite = []
    for active_site in sites:
    # filter PDB data to only include 1 instance of active site
        i=0   
        residue_cent_dict = {} # dictionary of residues and centroids of residues
        res_atom_dict = {} # dictionary of residues, their atoms, and corresponding coordinates
        for residue in active_site.residues:
            residue_cent_dict[str(residue)] = {}
            res_atom_dict[str(residue)] = {}
        
        # build dictionary of residues and centroids, dictionary of residues, atoms, and coordinates
        # building dictionary eliminates repeated sites in the PDB files
        for residue in active_site.residues:
            coord_list_residue = []
            j = 0
            coord_sum = [0,0,0]
            num_atoms = 0
            for atom in active_site.residues[i].atoms:
                coord_list = []
                k = 0
                num_atoms+=1
                for coord in active_site.residues[i].atoms[j].coords:
                    coord_list.append(coord)
                    coord_sum[k]+=coord
                    k += 1
                res_atom_dict[str(residue)][str(atom)] = coord_list
                j+=1
            coord_cent = [0,0,0]
            
            for index,value in enumerate(coord_sum):
                coord_cent[index] = value/num_atoms
            coord_list_residue.append(coord_list)        
            residue_cent_dict[str(residue)] = coord_cent       
            i+=1
        
        # calculate centroid of active site    
        asite_centroid_sum = [0,0,0]    
        res_total = 0 
        for residue,centroid in residue_cent_dict.items():
            res_total += 1
            for index in range(len(centroid)):
                asite_centroid_sum[index] += centroid[index]
        asite_centroid = []        
        for cen in asite_centroid_sum:
            asite_centroid.append(cen/res_total) #asite_centroid is active site centroid
        
         
        res_dist = {}
        # calculate distance of closest atom to centroid
        for residue, atom_dict in res_atom_dict.items():
            min_dist = 0
            for atom, coord in atom_dict.items():
                dist = math.sqrt(math.pow(coord[0]-asite_centroid[0],2)+math.pow(coord[1]-asite_centroid[1],2)+math.pow(coord[2]-asite_centroid[2],2))
                if min_dist == 0:
                    min_dist = dist
                elif dist < min_dist:
                    min_dist=dist
            res_dist[residue] = min_dist
        
        # Calculate score for each category of residue
        charged_score = 0
        polar_score = 0
        non_polar_score = 0        
        active_site_score_list = []
        for residue, distance in res_dist.items():
            residue = str(residue)
            residue = residue.split()
            if residue[0] in charged_residues:
                charged_score += (1/distance * 100)
            elif residue[0] in polar_residues:
                polar_score += (1/distance * 100)
            elif residue[0] in nonpolar_residues:
                non_polar_score += (1/distance * 100)
            else:
                print ("Should not get here")
        active_site_score_list.append(charged_score)
        active_site_score_list.append(polar_score)
        active_site_score_list.append(non_polar_score)
        active_site_score[active_site] = active_site_score_list
        all_asite.append(active_site_score_list)
        
    #calculate Euclidean distance
    euc_dist = math.sqrt(math.pow(all_asite[0][0]-all_asite[1][0],2)+math.pow(all_asite[0][1]-all_asite[1][1],2)+math.pow(all_asite[0][2]-all_asite[1][2],2))

    return (active_site_score, euc_dist)


def cluster_by_partitioning(active_sites, k):
    """
    Cluster a given set of ActiveSite instances using a partitioning method (kMedoids).

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
    """
    # First need to construct pairwise distance matrix using similarity metric
    active_site_score_dict = {}
    active_site_score_list = []
    active_site_list = []
    matrix = np.zeros([len(active_sites), len(active_sites)])
    for i in range(len(active_sites)):
        for j in range(len(active_sites)):
            active_site_scores, matrix[i,j] = compute_similarity(active_sites[i],active_sites[j])
            active_site_score_dict.update(active_site_scores)
        
    for key, value in active_site_score_dict.items():
        active_site_list.append(key)
        active_site_score_list.append(value)
            
    # Plot in 3D space     
    plot_init_3D(active_site_score_dict)

    # Run kmedoids clustering on the pairwise distance matrix
    M, C = kMedoids(matrix,k)
    # Make labels from kmedoids into list: each index = active site, value is cluster
    labels = np.zeros(len(active_sites))
    for c, value in C.items():
        labels[value] = c
              
    # Compute silhouette score
    if len(active_sites) > 3:
          sil = metrics.silhouette_score(matrix, labels, metric='euclidean')
          print("Silhouette score for partition clustering:",sil)
    
    # Sort label list to be in same order as active_site_list   
    label_list = []    
    for site in active_site_list:
        label_list.append(labels[active_sites.index(site)])

    # Separate active site scores into 3 lists for plotting
    charged_c = []
    polar_c = []
    nonpolar_c = []    
    for score in active_site_score_list:
        charged_c.append(score[0])
        polar_c.append(score[1])
        nonpolar_c.append(score[2])
    
    # Generate color map for plotting each cluster a different color
    color_map = {0:'r',1:'b',2:'c',3:'m',4:'k'}   
    label_color = [color_map[l] for l in label_list]
    
    # Plot dataset with clusters colored according to kmedoids output
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(charged_c, polar_c, nonpolar_c, c=label_color)
    ax.set_xlabel('Charged score')
    ax.set_ylabel('Polar score')
    ax.set_zlabel('Nonpolar score')  
    plt.show(block=False)
    plt.savefig('Active sites plotted after clustering using partitioning method')
    
    return(M, C)
     
def kMedoids(D, k, tmax=100):
    """
    Obtained from: https://github.com/letiantian/kmedoids/blob/master/kmedoids.py
    Implementation of kmedoids partitioning algorithm
    """
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoids, referenced by indices
    # for example, if M = 1, the medoid would be the active site in active_sites[1]
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax): # stopping condition if no optimum is found 
        # determine clusters, i. e. arrays of data indices
        # A submatrix of D is made that consists of the distances of all values to the medoids, where
        # the medoid is the column index. J stores the index of the smallest distance of each active site
        # (by row) compared to each column (the medoids), thereby storing which active site should join which cluster
        # defined by the medoid
        J = np.argmin(D[:,M], axis=1)
        
        # Then assign each active site to its cluster based on the output of J
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
            
        # update cluster medoids by finding the active site with lowest average distance
        # to all other active sites in the cluster and reassigning that cluster medoid to the new active site
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        
        # check for convergence - if nothing changed during the previous round of medoid assignments, exit the loop
        # Otherwise, 
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    # return results
    return (M, C)   
    
def cluster_hierarchically(active_sites, n):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # First need to construct pairwise distance matrix using similarity metric
    active_site_score_dict = {}
    active_site_score_list = []
    active_site_list = []
    matrix = np.zeros([len(active_sites), len(active_sites)])
    for i in range(len(active_sites)):
        for j in range(len(active_sites)):
            active_site_scores, matrix[i,j] = compute_similarity(active_sites[i],active_sites[j])
            active_site_score_dict.update(active_site_scores)
        
    for key, value in active_site_score_dict.items():
        active_site_list.append(key)
        active_site_score_list.append(value)
            
    # Plot in 3D space 
    plot_init_3D(active_site_score_dict)       

    # From http://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
    # This converts the n*n matrix to condense nC2 array for scipy    
    distArray = ssd.squareform(matrix)
    
    z_centroid = hac.centroid(distArray)
    
    cuttree = hac.cut_tree(z_centroid, n_clusters = n)
    labels_cuttree = []
    for value in cuttree:
        labels_cuttree.append(int(value))
    # Sort label list to be in same order as active_site_list   
    label_list = []    
    for site in active_site_list:
        label_list.append(labels_cuttree[active_sites.index(site)])
    # Calculate Silhouette score:
    if len(active_sites) > 3:
        sil_cuttree = metrics.silhouette_score(matrix, labels_cuttree, metric='euclidean')
        print("Silhouette score for hierarchical clustering:",sil_cuttree)
    
    cluster_dict = {}
    for index, lbl in enumerate(label_list):
        try:
            test = cluster_dict[lbl]
        except KeyError:
            cluster_dict[lbl] = []
        cluster_dict[lbl].append(index)  
    
    # Separate active site scores into 3 lists for plotting
    charged_c = []
    polar_c = []
    nonpolar_c = []    
    for score in active_site_score_list:
        charged_c.append(score[0])
        polar_c.append(score[1])
        nonpolar_c.append(score[2])
    
    # Generate color map for plotting each cluster a different color
    color_map = {0:'r',1:'b',2:'c',3:'m',4:'k'}   
    label_color = [color_map[l] for l in label_list]
    
    # Plot dataset with clusters colored according to hac output
    #fig = plt.figure(2)
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(charged_c, polar_c, nonpolar_c, c=label_color)
    #ax.set_xlabel('Charged score')
    #ax.set_ylabel('Polar score')
    #ax.set_zlabel('Nonpolar score')
    #plt.show()  

    return (cluster_dict)
    
def plot_init_3D(active_site_score_dict):
    """
    Takes in the active site score dictionary and plots each active site in 3D space
    based on the charged, polar, and nonpolar scores calculated for the similarity
    metric
    """
    charged = []
    polar = []
    nonpolar = []        
    for key, lst in active_site_score_dict.items():
        charged.append(lst[0])
        polar.append(lst[1])
        nonpolar.append(lst[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(charged, polar, nonpolar, c='r', marker='o')
    
    ax.set_xlabel('Charged score')
    ax.set_ylabel('Polar score')
    ax.set_zlabel('Nonpolar score')
    plt.savefig('Active sites plotted based on similarity metric')
    plt.show(block=False)
