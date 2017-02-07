# -*- coding: utf-8 -*-
"""
This includes the most important code used from the scipy.cluster.hierarchical
package to demonstrate my understanding of the implementation

The code was obtained from:
https://github.com/scipy/scipy/blob/master/scipy/cluster/hierarchy.py
https://github.com/scipy/scipy/blob/master/scipy/cluster/_hierarchy.pyx
https://github.com/scipy/scipy/blob/master/scipy/cluster/_hierarchy_distance_update.pxi

Note: the other methods use a function that uses nearest-neighbor chain algorithm.
I am only showing the code for hac.centroid, which is the algorithm I ultimately
implemented
"""
    
def centroid(y):
    """
    This would be the first function called when implementing hac.centroid()
    It calls to use linkage (below)
    """
    return linkage(y, method='centroid', metric='euclidean')
    

def linkage(y, method='single', metric='euclidean'):
    """
    This function goes through all options from hac and sends the call to the appropriate
    function. For my purposes, "centroid" goes to _hierarchy.fast_linkage
    """
    # Basic error handling
    if method not in _LINKAGE_METHODS:
        raise ValueError("Invalid method: {0}".format(method))

    y = _convert_to_double(np.asarray(y, order='c'))

    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
        [y] = _copy_arrays_if_base_present([y])
    elif y.ndim == 2:
        if method in _EUCLIDEAN_METHODS and metric != 'euclidean':
            raise ValueError("Method '{0}' requires the distance metric "
                             "to be Euclidean".format(method))
        if y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0):
            if np.all(y >= 0) and np.allclose(y, y.T):
                _warning('The symmetric non-negative hollow observation '
                         'matrix looks suspiciously like an uncondensed '
                         'distance matrix')
        y = distance.pdist(y, metric)
    else:
        raise ValueError("`y` must be 1 or 2 dimensional.")

    if not np.all(np.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only finite values.")
    # Call appropriate function from user input. My case calls _hierarchy.fast_linkage
    n = int(distance.num_obs_y(y))
    method_code = _LINKAGE_METHODS[method]
    if method == 'single':
        return _hierarchy.mst_single_linkage(y, n)
    elif method in ['complete', 'average', 'weighted', 'ward']:
        return _hierarchy.nn_chain(y, n, method_code)
    else:
        return _hierarchy.fast_linkage(y, n, method_code)

def fast_linkage(double[:] dists, int n, int method):
    """
    This function implements the clustering. See comments below for details.
    Simple comments from GitHub were included, but I elaborated quite a bit and included my own
    """
    # Initialize variables 
    cdef double[:, :] Z = np.empty((n - 1, 4))

    cdef double[:] D = dists.copy()  # Copy of distances between clusters.
    cdef int[:] size = np.ones(n, dtype=np.intc)  # Sizes of clusters (starts with each active site in its own cluster)
    # ID of a cluster to put into linkage matrix.
    cdef int[:] cluster_id = np.arange(n, dtype=np.intc)

    # Nearest neighbor candidate and lower bound of the distance to the
    # true nearest neighbor for each cluster among clusters with higher
    # indices (thus size is n - 1).
    cdef int[:] neighbor = np.empty(n - 1, dtype=np.intc)
    cdef double[:] min_dist = np.empty(n - 1)

    # Specify how the distance matrix will be updated. For my case, it will be 
    # by linking centroids of clusters. This function is elaborated on after this function
    cdef linkage_distance_update new_dist = linkage_methods[method]

    cdef int i, k
    cdef int x, y, z
    cdef int nx, ny, nz
    cdef int id_x, id_y
    cdef double dist
    cdef Pair pair

    # For each active site, find the closest active site to it and store index of active site
    # and distance in min_dist queue
    for x in range(n - 1):
        pair = find_min_dist(n, D, size, x)
        neighbor[x] = pair.key
        min_dist[x] = pair.value
    cdef Heap min_dist_heap = Heap(min_dist)

    for k in range(n - 1):
        # Theoretically speaking, this can be implemented as "while True", but
        # having a fixed size loop when floating point computations involved
        # looks more reliable. The idea that we should find the two closest
        # clusters in no more that n - k (1 for the last iteration) distance
        # updates.
        for i in range(n - k):
            # Find the minimum distance pair from queue and store values
            pair = min_dist_heap.get_min()
            x, dist = pair.key, pair.value
            y = neighbor[x]

            if dist == D[condensed_index(n, x, y)]:
                break
            #recalculate nearest neighbors
            pair = find_min_dist(n, D, size, x)
            y, dist = pair.key, pair.value
            neighbor[x] = y
            min_dist[x] = dist
            min_dist_heap.change_value(x, dist)
        min_dist_heap.remove_min() # remove distance from queue

        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = size[x]
        ny = size[y]
        
        # Keep order consistent
        if id_x > id_y:
            id_x, id_y = id_y, id_x
        
        # Put information of which clusters joined together, what their distance was,
        # and how many active sites were joined by merging the clusters
        Z[k, 0] = id_x 
        Z[k, 1] = id_y
        Z[k, 2] = dist
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the newly merged
        cluster_id[y] = n + k  # Update ID of y.

        # Update the distance matrix.
        for z in range(n):
            nz = size[z]
            if nz == 0 or z == y:
                continue
            # This is where each distance in the pairwise distance matrix is updated
            # For my code, this is by linking centroids (see function cdef_double_centroid)
            D[condensed_index(n, z, y)] = new_dist(
                D[condensed_index(n, z, x)], D[condensed_index(n, z, y)],
                dist, nx, ny, nz)

        # Reassign neighbor candidates from x to y.
        # This reassignment is just a (logical) guess.
        for z in range(x):
            if size[z] > 0 and neighbor[z] == x:
                neighbor[z] = y

        # Update lower bounds of distance.
        for z in range(y):
            # ignore if z has already been assigned to a new cluster
            if size[z] == 0:
                continue

            dist = D[condensed_index(n, z, y)]
            if dist < min_dist[z]:
                neighbor[z] = y
                min_dist[z] = dist
                min_dist_heap.change_value(z, dist)

        # Find nearest neighbor for y and update min_dist and min_dist_heap (queue)
        if y < n - 1:
            pair = find_min_dist(n, D, size, y)
            z, dist = pair.key, pair.value
            if z != -1:
                neighbor[y] = z
                min_dist[y] = dist
                min_dist_heap.change_value(y, dist)

    return Z.base # returns stepwise dendrogram

"""
Below is the code for calculating the distance between centroids of clusters used
to update the distance matrix
"""    
cdef double _centroid(double d_xi, double d_yi, double d_xy,
                      int size_x, int size_y, int size_i):
    return sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                 (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                (size_x + size_y))

    
def cut_tree(Z, n_clusters=None, height=None):
    """
    Given a linkage matrix Z, return the cut tree.
    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.
    n_clusters : array_like, optional
        Number of clusters in the tree at the cut point.
    height : array_like, optional
        The height at which to cut the tree.  Only possible for ultrametric
        trees.
    Returns
    -------
    cutree : array
        An array indicating group membership at each agglomeration step.  I.e.,
        for a full cut tree, in the first column each data point is in its own
        cluster.  At the next step, two nodes are merged.  Finally all singleton
        and non-singleton clusters are in one group.  If `n_clusters` or
        `height` is given, the columns correspond to the columns of `n_clusters` or
        `height`.
    """
    nobs = num_obs_linkage(Z) # This simply counts the total observations from input matrix M, which is 136 for the active site example
    nodes = _order_cluster_tree(Z) # This creates a list of nodes from the bottom up

    if height is not None and n_clusters is not None:
        raise ValueError("At least one of either height or n_clusters "
                         "must be None")
    elif height is None and n_clusters is None:  # return the full cut tree
        cols_idx = np.arange(nobs)
    elif height is not None:
        heights = np.array([x.dist for x in nodes])
        cols_idx = np.searchsorted(heights, height)
    else: # this is what I use by specifying n_clusters-calculates number of columns will need: 136-8 = 128
        cols_idx = nobs - np.searchsorted(np.arange(nobs), n_clusters)

    try:
        n_cols = len(cols_idx)
    except TypeError:  # scalar
        n_cols = 1
        cols_idx = np.array([cols_idx])
    # Initialize array to store cluster ID for each active sites
    groups = np.zeros((n_cols, nobs), dtype=int)
    last_group = np.arange(nobs)
    if 0 in cols_idx:
        groups[0] = last_group
    
    # This iterates over the list of nodes until the number assigned to each leaf node corresponds to its cluster number
    for i, node in enumerate(nodes):
        idx = node.pre_order() # performs preorder transversal of tree from node (root, then left, then right), returns leaf nodes
        this_group = last_group.copy()
        this_group[idx] = last_group[idx].min()
        this_group[this_group > last_group[idx].max()] -= 1
        if i + 1 in cols_idx:
            groups[np.where(i + 1 == cols_idx)[0]] = this_group
        last_group = this_group

    return groups.T #this is a list indexed by each active site containing which cluster it belongs to at cuttoff 