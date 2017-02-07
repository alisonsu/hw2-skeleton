#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code from skikit-lean, obtained from:
https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/metrics/cluster/unsupervised.py#L23
See commenting on the code below.
"""

def silhouette_score(X, labels, metric='euclidean', sample_size=None,
                     random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficent is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.
    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.
    labels : array, shape = [n_samples]
         Predicted labels for each sample.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
        array itself, use ``metric="precomputed"``.
    sample_size : int or None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : integer or numpy.RandomState, optional
        The generator used to randomly select a subset of samples if
        ``sample_size is not None``. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    
    # The main point of this function is to calculate the mean silhouette score
    # of the entire clustering after calculating each individual silhouette score
    # for each sample in each cluster (ie for each active site)
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))


def silhouette_samples(X, labels, metric='euclidean', **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficent is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.
    This function returns the Silhouette Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`. If X is
        the distance array itself, use "precomputed" as the metric.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array, shape = [n_samples]
        Silhouette Coefficient for each samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr']) # This checks to make sure the number of samples is the same as the number of labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), X.shape[0])

    distances = pairwise_distances(X, metric=metric, **kwds) #since my metric is precomputed, no pairwise distances are calculated
    unique_labels = le.classes_ #counts number of clusters
    n_samples_per_label = bincount(labels, minlength=len(unique_labels)) #counts number of occurrences of each label

    # For sample i, store the mean distance of the cluster to which
    # it belongs in intra_clust_dists[i]
    intra_clust_dists = np.zeros(distances.shape[0], dtype=distances.dtype)

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists = np.inf + intra_clust_dists

    for curr_label in range(len(unique_labels)):

        # Find inter_clust_dist for all samples belonging to the same
        # label (cluster).
        mask = labels == curr_label #mask = all active sites in cluster indexed by label
        current_distances = distances[mask] # take a submatrix of distances that involve active sites within the cluster

        # Leave out current sample.
        n_samples_curr_lab = n_samples_per_label[curr_label] - 1 # total number of samples in that cluster minus 1
        if n_samples_curr_lab != 0:
            intra_clust_dists[mask] = np.sum(
                current_distances[:, mask], axis=1) / n_samples_curr_lab # calculate the mean distance of each active site in the cluster

        # Now iterate over all other labels (clusters), finding the mean
        # inter-cluster distance, and save distance of closest neighboring cluster
        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = np.mean(
                    current_distances[:, other_mask], axis=1)
                inter_clust_dists[mask] = np.minimum(
                    inter_clust_dists[mask], other_distances)
    # This is the definition of a silhouette score where all necessary parts have now been calculated
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # score 0 for clusters of size 1, according to the paper
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples