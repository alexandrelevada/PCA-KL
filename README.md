# PCA-KL

Python code for the paper 

Levada, A.L.M. PCA-KL: a parametric dimensionality reduction approach for unsupervised metric learning. 
Advances in Data Analysis and Classification, 15, 829–868 (2021). 
https://doi.org/10.1007/s11634-020-00434-3

PCA-KL is a parametric algorithm for unsupervised dimensionality reduction based on the computation of the entropic covariance matrix, a surrogate for the covariance matrix of the data using the relative entropy between Gaussian distributions instead of the usual Euclidean distance between data points. The PCA-KL algorithm can be summarized as:

1. From the input data build an undirected proximity graph using the KNN rule;
2. For each patch, that is, a central point and its neighbors compute the mean and variance of each feature:
	 For simplicity we assume a Gaussian model, but other distributions could be adopted at this stage. At the end, this step generates for each patch, a parametric vector.
3. Compute the mean parametric vector for all patches, which represents the average distribution, given all the dataset:
4. Compute the matrix C, a surrogate for the covariance matrix based on the relative entropy (KL-divergence) between each parametric vector and the average distribution;
5. Select the d < m eigenvectors associated to the d largest eigenvectors of the surrogate matrix C to compose the projection matrix;
6. Project the data into the linear subspace.

