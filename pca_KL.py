#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA-KL: a parametric algorithm based on relative entropy for unsupervised 
dimensionality reduction

Created on Wed Jul 24 16:59:33 2019

@author: Alexandre L. M. Levada

"""

# Imports
import warnings
import numpy as np
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# Computes the KL-divergence between two Gaussian distributions
def divergenciaKL(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:
        sigma1 = 0.001
    if sigma2 == 0:
        sigma2 = 0.001
    dKL1 = np.log(np.sqrt(sigma2)/np.sqrt(sigma1)) + (sigma1 + (mu1 - mu2)**2)/(2*sigma2) - 0.5
    dKL2 = np.log(np.sqrt(sigma1)/np.sqrt(sigma2)) + (sigma2 + (mu2 - mu1)**2)/(2*sigma1) - 0.5
    return 0.5*(dKL1 + dKL2)

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%%%%%%%%%%%%%%%%%%%  Data loading

# Dataset iris
X = skdata.load_iris()   # use K = 20
dados = X['data']
target = X['target']

# OpenML datasets
#X = skdata.fetch_openml(name='cardiotocography', version=1) # use K = 90 for best results
#dados = X['data']
#target = X['target']    # in this dataset, target is a vector of strings

#lista = []
#for x in target:
#  lista.append(int(x[0]))
#target = np.array(lista)


#%%%%%%%%%%%%%%%%%%%%%% Data processing
# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

# Inicial matrices for storing the means and variances for each patch
medias = np.zeros((dados.shape[0], dados.shape[1]))
variancias = np.zeros((dados.shape[0], dados.shape[1]))

# Creates a KNN graph from the dataset (the value of K sffects the results )
esparsa = sknn.kneighbors_graph(dados, 20, mode='connectivity', include_self=True)
A = esparsa.toarray()

# Computes the local means and variances for each patch
for i in range(dados.shape[0]):
    vizinhos = A[i, :]
    indices = vizinhos.nonzero()[0]
    amostras = dados[indices]
    medias[i, :] = amostras.mean(0)
    matriz_covariancias = np.cov(amostras.T)
    for j in range(dados.shape[1]):
        variancias[i, j] = matriz_covariancias[j, j]

# Compute the average parameters (for the average distribution)       
mus = medias.mean(0)
sigmas = variancias.mean(0)

# Define the surrogate for the covariance matrix
matriz_final = np.zeros((dados.shape[1], dados.shape[1]))

# Define the vector of KL-divergences
vetor_dKL = np.zeros(dados.shape[1])

# Computes the surrogate for the covariance matrix
for i in range(dados.shape[0]):
    for j in range(dados.shape[1]):
        vetor_dKL[j] = divergenciaKL(medias[i,j], variancias[i,j], mus[j], sigmas[j])
        matriz_final = matriz_final + np.outer(vetor_dKL, vetor_dKL)
matriz_final = matriz_final/dados.shape[0]

#%%%%%%%%%%%%%%%%%%%%%%%  PCA-KL 
# Eigenvalues and eigenvectors of the surrogate matrix
v, w = np.linalg.eig(matriz_final)

# Sort the eigenvalues
ordem = v.argsort()

# Select the two eigenvectors associated to the two largest eigenvalues
maior_autovetor = w[:, ordem[-1]]
segundo_maior = w[:, ordem[-2]]

# Projection matrix
WpcaKL = np.array([maior_autovetor, segundo_maior])

# Linear projection into the 2D subspace
novos_dados = np.dot(WpcaKL, dados.T)

#%%%%%%%############## Simple PCA 
# Eigenvalues and eigenvectors of the covariance matrix
v1, w1 = np.linalg.eig(np.cov(dados.T))

# Sort the eigenvalues
ordem = v1.argsort()

# Select the two eigenvectors associated to the two largest eigenvalues
maior_autovetor1 = w1[:, ordem[-1]]
segundo_maior1 = w1[:, ordem[-2]]

# Projection matrix
Wpca = np.array([maior_autovetor1, segundo_maior1])

# Linear projection into the 2D subspace
novos_dados_pca = np.dot(Wpca, dados.T)


#%%%%%%%%%%%%%%%%%%% Data plotting for PCA-KL

# Number of classes
nclass = int(max(target))

plt.figure(1)
for i in range(nclass+1):
    indices = np.where(target==i)[0]
    if (i == 0):
        cor = 'blue'
    elif (i == 1):
        cor = 'red'
    elif (i == 2):
        cor = 'green'
    elif (i == 3):
        cor = 'black'
    elif (i == 4):
        cor = 'yellow'
    elif (i == 5):
        cor = 'magenta'
    elif (i == 6):
        cor = 'cyan'
    elif (i == 7):
        cor = 'orange'
    elif (i == 8):
        cor = 'darkkhaki'
    elif (i == 9):
        cor = 'brown'
    else:
        cor = 'aqua'
    plt.scatter(novos_dados[0, indices], novos_dados[1, indices], c=cor)    

plt.axis([-3.0, 3.0, -2.0, 2.0])  # for iris dataset only

plt.show()

#%%%%%%%%%%%%%%%%%% Data plotting for PCA

plt.figure(2)
for i in range(nclass+1):
    indices = np.where(target==i)[0]
    if (i == 0):
        cor = 'blue'
    elif (i == 1):
        cor = 'red'
    elif (i == 2):
        cor = 'green'
    elif (i == 3):
        cor = 'black'
    elif (i == 4):
        cor = 'yellow'
    elif (i == 5):
        cor = 'magenta'
    elif (i == 6):
        cor = 'cyan'
    elif (i == 7):
        cor = 'orange'
    elif (i == 8):
        cor = 'darkkhaki'
    elif (i == 9):
        cor = 'brown'
    else:
        cor = 'aqua'
    plt.scatter(novos_dados_pca[0, indices], novos_dados_pca[1, indices], c=cor)    

plt.show()

#%%%%%%%%%%%%%%%%%%%% Supervised classification for PCA features

print('Supervised classification for PCA features')
print()

X_train, X_test, y_train, y_test = train_test_split(novos_dados_pca.real.T, target, test_size=.4, random_state=42)

# KNN
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train) 
print('KNN accuracy: ', neigh.score(X_test, y_test))

# SMV
svm = SVC(gamma='auto')
svm.fit(X_train, y_train) 
print('SVM accuracy: ', svm.score(X_test, y_test))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print('NB accuracy: ', nb.score(X_test, y_test))

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
print('DT accuracy: ', dt.score(X_test, y_test))

# Quadratic Discriminant 
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
print('QDA accuracy: ', qda.score(X_test, y_test))

# MPL classifier
mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
mpl.fit(X_train, y_train)
print('MPL accuracy: ', mpl.score(X_test, y_test))

# Gaussian Process
gpc = GaussianProcessClassifier()
gpc.fit(X_train, y_train)
print('GPC accuracy: ', gpc.score(X_test, y_test))

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('RFC accuracy: ', rfc.score(X_test, y_test))

# Computes the Silhoutte coefficient
print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_pca.real.T, target, metric='euclidean'))
print()

#%%%%%%%%%%%%%%%%%%%% Supervised classification for PCA-KL features

print('Supervised classification for PCA-KL features')
print()

X_train, X_test, y_train, y_test = train_test_split(novos_dados.real.T, target, test_size=.4, random_state=42)

# KNN
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train) 
print('KNN accuracy: ', neigh.score(X_test, y_test))

# SMV
svm = SVC(gamma='auto')
svm.fit(X_train, y_train) 
print('SVM accuracy: ', svm.score(X_test, y_test))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print('NB accuracy: ', nb.score(X_test, y_test))

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
print('DT accuracy: ', dt.score(X_test, y_test))

# Quadratic Discriminant 
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
print('QDA accuracy: ', qda.score(X_test, y_test))

# MPL classifier
mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
mpl.fit(X_train, y_train)
print('MPL accuracy: ', mpl.score(X_test, y_test))

# Gaussian Process
gpc = GaussianProcessClassifier()
gpc.fit(X_train, y_train)
print('GPC accuracy: ', gpc.score(X_test, y_test))

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('RFC accuracy: ', rfc.score(X_test, y_test))

# Computes the Silhoutte coefficient
print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados.real.T, target, metric='euclidean'))