#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Python code for the paper 

Levada, A.L.M. PCA-KL: a parametric dimensionality reduction approach for unsupervised metric learning. 
Advances in Data Analysis and Classification, 15, 829–868 (2021). 
https://doi.org/10.1007/s11634-020-00434-3

@author: Alexandre Levada

"""

# Imports
import warnings
import umap         # Umap package: pip install umap-learn ou conda install -c conda-forge umap-learn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from scipy.special import erf
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Computes the KL-divergence between two Gaussian distributions
def divergenciaKL(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variance cannot be zero
        sigma1 = 0.01
    if sigma2 == 0:     # variance cannot be zero
        sigma2 = 0.01

    dKL1 = np.log(np.sqrt(sigma2)/np.sqrt(sigma1)) + (sigma1 + (mu1 - mu2)**2)/(2*sigma2) - 0.5
    dKL2 = np.log(np.sqrt(sigma1)/np.sqrt(sigma2)) + (sigma2 + (mu2 - mu1)**2)/(2*sigma1) - 0.5

    return 0.5*(dKL1 + dKL2)  # symmetrized KL divergence


# Simple PCA implementation
def myPCA(dados, d):

    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

# Parametric PCA using the KNN graph
def ParametricPCA_KNN(dados, k, d, dist):
    # Inicial matrices for storing the means and variances for each patch
    medias = np.zeros((dados.shape[0], dados.shape[1]))
    variancias = np.zeros((dados.shape[0], dados.shape[1]))

    # Creates a KNN graph from the dataset (the value of K affects the results )
    # The second parameter is the number of neighbors K
    esparsa = sknn.kneighbors_graph(dados, k, mode='connectivity', include_self=True)
    A = esparsa.toarray()
    
    # Computes the local means and variances for each patch
    for i in range(dados.shape[0]):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = dados[indices]
        medias[i, :] = amostras.mean(0)
        variancias[i, :] = amostras.std(0)**2
    
    # Compute the standard deviation for Fisher information based metrics    
    desvios = np.sqrt(variancias)
        
    # Compute the average parameters (for the average distribution)       
    mus = medias.mean(0)
    sigmas = variancias.mean(0)
    # Define the surrogate for the covariance matrix
    matriz_final = np.zeros((dados.shape[1], dados.shape[1]))
    # Define the vector of parametric distances
    vetor = np.zeros(dados.shape[1])

    # Computes the surrogate for the covariance matrix
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            if dist == 'KL':
                vetor[j] = divergenciaKL(medias[i,j], variancias[i,j], mus[j], sigmas[j])
        matriz_final = matriz_final + np.outer(vetor, vetor)
            
    matriz_final = matriz_final/dados.shape[0]
    
    # Eigenvalues and eigenvectors of the surrogate matrix
    v, w = np.linalg.eig(matriz_final)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


# Parametric PCA using e-neighborhood graph
def ParametricPCA_Ball(dados, p, d, dist):

    # Number of features
    m = dados.shape[1]
    # Number of samples
    n = dados.shape[0]

    # Inicial matrices for storing the means and variances for each patch
    medias = np.zeros((n, m))
    variancias = np.zeros((n, m))

    # Creates a KNN graph from the dataset (the value of K affects the results )
    # The second parameter is the number of neighbors K
    esparsa = sknn.kneighbors_graph(dados, n, mode='distance', include_self=True)
    A = esparsa.toarray()

    percentiles = np.percentile(A, p, axis=1)   # Calcula os percentis p para cada vértice do grafo
    # Se distância entre xi e xj é maior que o percentil p, desconecta do grafo
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] <= percentiles[i]:
                A[i, j] = 1
            else:
                A[i, j] = 0

    # Computes the local means and variances for each patch
    for i in range(dados.shape[0]):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = dados[indices]
        medias[i, :] = amostras.mean(0)
        variancias[i, :] = amostras.std(0)**2

    # Compute the standard deviation for Fisher information based metrics    
    desvios = np.sqrt(variancias)
        
    # Compute the average parameters (for the average distribution)       
    mus = medias.mean(0)
    sigmas = variancias.mean(0)
    # Define the surrogate for the covariance matrix
    matriz_final = np.zeros((dados.shape[1], dados.shape[1]))
    # Define the vector of parametric distances
    vetor = np.zeros(dados.shape[1])

    # Computes the surrogate for the covariance matrix
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            if dist == 'KL':
                vetor[j] = divergenciaKL(medias[i,j], variancias[i,j], mus[j], sigmas[j])
        matriz_final = matriz_final + np.outer(vetor, vetor)
            
    matriz_final = matriz_final/dados.shape[0]
    
    # Eigenvalues and eigenvectors of the surrogate matrix
    v, w = np.linalg.eig(matriz_final)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    print('KNN accuracy: ', acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)
    print('SVM accuracy: ', acc)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc = nb.score(X_test, y_test)
    lista.append(acc)
    print('NB accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    print('QDA accuracy: ', acc)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc = mpl.score(X_test, y_test)
    lista.append(acc)
    print('MPL accuracy: ', acc)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc = gpc.score(X_test, y_test)
    lista.append(acc)
    print('GPC accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]

# Usa estratégia de busca para o melhor valor de raio na construção do grafo e-neighborhood
def batch_ParametricPCA_Ball(dados, target, dist):

    # Search for the best redius
    inicio = 1
    incremento = 1
    percs = list(range(inicio, 99, incremento))
    acuracias = []
    scs = []

    for p in percs:
        print('Percentil = %d' %p)
        dados_pcakl = ParametricPCA_Ball(dados, p, 2, dist)
        s = 'Parametric PCA Ball ' + dist
        L_pcakl = Classification(dados_pcakl, target, s)
        scs.append(L_pcakl[0])
        acuracias.append(L_pcakl[1])

    print('List of values for percentiles: ', percs)
    print('Supervised classification accuracies: ', acuracias)
    acuracias = np.array(acuracias)
    print('Max Acc: ', acuracias.max())
    print('P* = ', percs[acuracias.argmax()])
    print()

    plt.figure(1)
    plt.plot(percs, acuracias)
    plt.title('Mean accuracies for different values of percentiles')
    plt.show()

    print('List of values for percentiles: ', percs)
    print('Silhouette Coefficients: ', scs)
    scs = np.array(scs)
    print('Max SC: ', scs.max())
    print('P* = ', percs[scs.argmax()])
    print()

    plt.figure(2)
    plt.plot(percs, scs, color='red')
    plt.title('Silhouette coefficients for different values of percentiles')
    plt.show()


# Usa estratégia de busca para o melhor valor de K na construção do grafo KNN
def batch_ParametricPCA_KNN(dados, target, dist):

    n = dados.shape[0]

    # Search for the best K
    inicio = 5
    incremento = 5
    vizinhos = list(range(inicio, n, incremento))
    acuracias = []
    scs = []

    for viz in vizinhos:
        print('K = %d' %viz)
        dados_pcakl = ParametricPCA_KNN(dados, viz, 2, dist)
        s = 'Parametric PCA KNN ' + dist
        L_pcakl = Classification(dados_pcakl, target, s)
        scs.append(L_pcakl[0])
        acuracias.append(L_pcakl[1])

    print('List of values for K: ', vizinhos)
    print('Supervised classification accuracies: ', acuracias)
    acuracias = np.array(acuracias)
    print('Best Acc: ', acuracias.max())
    print('K* = ', vizinhos[acuracias.argmax()])
    print()

    plt.figure(1)
    plt.plot(vizinhos, acuracias)
    plt.title('Mean accuracies for different values of K')
    plt.show()

    print('List of values for K: ', vizinhos)
    print('Silhouette Coefficients: ', scs)
    scs = np.array(scs)
    print('Best SC: ', scs.max())
    print('K* = ', vizinhos[scs.argmax()])
    print()

    plt.figure(2)
    plt.plot(vizinhos, scs, color='red')
    plt.title('Silhouette coefficients for different values of K')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%  Data loading

# Datasets selecionados 
#X = skdata.load_iris()     
#X = skdata.fetch_openml(name='Engine1', version=1)    # (remover classificador QDA: não funciona pois tem apenas uma amostra na classe 2)
#X = skdata.fetch_openml(name='prnn_crabs', version=1) 
#X = skdata.fetch_openml(name='xd6', version=1) 
#X = skdata.fetch_openml(name='analcatdata_happiness', version=1) 
#X = skdata.fetch_openml(name='analcatdata_wildcat', version=2) 
#X = skdata.fetch_openml(name='mux6', version=1) 
#X = skdata.fetch_openml(name='threeOf9', version=1) 
#X = skdata.fetch_openml(name='diggle_table_a2', version=2) 
#X = skdata.fetch_openml(name='sa-heart', version=1) 
#X = skdata.fetch_openml(name='breast-tissue', version=2) 
#X = skdata.fetch_openml(name='hayes-roth', version=2)  
#X = skdata.fetch_openml(name='rabe_131', version=2)
#X = skdata.fetch_openml(name='plasma_retinol', version=2)  
#X = skdata.fetch_openml(name='aids', version=1) 
#X = skdata.fetch_openml(name='lupus', version=1) 
#X = skdata.fetch_openml(name='pwLinear', version=2)  
#X = skdata.fetch_openml(name='visualizing_livestock', version=1)
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)  
#X = skdata.fetch_openml(name='sensory', version=2)
#X = skdata.fetch_openml(name='fri_c3_100_5', version=2)
#X = skdata.fetch_openml(name='analcatdata_creditscore', version=1) 
#X = skdata.fetch_openml(name='blogger', version=1)
#X = skdata.fetch_openml(name='chscase_geyser1', version=2)  
#X = skdata.fetch_openml(name='servo', version=1)  
#X = skdata.fetch_openml(name='vineyard', version=2)
#X = skdata.fetch_openml(name='veteran', version=2)  
#X = skdata.fetch_openml(name='disclosure_z', version=2)
#X = skdata.fetch_openml(name='Australian', version=4)  
#X = skdata.fetch_openml(name='monks-problems-1', version=1)
#X = skdata.fetch_openml(name='visualizing_galaxy', version=2) 
#X = skdata.fetch_openml(name='xd6', version=1) 
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='analcatdata_boxing1', version=1) 
#X = skdata.fetch_openml(name='collins', version=2) 
#X = skdata.fetch_openml(name='baskball', version=2)  
#X = skdata.fetch_openml(name='bolts', version=2) 

# Outras opções para testar
#X = skdata.fetch_openml(name='datatrieve', version=1)  
#X = skdata.fetch_openml(name='machine_cpu', version=2) 
#X = skdata.fetch_openml(name='grub-damage', version=2)
#X = skdata.fetch_openml(name='arsenic-female-bladder', version=2)
#X = skdata.fetch_openml(name='breast-cancer-dropped-missing-attributes-values', version=1)
#X = skdata.fetch_openml(name='dbworld-subjects', version=1)
#X = skdata.fetch_openml(name='breast-cancer-dropped-missing-attributes-values', version=1)
#X = skdata.fetch_openml(name='parity5', version=1) # K = 25
#X = skdata.fetch_openml(name='planning-relax', version=1)
#X = skdata.fetch_openml(name='pm10', version=2) 
#X = skdata.fetch_openml(name='cloud', version=2) 
#X = skdata.fetch_openml(name='kc3', version=1)
##X = skdata.fetch_openml(name='KnuggetChase3', version=1) 
#X = skdata.fetch_openml(name='fl2000', version=2) 
#X = skdata.fetch_openml(name='triazines', version=2) 
#X = skdata.fetch_openml(name='fri_c2_100_10', version=2) 
#X = skdata.fetch_openml(name='Touch2', version=1) 
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='diabetes_numeric', version=2) 
#X = skdata.fetch_openml(name='prnn_fglass', version=2) 
#X = skdata.fetch_openml(name='parkinsons', version=1) 
#X = skdata.fetch_openml(name='acute-inflammations', version=2) 
#X = skdata.fetch_openml(name='prnn_viruses', version=1) 
#X = skdata.fetch_openml(name='zoo', version=1) 
X = skdata.fetch_openml(name='confidence', version=2) 
#X = skdata.fetch_openml(name='blood-transfusion-service-center') 
#X = skdata.fetch_openml(name='kc1') 
#X = skdata.fetch_openml(name='mfeat-fourier', version=1) 
#X = skdata.fetch_openml(name='cardiotocography', version=1) 
#X = skdata.fetch_openml(name='texture', version=1)  
#X = skdata.fetch_openml(name='satimage', version=1)  
#X = skdata.fetch_openml(name='first-order-theorem-proving', version=1)  
#X = skdata.fetch_openml(name='wall-robot-navigation', version=1)  
#X = skdata.fetch_openml(name='synthetic_control', version=1)  
#X = skdata.fetch_openml(name='car', version=3)  
#X = skdata.fetch_openml(name='car', version=2)  
#X = skdata.fetch_openml(name='haberman', version=1)  
#X = skdata.fetch_openml(name='heart-statlog', version=1) 
#X = skdata.fetch_openml(name='sonar', version=1)  
#X = skdata.fetch_openml(name='tae', version=1)  
#X = skdata.fetch_openml(name='Speech', version=1)  
#X = skdata.fetch_openml(name='transplant', version=2) 
#X = skdata.fetch_openml(name='chscase_geyser1', version=2) 
#X = skdata.fetch_openml(name='hayes-roth', version=2)  
#X = skdata.fetch_openml(name='SPECTF', version=1)  
#X = skdata.fetch_openml(name='newton_hema', version=2)  
#X = skdata.fetch_openml(name='veteran', version=2)  
#X = skdata.fetch_openml(name='datatrieve', version=1)  
#X = skdata.fetch_openml(name='mu284', version=2)  
#X = skdata.fetch_openml(name='chscase_census6', version=2)  
#X = skdata.fetch_openml(name='disclosure_z', version=2)  
#X = skdata.fetch_openml(name='triazines', version=2)  
#X = skdata.fetch_openml(name='page-blocks', version=2) 
#X = skdata.fetch_openml(name='arsenic-male-lung', version=2)  
#X = skdata.fetch_openml(name='stock', version=2)  
#X = skdata.fetch_openml(name='glass', version=2)  
#X = skdata.fetch_openml(name='mw1', version=1)  
#X = skdata.fetch_openml(name='optdigits', version=2) 
#X = skdata.fetch_openml(name='plasma_retinol', version=2) 
#X = skdata.fetch_openml(name='ar1', version=1) 
#X = skdata.fetch_openml(name='diggle_table_a2', version=2) 
#X = skdata.fetch_openml(name='rmftsa_ladata', version=2)  
#X = skdata.fetch_openml(name='bodyfat', version=2)  
#X = skdata.fetch_openml(name='segment', version=2)  
#X = skdata.fetch_openml(name='strikes', version=2)  
#X = skdata.fetch_openml(name='kc3', version=1)  
#X = skdata.fetch_openml(name='fri_c4_500_10', version=2)  
#X = skdata.fetch_openml(name='jEdit_4.2_4.3', version=1)  
#X = skdata.fetch_openml(name='space_ga', version=2)  
#X = skdata.fetch_openml(name='usp05', version=1)  
#X = skdata.fetch_openml(name='diabetes', version=1)  
#X = skdata.fetch_openml(name='mammography', version=1) 
#X = skdata.fetch_openml(name='banknote-authentication', version=1) 
#X = skdata.fetch_openml(name='spambase', version=1)  
#X = skdata.fetch_openml(name='churn', version=1)  
#X = skdata.fetch_openml(name='balance-scale', version=1) 
#X = skdata.fetch_openml(name='vehicle', version=1)  
#X = skdata.fetch_openml(name='monks-problems-1', version=1) 
#X = skdata.fetch_openml(name='bank-marketing', version=2) 
#X = skdata.fetch_openml(name='steel-plates-fault', version=3)  
#X = skdata.fetch_openml(name='qsar-biodeg', version=1)  
#X = skdata.fetch_openml(name='molecular-biology_promoters', version=1) 
#X = skdata.fetch_openml(name='delta_ailerons', version=1) 
#X = skdata.fetch_openml(name='pc3', version=1) 
#X = skdata.fetch_openml(name='ar4', version=1) 
#X = skdata.fetch_openml(name='KnuggetChase3', version=1) 
#X = skdata.fetch_openml(name='thoracic_surgery', version=1) 

dados = X['data']
target = X['target']  

# Apenas para datasets do OpenML
# Precisa tratar dados categóricos manualmente
cat_cols = dados.select_dtypes(['category']).columns
dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
# Converte para numpy (openml agora é dataframe)
dados = dados.to_numpy()
target = target.to_numpy()

#%%%%%%%%%%%%%%%%%%%%%% Data processing
# Data standardization (to deal with variables having different units/scales)
dados_nn = dados      # Save a copy of the unnormalized data
dados = preprocessing.scale(dados)

# Number of samples
n = dados.shape[0]
print('Number of samples: %d' %n)
# Number of features
m = dados.shape[1]
print('Number of features: %d' %m)
# Number of classes
c = len(np.unique(target))
print('Number of classes: %d' %c)
print()

#%%%%%%%%%%% Parametric PCA e-ball (grafo e-neighborhood): na maioria dos casos tem produzido melhores resultados que o grafo KNN
'''
'KL': Kullback-Leibler divergence (the ideia is to incorporate other distances in the future)
'''
batch_ParametricPCA_Ball(dados, target, 'KL')   # e-neighborhood graph

#%%%%%%%%%% Parametric PCA KNN (grafo KNN)
'''
'KL': Kullback-Leibler divergence (The idea is to incorporate other distances in the future)
'''
#batch_ParametricPCA_KNN(dados, target, 'KL')   # KNN graph

#%%%%%%%%%%%% Simple PCA
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%%% Kernel PCA
model = KernelPCA(n_components=2, kernel='rbf')   
dados_kpca = model.fit_transform(dados)
dados_kpca = dados_kpca.T

#%%%%%%%%%%%% Sparse PCA
model = SparsePCA(n_components=2)   
dados_spca = model.fit_transform(dados)
dados_spca = dados_spca.T

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Lap. Eig.
model = SpectralEmbedding(n_neighbors=20, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#%%%%%%%%%%%%% t-SNE
model = TSNE(n_components=2)
dados_tsne = model.fit_transform(dados)
dados_tsne = dados_tsne.T

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T

#%%%%%%%%% Classifica dados
L_pca = Classification(dados_pca, target, 'PCA')
L_kpca = Classification(dados_kpca, target, 'KPCA')
L_spca = Classification(dados_spca, target, 'SPCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
L_tsne = Classification(dados_tsne, target, 't-SNE')
L_umap = Classification(dados_umap, target, 'UMAP')