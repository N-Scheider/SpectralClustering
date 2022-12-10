#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:04:24 2022

@author: darktanuki
"""

import             numpy as np
import matplotlib.pyplot as plt
import       scipy.stats as st
import pandas            as pd

import scipy.sparse       as sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigs
import scipy.sparse.linalg as linalgs
from sklearn.cluster import KMeans


"""
SimKNearestNeighborGraphs
The frac of non zeros element for k=15  are 0.002199531849817421
"""

"""
SimMutualKNearestNeighborGraphs
The frac of non zeros element for k=10  are 0.0006125401828906713
The frac of non zeros element for k=20  are 0.0013149652008364284
The frac of non zeros element for k=30  are 0.0020235573171873537
The frac of non zeros element for k=40  are 0.002727954807902375
The frac of non zeros element for k=50  are 0.0034347991635716737
The frac of non zeros element for k=60  are 0.004144439936331575
The frac of non zeros element for k=70  are 0.004851334228020349
The frac of non zeros element for k=80  are 0.005561199712867888
The frac of non zeros element for k=90  are 0.006268493492712462
The frac of non zeros element for k=100  are 0.006981105458631129
The frac of non zeros element for k=110  are 0.007687200774008302
The frac of non zeros element for k=120  are 0.00839888892356668
The frac of non zeros element for k=130  are 0.009107680783995506
The frac of non zeros element for k=140  are 0.0098224399987516
The frac of non zeros element for k=150  are 0.010538197933897195
"""

# Creating a n * n sparse matrix =: Sim 
#Sim = csr_matrix((n, n))

# we want to have f: R^{n*d} x R^{n*d} --> R^{n*n}
#                      (x    ,    y  )|--> np.exp(-euclideandist(x,y)) 

def SimFullGraph(X, sigma, treshold=5*1e-5):
    Sim = np.exp(-np.square(euclidean_distances(X,X))/(2*sigma**2))
    Sim = Sim*(Sim>=treshold)
    sSim = csr_matrix(Sim) 
    return sSim

def SimEpsNeighborGraphs(X, eps):
    Sim = euclidean_distances(X,X)
    Sim = Sim*(Sim<=eps)
    sSim = csr_matrix(Sim) 
    return sSim

def SimKNearestNeighborGraphs(X, k):
    n = X.shape[0]
    Sim = euclidean_distances(X,X)
    kth = np.sort(Sim, axis=1)[:,k-1]
    BoolSim = (Sim <= np.repeat(kth,n).reshape((n,n))) + (Sim <= np.repeat(kth,n).reshape((n,n))).T
    Sim = Sim*BoolSim
    sSim = csr_matrix(Sim) 
    #sSim = (sSim + sSim.T)/2
    return sSim

def SimMutualKNearestNeighborGraphs(X, k):
    n = X.shape[0]
    Sim = euclidean_distances(X,X)
    kth = np.sort(Sim, axis=1)[:,k-1]
    BoolSim = (Sim <= np.repeat(kth,n).reshape((n,n))) * (Sim <= np.repeat(kth,n).reshape((n,n))).T
    Sim = Sim*BoolSim
    sSim = csr_matrix(Sim) 
    #sSim = (sSim + sSim.T)/2
    return sSim

def SimFullyConnectedGraph(X, sigma):
    n = X.shape[0]
    dist = lambda x, y, sig: np.exp(-np.linalg.norm(x-y,2)**2/2/sig**2)
    sSim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sSim[i, j] = dist(X[i,:], X[j,:], sigma)
    return sSim.reshape((n,n))-np.diag(np.ones(n))


#eigenvalues, eigenvector = eigs(Sim,k=k,which="SM")

def getDW(Sim):
    
    W = Sim - sparse.diags(Sim.diagonal(0))
    D = sparse.diags(np.array(Sim.sum(axis=1)).reshape(-1))
    return D,W

def getL(D,W):
    return D-W


def getLrw(D,W):
    n = D.shape[0]
    Dinv = D.power(-1)
    I = sparse.diags(np.ones(n))
    
    P = Dinv @ W
    return I - P

def getLsym(D,W):
    n = D.shape[0]
    D_ = D.power(-1/2)
    I = sparse.diags(np.ones(n))
    
    return I - D_ @ W @ D_

def Kmeans(k, Data, inertia= False): 
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Data)
    Labels = kmeans.labels_
    A = {}
    for i in range(k):
        A[i] = np.where(Labels==i)[0]
    #print(Labels.shape)
    if inertia:
        return A, kmeans.inertia_
    else:
        return A

def NormalizedSpectralClustering(Sim,k, inertia = False):
    D,W = getDW(Sim)
    Lrw = getLrw(D,W)
    
    eigvalues, eigvectors = eigs(Lrw,k=k,which="SM")
    U = np.real(eigvectors)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    Labels = kmeans.labels_
    A = {}
    for i in range(k):
        A[i] = np.where(Labels==i)[0]
        
    if inertia:
        return A, kmeans.inertia_
    else:
        return A

def NormalizedNGSpectralClustering(Sim,k, inertia = False):
    D,W = getDW(Sim)
    Lsym = getLsym(D,W)
    
    eigvalues, eigvectors = eigs(Lsym,k=k,which="SM")
    U = np.real(eigvectors)
    U = (U.T / np.linalg.norm(U, axis =1)).T
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    Labels = kmeans.labels_
    A = {}
    for i in range(k):
        A[i] = np.where(Labels==i)[0]
    if inertia:
        return A, kmeans.inertia_
    else:
        return A

def UnormalizedSpectralClustering(Sim,k, inertia= False):
    D,W = getDW(Sim)
    L = getL(D,W)
    eigvalues, eigvectors = eigs(L,k=k,which="SM")
    #U = np.real(eigvectors)
    U = np.real_if_close(eigvectors)
    #eigenvalues = np.real_if_close(eigenvalues)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    Labels = kmeans.labels_
    A = {}
    for i in range(k):
        A[i] = np.where(Labels==i)[0]
        
    if inertia:
        return A, kmeans.inertia_
    else:
        return A
    

    
if __name__ == "__main__":
    #Generate easy data to test n point in dimension d
    #put it un 
    n = 100
    d = 2
    X = np.empty((n,d))

    # create artificial data 
    #nb item per closter m
    m=25
    I = [2,4,6,8]
    sigma = 0.244
    for i in range(len(I)):
        X[m*i:m*(i+1),:] =  st.norm(I[i],sigma).rvs(size=m*d).reshape((m,2))
    
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    
    sigma = 1/2
    Sim = SimFullGraph(X, sigma, treshold=5*1e-5)
    #eps = 0.75
    #Sim = SimEpsNeighborGraphs(X, eps)
    #k = 20
    #Sim = SimMutualKNearestNeighborGraphs(X, k)
    #Sim = SimKNearestNeighborGraphs(X, k)
    a = Sim.toarray()
    k = 4
    A = NormalizedNGSpectralClustering(Sim,k)    
    #k = 4    
    #A = Kmeans(k,X)
    for i in range(k):
        plt.scatter( X[A[i],0], X[A[i],1] )
        #print(A[i])
    
    