#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:29:21 2022

@author: darktanuki
"""

import scipy.stats as st 
import pandas      as pd 
import numpy       as np
import matplotlib.pyplot as plt
import sys
sys.path.append( '/Users/darktanuki/Desktop/Statistical machine learning/' )
from SpectralClustering import *
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as st
import networkx as nx
#Create an artificial dataset 
# Moon_set 
np.random.seed(193029)

def Moon_set(n,mintheta, maxtheta, center, sigma = 0.1):
    theta = (maxtheta-mintheta)*st.uniform.rvs(size=n) + mintheta
    #theta = np.linspace(  mintheta ,maxtheta , n )
    radius = 1
    a = radius * np.cos( theta ) + center[0]
    b = radius * np.sin( theta ) + center[1]
    U = st.norm(scale=sigma).rvs(size=n*2).reshape((n,2))
    DataMoon = np.c_[a,b] + U
    return DataMoon

def Cirlce_set(n,r, center, sigma = 0.1):
    theta = (2*np.pi)*st.uniform.rvs(size=n) 
    #theta = np.linspace(  mintheta ,maxtheta , n )
    radius = r
    a = radius * np.cos( theta ) + center[0]
    b = radius * np.sin( theta ) + center[1]
    U = st.norm(scale=sigma).rvs(size=n*2).reshape((n,2))
    DataCircle = np.c_[a,b] + U
    return DataCircle


def Gauss_set(n,center, sigma):
    return st.multivariate_normal(mean = center, cov=sigma).rvs(size=n)

def Shittyform():
    n1 = 35
    n2 = 100
    n3 = 40
    DataMoon1 = Moon_set(n1,0, np.pi, [0,0])
    DataMoon2 = Moon_set(n2,-np.pi, 0, [4.5/5,0])
    DataGauss = Gauss_set(n3,[0,-2.5], np.array([[0.6,0],[0, 0.25]]))
    Data = np.vstack((DataMoon1, DataMoon2, DataGauss))
    
    #plt.scatter(DataMoon1[:,0], DataMoon1[:,1])
    #plt.scatter(DataMoon2[:,0], DataMoon2[:,1])
    #plt.scatter(DataGauss[:,0], DataGauss[:,1])
    return Data

def Generation2Circles():
    n = 250
    r0=1
    r1=2
    DataCircle0 = Cirlce_set(n,r0, [0,0], sigma = 0.1)
    DataCircle1 = Cirlce_set(n,r1, [0,0], sigma = 0.1)
    Data = np.vstack((DataCircle0, DataCircle1))
    return Data

def GenerationDataZ():
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)


    simple_datasets = [
    (noisy_circles, {'name': 'Noisy Circles','n_clusters': 2}),
    (noisy_moons, {'name': 'Noisy Moons', 'n_clusters': 2}),
    (varied, {'name': 'Blobs with varied variances','n_clusters': 3}),
    (aniso, {'name': 'Anisotropic data', 'n_clusters': 3}),
    (blobs, {'name': 'Blobs', 'n_clusters': 3}),
    (no_structure, {'name': 'No structure', 'n_clusters': 3})]
    
    plt.figure(figsize=(9 * 2 + 3, 3))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    plot_num = 1
    for i_dataset, (dataset, dataset_params) in enumerate(simple_datasets):
        
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
        
        name = dataset_params['name']
        plt.subplot(1, len(simple_datasets), plot_num)
        plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], s=10) #c=y)#, cmap='Set1')
    
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        
        plot_num += 1

def GenerationData():
    n = 500
    r0=1
    r1=2
    DataCircle0 = Cirlce_set(n,r0, [0,0], sigma = 0.07)
    DataCircle1 = Cirlce_set(n,r1, [0,0], sigma = 0.07)
    Data0 = np.vstack((DataCircle0, DataCircle1))
    
    n1=750;n2=750
    sigma= 0.06
    DataMoon0 = Moon_set(n1,0, np.pi, [0,0], sigma =sigma)
    DataMoon1 = Moon_set(n2,-np.pi, 0, [4.5/5,0.5], sigma =sigma)
    Data1 = np.vstack((DataMoon0, DataMoon1))
    
    n=500
    Data_Gauss0 = Gauss_set(n,[-2,-1], sigma=0.1)
    Data_Gauss1 = Gauss_set(700,[2,1], sigma=0.1)
    Data_Gauss2 = Gauss_set(n,[0,0], sigma=np.array([[0.7,0],[0, 1]]))
    Data2 = np.vstack((Data_Gauss0, Data_Gauss1, Data_Gauss2))
    
    n0=500
    theta = np.pi/3; a=5;b=1
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    scale = np.array([[a,0],[0,b]])
    Data_Gauss0 = Gauss_set(n,[0.5,-1], sigma=0.08)
    Data_Gauss1 = Gauss_set(700,[2,0.5], sigma=0.08)
    Data_Gauss2 = Gauss_set(n,[0.5,1], sigma=0.08)
    Data3 = np.vstack((Data_Gauss0, Data_Gauss1, Data_Gauss2))
    #Data3,_= datasets.make_blobs(n_samples=n,random_state=175, centers=3)
    #Data = np.vstack((Data_Gauss0, Data_Gauss1, Data_Gauss2))
    Data3 = Data3 @ scale @ rot 
    
    n0=500
    theta = np.pi/4; a=3;b=1
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    scale = np.array([[a,0],[0,b]])
    Data4,_= datasets.make_blobs(n_samples=n,random_state=175, centers=3)
    #Data = np.vstack((Data_Gauss0, Data_Gauss1, Data_Gauss2))
    
    n=1000
    Data5 = st.uniform.rvs(size=n*2).reshape((n,2))
    
    Data6 = Shittyform()
    #plt.scatter(Data[:,0], Data[:,1], s=5)
    return Data0,Data1,Data2,Data3,Data4,Data5, Data6

def plottingDifferentSim(Data):
    
    plt.scatter(Data[:,0], Data[:,1], marker='*')
    plt.title("Data points")
    plt.axis('off')
    plt.savefig("Img/DataPointsMoon.png")
    plt.show()
    posi = {}
    for i in range(len(Data)):
        posi[i]= Data[i,:]
        
    eps = 0.4
    Sim = SimEpsNeighborGraphs(Data, eps)
    G = nx.from_scipy_sparse_matrix(Sim)
    print(f"SimEpsNeighborGraphs with eps={eps}")
    nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color='g')
    plt.legend([f'eps={eps}'], loc="upper left")
    plt.savefig("Img/SimEpsNeighbor.png")
    plt.show()

    k=6
    Sim = SimKNearestNeighborGraphs(Data, k)
    G = nx.from_scipy_sparse_matrix(Sim)
    print(f"SimKNearestNeighborGraphs with k={k}")
    nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color='g',)
    plt.legend([f'k={k}'], loc="upper left")
    plt.savefig("Img/SimKNeighbor.png")
    plt.show()
    
    k=6
    Sim = SimMutualKNearestNeighborGraphs(Data, k)
    G = nx.from_scipy_sparse_matrix(Sim)
    print(f"SimMutualKNearestNeighborGraphs with k={k}")
    nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color='g')
    plt.legend([f'k={k}'], loc="upper left")
    plt.savefig("Img/SimMutualKNeighbor.png")
    plt.show()    
    
    # sig = 0.8
    # Sim = SimFullyConnectedGraph(Data, sig)
    # G = nx.from_numpy_matrix(Sim, parallel_edges=False)
    # print(f"SimFullyConnectedGraph with k={k}")
    # weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
    # nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color=weights, edge_cmap=plt.cm.Greens)
    # plt.legend([f'sigma={sig}'], loc="upper left")
    # plt.savefig("Img/SimFullConnected.png")
    # plt.show()
    
# =============================================================================
#     sigma = 100
#     Sim = SimFullGraph(Data, sigma, treshold=5*1e-5)
#     G = nx.from_scipy_sparse_matrix(Sim)
#     print(f"SimFullGraph with sigma={sigma}")
#     nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color='g')
#     plt.show()
# =============================================================================


def plottingCluster(Data):
    #A = Kmeans(k,Data)
    for i in range(len(A.keys())):
        plt.scatter( Data[A[i],0], Data[A[i],1] )
    #plt.title("Kmeans with k=3")
    plt.show()
    
    Sim = SimMutualKNearestNeighborGraphs(Data, 10)
    k = 3    
    A = NormalizedSpectralClustering(Sim,k)
    for i in range(k):
        plt.scatter( Data[A[i],0], Data[A[i],1] )
    plt.title("NormalizedSpectralClustering  with SimMutualKNearestNeighborGraphs with k=3")
    plt.show()
    
    
    Sim = SimMutualKNearestNeighborGraphs(Data, 10)
    k = 3    
    A = NormalizedNGSpectralClustering(Sim,k)
    for i in range(k):
        plt.scatter( Data[A[i],0], Data[A[i],1] )
    plt.title("NormalizedNGSpectralClustering  with SimMutualKNearestNeighborGraphs with k=3")
    plt.show()
    
    
    Sim = SimMutualKNearestNeighborGraphs(Data, 10)
    k = 3    
    A = UnormalizedSpectralClustering(Sim,k)
    for i in range(k):
        plt.scatter( Data[A[i],0], Data[A[i],1] )
    plt.title("UnormalizedSpectralClustering  with SimMutualKNearestNeighborGraphs with k=3")
    plt.show()
    
    

if __name__ == "__main__":
    BoolPlottingSimGraph = True
    BoolPlottingClusters = False
    BoolPlottingData = False
    #BoolPlottingCluster = False #useless
    
    
    #plt.scatter(Data[:,0], Data[:,1])
    if BoolPlottingSimGraph == True:
        Data = Shittyform()
        plottingDifferentSim(Data)
        
    #if BoolPlottingCluster == True:
    #    Data = Shittyform()
    #    plottingCluster(Data)
    
    if BoolPlottingData == True:
        
        Data0,Data1,Data2,Data3,Data4,Data5, Data6 = GenerationData()
        Data = {0:Data0, 1:Data1, 2:Data2, 3:Data3, 4:Data4, 5:Data5, 6:Data6}
        print("--Generation Data done")
        
        plt.figure(figsize=(9 * 2 + 3, 3))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

        for i in range(7):
            
            if i==6:
                plt.subplot(1,7,(i+1))
                plt.scatter(Data[i][:,0], Data[i][:,1])
            else:
                plt.subplot(1,7,(i+1))
                plt.scatter(Data[i][:,0], Data[i][:,1], s=1)
        plt.show()
        
    if BoolPlottingClusters == True:
        Data0,Data1,Data2,Data3,Data4,Data5, Data6 = GenerationData()
        Data = {0:Data0,
                1:Data1,
                2:Data2,
                3:Data3,
                4:Data4,
                5:Data5,
                6:Data6}
        print("--Generation Data done")
        Choice= {0:[2,SimKNearestNeighborGraphs,12],
                 1:[2,SimKNearestNeighborGraphs,12],
                 2:[3,SimKNearestNeighborGraphs,12],
                 3:[3,SimKNearestNeighborGraphs,12],
                 4:[3,SimKNearestNeighborGraphs,12],
                 5:[3,SimKNearestNeighborGraphs,12],
                 6:[3,SimMutualKNearestNeighborGraphs,12]}
        n=7 # number of Data
        
        clustering_algorithms = {
        0: Kmeans,
        1: UnormalizedSpectralClustering,
        2: NormalizedNGSpectralClustering,
        3: NormalizedSpectralClustering}
        m=4 # number of method we tested
        

        
        for i in range(n): #n Data
            
            plt.figure(figsize=(9 * 2 + 3, 3))
            plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
            
            Sim = Choice[i][1](Data[i], Choice[i][2])
            print(f"--Sim for Data{i}/{n} done")
            
            for j in range(m): # Cluster method
                plt.subplot(1,m,j+1)
                if j==0:
                    
                    A = clustering_algorithms[j](Choice[i][0], Data[i])
                    print(f"--Clustering{j+1}/{m} done")
                else:
                    A = clustering_algorithms[j](Sim, Choice[i][0])
                    print(f"--Clustering{j+1}/{m} done")
            
                for k in range(len(A.keys())):
                    plt.scatter( Data[i][A[k],0], Data[i][A[k],1] )
            
    
   
    
    
    
    
    