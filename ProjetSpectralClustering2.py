#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:19:18 2022

@author: darktanuki
"""

import sys
path = '/Users/darktanuki/Desktop/Statistical machine learning/'
sys.path.append( '/Users/darktanuki/Desktop/Statistical machine learning/' )
from SpectralClustering import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import pandas as pd






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
    n = 800
    r0=1
    r1=2
    DataCircle0 = Cirlce_set(n,r0, [0,0], sigma = 0.07)
    DataCircle1 = Cirlce_set(n,r1, [0,0], sigma = 0.07)
    Data = np.vstack((DataCircle0, DataCircle1))
    return Data

def plottingCluster(K,df, SimGraphfunc, SpectralClusteringMethod, Sim=None):
    Data = df.to_numpy()
    if Sim == None:
        Sim = SimGraphfunc(Data)
        print('--Sim calculate')
    else:
        pass
    for k in K:    
        A = SpectralClusteringMethod(Sim,k)
        #print(A)
        for i in range(k):
            plt.scatter( Data[A[i],0], Data[A[i],1] )
        plt.title(f"SpectralClusteringMethod with k={k}")
        plt.show()


def ComparedA0todA1(k, Sim, A0, A1, idx0, idx1):
    Q = np.zeros((k,k))
    #Comparaison from 0 to 1 
    for k0 in range(k):
        
        for i in A0[k0]:
            #Calculate an array of probability to be in the jth cluser
            iSim = Sim[idx0[i],:]
            nz = iSim.nonzero()
            
            ClosestPointSet1 = np.intersect1d(nz[1], idx1, assume_unique=True, return_indices=False)
            #SimClosestPoint  = np.array(iSim[0,ClosestPointSet1].todense())[0]
            
            b = np.zeros(k)
            for k1 in range(k):
                ClosestPointSet1Clusterk1 = np.intersect1d(idx1[A1[k1]], ClosestPointSet1, assume_unique=True, return_indices=False) 
                b[k1] = np.sum(np.array(iSim[0,ClosestPointSet1Clusterk1].todense())[0])
            if np.sum(b)!=0:
                Q[k0,:] += b / np.sum(b)
        Q[k0,:] /= len(A0[k0])
    
    return Q

def Choosek(K, rep, df, SimGraphfunc, SpectralClusteringMethod,Sim=None):
    # SimGraph has to be a func with only the data has the variable 
    #no hpyerparameters
    
    X = np.zeros((len(K),2))
    X[:,0] = np.array(K)
    if Sim == None:
        Sim = SimGraphfunc(df.to_numpy())
        print("--Sim done")
    else:
        pass
    
    n = Sim.shape[0]
    arrn  = np.array(range(n))
    
    for i in range(rep):
        print(f"rep={i+1}/{rep}")
        np.random.shuffle(arrn)
        idx0 = arrn[:int(n/2)]
        idx1 = arrn[int(n/2):]
        Sim0 = Sim[np.ix_(idx0, idx0)]
        Sim1 = Sim[np.ix_(idx1, idx1)]
        for j in range(len(K)):
            print(f"k={K[j]}")
            A0 = SpectralClusteringMethod(Sim0,K[j])
            A1 = SpectralClusteringMethod(Sim1,K[j])
            

            for i in range(K[j]):
                plt.scatter( df.to_numpy()[idx0[A0[i]],0], df.to_numpy()[idx0[A0[i]],1] )
            plt.show()
            for i in range(K[j]):
                plt.scatter( df.to_numpy()[idx1[A1[i]],0], df.to_numpy()[idx1[A1[i]],1] )
            plt.show()
                
            Q0 = ComparedA0todA1(K[j], Sim, A0, A1, idx0, idx1)
            Q1 = ComparedA0todA1(K[j], Sim, A1, A0, idx1, idx0)
            
            minprmax0 = (Q0.max(axis=1)).min()
            minprmax1 = (Q1.max(axis=1)).min()
    
            X[j,1] +=np.min(np.c_[minprmax0,minprmax1])
            #print(Q0)
            #print(Q1)
    X[:,1]/=rep                   
    return X
                
        
    
def plotClusterR2(df, SimGraphfunc, SpectralClusteringMethod, K=[9,10], Sim= None, embedding = None):
    if type(embedding) == type(None):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(df)
        print('--Embedding done')
    else:
        pass
    
    if Sim == None:
        Sim = SimGraphfunc(df.to_numpy())
        print('--Sim calculate')
    else:
        pass
    
    
    for k in K:
        A = SpectralClusteringMethod(Sim,k)
        print('--Cluster done')
        for i in range(k):
            plt.scatter( embedding[A[i],0], embedding[A[i],1], s=0.3 )
        plt.title(f'k={k}')
        plt.show()
    
    
    
def ElbowInertia(K,df, SimGraphfunc, SpectralClusteringMethod, Sim=None):
    if Sim == None:
        Sim = SimGraphfunc(df.to_numpy())
        print("--Sim done")
    else:
        pass
    
    elbow= np.empty((len(K),2))
    elbow[:,0] = np.array(K)
    for i in range(len(K)):
        print(f"k={K[i]}")
        _ , elbow[i,1]= SpectralClusteringMethod(Sim,K[i], inertia= True)
        #_ , elbow[i,1]= SpectralClusteringMethod(K[i],df.to_numpy(), inertia= True)
    return elbow

        
        
    
    
    
    
    
    
run=0
BoolTrueData = False
if (__name__ == "__main__")*(run==0):
    if BoolTrueData == True:
        df = pd.read_csv(path+"DataFrame.csv")
        del df['CUST_ID']
        df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
        df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
        cols = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
        for col in cols:
            df[col] = np.log(1 + df[col])
        for column in df.columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
            
        Data = df
        SimGraphfunc = lambda X : SimMutualKNearestNeighborGraphs(X,30)
        SpectralClusteringMethod = NormalizedSpectralClustering
        
        
        
    else:
        Data = Shittyform()
        Data = pd.DataFrame(Data)
        #SimGraphfunc = lambda X : SimMutualKNearestNeighborGraphs(X,12)
        SimGraphfunc = lambda X : SimKNearestNeighborGraphs(X,20)
        
        SpectralClusteringMethod = NormalizedSpectralClustering
        
        
        #plt.scatter( Data.to_numpy()[:,0], Data.to_numpy()[:,1] )
        #plt.show()
    
    
    
    Sim = SimGraphfunc(Data.to_numpy())
    print('--Sim done')
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(Data)
    print('--embedding done')

if __name__ == "__main__":
    
    BoolplotClusterR2 = False
    BoolChoosek = True
    BoolElbowInertia = False

    
    if BoolplotClusterR2 == True:
        
        plotClusterR2(Data, SimGraphfunc, SpectralClusteringMethod, K=[9,10,11], Sim=Sim, embedding=embedding)
        
    if BoolChoosek == True:
        
        AA = Choosek([1,2,3,4,5], 1, Data, SimGraphfunc, SpectralClusteringMethod, Sim=Sim)
        print(AA)
    if BoolElbowInertia == True:
        elbow = ElbowInertia([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], Data,SimGraphfunc, SpectralClusteringMethod, Sim=Sim)
        plt.plot(elbow[:,0], elbow[:,1])
        plt.title('ElbowInertia')
        plt.show()
    