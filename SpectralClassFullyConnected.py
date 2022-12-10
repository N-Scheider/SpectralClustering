#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:44:28 2022

@author: noahscheider
"""

import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


def SimFullyConnectedGraph(X, sigma):
    n = X.shape[0]
    dist = lambda x, y, sig: np.exp(-np.linalg.norm(x-y,2)**2/2/sig**2)
    Sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Sim[i, j] = dist(X[i,:], X[j,:], sigma)
    return Sim.reshape((n,n))-np.diag(np.ones(n))


n = 100
d = 2
X = np.empty((n,d))

# create artificial data 
#nb item per closter m
m=25
I = [2,4,6,8]
sigma = 0.244
np.random.seed(seed=2001)
for i in range(len(I)):
    X[m*i:m*(i+1),:] =  st.norm(I[i],sigma).rvs(size=m*d).reshape((m,2))

plt.scatter(X[:,0], X[:,1], s=10, c='b', marker='*')
plt.title("Data Points")
plt.axis('off')
plt.savefig("Img/DataPointsBlobs.png")
plt.show()

Data = X
posi = {}
for i in range(len(Data)):
    posi[i]= Data[i,:]

sig = 0.8
Sim = SimFullyConnectedGraph(Data, sig)
G = nx.from_numpy_matrix(Sim)
pos = nx.spring_layout(G)
Sim1 = Sim.flatten()
weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
nx.draw(G, pos = posi, node_size=10, node_shape='*', node_color='b', edge_color=weights, edge_cmap=plt.cm.Greens)
plt.legend([f'sigma={sig}'], loc="upper left")
plt.axis('off')
plt.savefig("Img/SimFullConnected.png")
plt.show()

