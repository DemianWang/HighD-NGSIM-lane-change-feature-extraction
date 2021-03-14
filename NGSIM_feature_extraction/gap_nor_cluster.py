#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:15:23 2021

@author: demian
"""

import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
with open('/home/demian/code/python/feature_extration/NGSIM_feature_extraction/gap_normal_action.pkl', "rb") as fp:
    cluster_data_total = pickle.load(fp)
# cluster_data_total=np.delete(cluster_data_total,666,axis=0)
# cluster_data=cluster_data_total[:,:6]

c1=None
c2=None
c3=None

for i in range(cluster_data_total.shape[0]):
    if 0.2<cluster_data_total[i,-1] and 5<cluster_data_total[i,0] and 5<cluster_data_total[i,1] and cluster_data_total[i,-1]<0.8:
        if c1 is None:
            c1=cluster_data_total[i,:]
        else:
            c1=np.vstack((c1,cluster_data_total[i,:]))
    elif 0.2>=cluster_data_total[i,-1] or cluster_data_total[i,1]<=5:
        if c2 is None:
            c2=cluster_data_total[i,:]
        else:
            c2=np.vstack((c2,cluster_data_total[i,:]))        
    elif 0.8<=cluster_data_total[i,-1] or cluster_data_total[i,0]<=5:
        if c3 is None:
            c3=cluster_data_total[i,:]
        else:
            c3=np.vstack((c3,cluster_data_total[i,:]))       

ACTION_EGO_PRE_S_DISTANCE="action_ego_preceding_S_distance"
ACTION_EGO_LTCP_S_DISTANCE="action_ego_LTCpreceding_S_distance"
ACTION_EGO_LTCF_S_DISTANCE="action_ego_LTCfollowing_S_distance"
ACTION_GAP="action_gap"
ACTION_EGO_IN_GAP_NORMALIZATION="action_ego_in_gap_normalization"
ACTION_EGO_PRE_S_VELOCITY="action_ego_preceding_S_velocity"
ACTION_EGO_LTCP_S_VELOCITY="action_ego_LTCpreceding_S_velocity"
ACTION_EGO_LTCF_S_VELOCITY="action_ego_LTCfollowing_S_velocity"
ACTION_TTC="action_ttc"
ACTION_LTC_FORWARD_TTC="action_LTC_forward_TTC"
ACTION_LTC_BACKWARD_TTC="action_LTC_backward_TTC"
ACTION_THW="action_thw"
ACTION_LTC_FORWARD_THW="action_LTC_forward_THW"
ACTION_LTC_BACKWARD_THW="action_LTC_backward_THW"

attribute=c3[:,-1]

# attribute=attribute[attribute[:]<30]
# attribute=attribute[-20<attribute[:]]
  
print("total len:", len(attribute))            
print("mean: ",np.mean(attribute))
print("var: ",np.var(attribute))
print("median: ",np.median(attribute))
plt.hist(attribute,bins=60,alpha=0.8,rwidth=0.8)
plt.xlabel(ACTION_EGO_IN_GAP_NORMALIZATION)  
plt.ylabel('Number')  
plt.show()    

# __________________________________________________________________-

# def standardization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     print(mu,sigma)
#     return (data - mu) / sigma

# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     print(np.min(data),_range)
#     return (data - np.min(data)) / _range

# for i in range(6):
#     cluster_data[:,i]=normalization(standardization(cluster_data[:,i]))



# from numpy import *
# import pylab
# import random,math

# def gmm(cluster_data, K_or_centroids):
# # ============================================================  
# # Expectation-Maximization iteration implementation of  
# # Gaussian Mixture Model.  
# #  
# # PX = GMM(X, K_OR_CENTROIDS)  
# # [PX MODEL] = GMM(X, K_OR_CENTROIDS)  
# #  
# #  - X: N-by-D data matrix.  
# #  - K_OR_CENTROIDS: either K indicating the number of  
# #       components or a K-by-D matrix indicating the  
# #       choosing of the initial K centroids.  
# #  
# #  - PX: N-by-K matrix indicating the probability of each  
# #       component generating each point.  
# #  - MODEL: a structure containing the parameters for a GMM:  
# #       MODEL.Miu: a K-by-D matrix.  
# #       MODEL.Sigma: a D-by-D-by-K matrix.  
# #       MODEL.Pi: a 1-by-K vector.  
# # ============================================================          
#     ## Generate Initial Centroids  
#     threshold = 1e-15
#     dataMat = mat(cluster_data)
#     [N, D] = shape(dataMat)
#     K_or_centroids = 2
#     # K_or_centroids可以是一个整数，也可以是k个质心的二维列向量
#     if shape(K_or_centroids)==(): #if K_or_centroid is a 1*1 number  
#         K = K_or_centroids
#         Rn_index = list(range(N))
#         random.shuffle(Rn_index) #random index N samples  
#         centroids = dataMat[Rn_index[0:K], :]; #generate K random centroid  
#     else: # K_or_centroid is a initial K centroid  
#         K = size(K_or_centroids)[0];   
#         centroids = K_or_centroids;  

#     ## initial values  
#     [pMiu,pPi,pSigma] = init_params(dataMat,centroids,K,N,D)      
#     Lprev = -inf #上一次聚类的误差  

#     # EM Algorithm  
#     while True:
#         # Estimation Step  
#         Px = calc_prob(pMiu,pSigma,dataMat,K,N,D)

#         # new value for pGamma(N*k), pGamma(i,k) = Xi由第k个Gaussian生成的概率  
#         # 或者说xi中有pGamma(i,k)是由第k个Gaussian生成的  
#         pGamma = mat(array(Px) * array(tile(pPi, (N, 1))))  #分子 = pi(k) * N(xi | pMiu(k), pSigma(k))  
#         pGamma = pGamma / tile(sum(pGamma, 1), (1, K)) #分母 = pi(j) * N(xi | pMiu(j), pSigma(j))对所有j求和  

#         ## Maximization Step - through Maximize likelihood Estimation  
#         #print 'dtypeddddddddd:',pGamma.dtype
#         Nk = sum(pGamma, 0) #Nk(1*k) = 第k个高斯生成每个样本的概率的和，所有Nk的总和为N。  

#         # update pMiu  
#         pMiu = mat(diag((1/Nk).tolist()[0])) * (pGamma.T) * dataMat #update pMiu through MLE(通过令导数 = 0得到)  
#         pPi = Nk/N

#         # update k个 pSigma  
#         print('kk=',K)
#         for kk in range(K):
#             Xshift = dataMat-tile(pMiu[kk], (N, 1))  

#             Xshift.T * mat(diag(pGamma[:, kk].T.tolist()[0])) *  Xshift / 2

#             pSigma[:, :, kk] = (Xshift.T * mat(diag(pGamma[:, kk].T.tolist()[0])) * Xshift) / Nk[kk]

#         # check for convergence  
#         L = sum(log(Px*(pPi.T)))  
#         if L-Lprev < threshold:
#             break        
#         Lprev = L

#     return Px


# def init_params(X,centroids,K,N,D):  
#     pMiu = centroids #k*D, 即k类的中心点  
#     pPi = zeros([1, K]) #k类GMM所占权重（influence factor）  
#     pSigma = zeros([D, D, K]) #k类GMM的协方差矩阵，每个是D*D的  

#     # 距离矩阵，计算N*K的矩阵（x-pMiu）^2 = x^2+pMiu^2-2*x*Miu  
#     #x^2, N*1的矩阵replicateK列\#pMiu^2，1*K的矩阵replicateN行
#     distmat = tile(sum(power(X,2), 1),(1, K)) + \
#         tile(transpose(sum(power(pMiu,2), 1)),(N, 1)) -  \
#         2*X*transpose(pMiu)
#     labels = distmat.argmin(1) #Return the minimum from each row  

#     # 获取k类的pPi和协方差矩阵
#     for k in range(K):
#         boolList = (labels==k).tolist()
#         indexList = [boolList.index(i) for i in boolList if i==[True]]
#         Xk = X[indexList, :]
#         #print cov(Xk)
#         # 也可以用shape(XK)[0]
#         pPi[0][k] = float(size(Xk, 0))/N
#         pSigma[:, :, k] = cov(transpose(Xk))  

#     return pMiu,pPi,pSigma

# # 计算每个数据由第k类生成的概率矩阵Px
# def calc_prob(pMiu,pSigma,X,K,N,D):
#     # Gaussian posterior probability   
#     # N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))  
#     Px = mat(zeros([N, K]))
#     for k in range(K):
#         Xshift = X-tile(pMiu[k, :],(N, 1)) #X-pMiu  
#         #inv_pSigma = mat(pSigma[:, :, k]).I
#         inv_pSigma = linalg.pinv(mat(pSigma[:, :, k]))

#         tmp = sum(array((Xshift*inv_pSigma)) * array(Xshift), 1) # 这里应变为一列数
#         tmp = mat(tmp).T
#         #print linalg.det(inv_pSigma),'54545'

#         Sigema = linalg.det(mat(inv_pSigma))

#         if Sigema < 0:
#             Sigema=0

#         coef = power((2*(math.pi)),(-D/2)) * sqrt(Sigema)              
#         Px[:, k] = coef * exp(-0.5*tmp)          
#     return Px


# gmm(cluster_data, 2)
# _______________________________________________-
# cluster_data=np.delete(cluster_data,666,axis=0)

# from numpy import *

# def pca(dataMat, topNfeat=999999):
#     meanVals = mean(dataMat, axis=0)
#     DataAdjust = dataMat - meanVals           #减去平均值
#     covMat = cov(DataAdjust, rowvar=0)
#     eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
#     #print eigVals
#     eigValInd = argsort(eigVals)
#     eigValInd = eigValInd[:-(topNfeat+1):-1]   #保留最大的前K个特征值
#     redEigVects = eigVects[:,eigValInd]        #对应的特征向量
#     lowDDataMat = DataAdjust * redEigVects     #将数据转换到低维新空间
#     reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构数据，用于调试
#     return lowDDataMat, reconMat

# pca1=pca(cluster_data,3)

# pca=PCA(n_components=6)
# pca.fit(cluster_data)
# newX=pca.fit_transform(cluster_data)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
# # print(newX)        
# print(pca.components_)
# print(pca.get_params())
# ___________________________________________________!
# # ts=TSNE(n_components=2,learning_rate=50,n_iter=50000,verbose=1,n_iter_without_progress=1000,random_state=8)
# ts=TSNE(n_components=3,learning_rate=200,n_iter=50000,verbose=1,n_iter_without_progress=4000,random_state=4)
# y=ts.fit_transform(cluster_data)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(y[:, 0], y[:, 1], y[:,2])
# plt.show()
# plt.scatter(y[:, 0], y[:, 1],cmap=plt.cm.Spectral)

# colors1 = '#00CED1'
# colors2 = '#DC143C'
# # colors3 = '#FFAA00'

# c1=None
# c2=None
# # c3=None

# cluster1=None
# cluster2=None

# for i in range(y.shape[0]):
#     if y[i, 0]>6:
#         if c1 is None:
#             c1=y[i, :]
#             cluster1=cluster_data_total[i,:]
#         else:
#             c1=np.vstack((c1,y[i, :]))
#             cluster1=np.vstack((cluster1,cluster_data_total[i, :]))
#     elif y[i, 0]<=6:
#         if c2 is None:
#             c2=y[i, :]
#             cluster2=cluster_data_total[i,:]
#         else:
#             c2=np.vstack((c2,y[i, :]))
#             cluster2=np.vstack((cluster2,cluster_data_total[i, :]))
#     # else:
#     #     if c3 is None:
#     #         c3=y[i, :]
#     #     else:
#     #         c3=np.vstack((c3,y[i, :]))

# plt.scatter(c1[:, 0], c1[:, 1], c=colors1, cmap=plt.cm.Spectral)
# plt.scatter(c2[:, 0], c2[:, 1], c=colors2, cmap=plt.cm.Spectral)
# # plt.scatter(c3[:, 0], c3[:, 1], c=colors3, cmap=plt.cm.Spectral)

# ________________________________________________-






# LCMOMENT_EGO_PRE_S_DISTANCE="LCmoment_ego_preceding_S_distance"
# LCMOMENT_EGO_LTCP_S_DISTANCE="LCmoment_ego_LTCpreceding_S_distance"
# LCMOMENT_EGO_LTCF_S_DISTANCE="LCmoment_ego_LTCfollowing_S_distance"
# LCMOMENT_GAP="LCmoment_gap"
# LCMOMENT_EGO_IN_GAP_NORMALIZATION="LCmoment_ego_in_gap_normalization"
# LCMOMENT_EGO_PRE_S_VELOCITY="LCmoment_ego_preceding_S_velocity"
# LCMOMENT_EGO_LTCP_S_VELOCITY="LCmoment_ego_LTCpreceding_S_velocity"
# LCMOMENT_EGO_LTCF_S_VELOCITY="LCmoment_ego_LTCfollowing_S_velocity"
# LCMOMENT_TTC="LCmoment_ttc"
# LCMOMENT_LTC_FORWARD_TTC="LCmoment_LTC_forward_TTC"
# LCMOMENT_LTC_BACKWARD_TTC="LCmoment_LTC_backward_TTC"
# LCMOMENT_THW="LCmoment_thw"
# LCMOMENT_LTC_FORWARD_THW="LCmoment_LTC_forward_THW"
# LCMOMENT_LTC_BACKWARD_THW="LCmoment_LTC_backward_THW"

# attribute=cluster1[:,5]

# # attribute=attribute[attribute[:]<15]
# # attribute=attribute[-5<attribute[:]]
  
# print("total len:", len(attribute))            
# print("mean: ",np.mean(attribute))
# print("var: ",np.var(attribute))
# print("median: ",np.median(attribute))
# plt.hist(attribute,bins=60,alpha=0.8,rwidth=0.8)
# plt.xlabel(LCMOMENT_LTC_BACKWARD_THW)  
# plt.ylabel('Number')  
# plt.show()    
 



