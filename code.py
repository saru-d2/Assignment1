# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle 


# %%
with open('./data/test.pkl', 'rb') as fd:
    unpickledTest = pickle.load(fd)

with open('./data/train.pkl', 'rb') as fd:
    unpickledTrain = pickle.load(fd)


# %%
#partition Data
testData = unpickledTest

np.random.shuffle(unpickledTrain)
trainData = np.split(unpickledTrain, 10)


# %%
def train(data, degree):
    poly = PolynomialFeatures(degree = degree)
    X = poly.fit_transform(data[:, 0].reshape(-1, 1))
    y = data[:, 1].reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, y)

    return reg


# %%
def bias2_calc(testPred, testData):
    num = len(testData)
    y = testData[:, 1]
    tot1 = 0
    meanTestPred = np.mean(testPred[:,0])
    # eFCap = tot1/ num
#     print(abs(meanTestPred - y))
    return np.mean((meanTestPred - y)**2)

# def bias2_calc(testPred, testData):
#     num = len(testData)
#     y = testData[:, 1]
#     tot1 = 0
#     for i in range(0, num):
#         tot1 += (testPred[i] - y[i]) ** 2
#     return tot1/num


# def variance_calc(testPred, testData):
#     num = len(testData)
#     y = testData[:, 1]
#     tot1 = 0
#     for i in range(0, num):
#         tot1 += testPred[i]
#     eFCap = tot1/num
#     tot2 = 0
#     for i in range(0, num):
#         tot2 += (testPred[i] - eFCap)**2
#     return tot2/num

def variance_calc(testPred, testData):
    var = np.mean((testPred[:,0] - np.mean(testPred[:,0]))**2)
    return var


# %%
bias2Arr = [0] * 20
varianceArr = [0] * 20
for deg in range(1, 20): #should be 1, 20
    for i in range(0, 3): #should be 0, 10
        ret = train(trainData[i], deg)
        plt.scatter(testData[:, 0], testData[:, 1])
        X = PolynomialFeatures(degree = deg).fit_transform(testData[:, 0].reshape(-1, 1))
        testPred = ret.predict( X )
        plt.scatter(testData[:, 0], testPred)
        plt.title('degree ' + str(deg) + ', i: ' + str(i))
        plt.show()
        print('bias^2: ' + str(bias2_calc(testPred, testData)) + ' variance: ' + str(variance_calc(testPred, testData)))
        bias2Arr[deg] += bias2_calc(testPred, testData)
        varianceArr[deg] += variance_calc(testPred, testData)
        


# %%
# plt.plot(bias2Arr, label='bias^2')
plt.plot( range(1,20), varianceArr[1:], label='variance')
plt.legend()
plt.show()

plt.plot(range(1,20), bias2Arr[1:], label='bias^2')
# plt.plot(varianceArr, label='variance')
plt.legend()
plt.show()


# %%



