import numpy as np
import pickle 
import matplotlib.pyplot as plot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


with open('./data/train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('./data/test.pkl', 'rb') as f:
    test = pickle.load(f)

x_train = train[:, 0]
y_train = train[:, 1]

x_test = test[:, 0]
y_test = test[:, 1]

x_train=np.array((np.array_split(x_train, 10)))
y_train=np.array((np.array_split(y_train, 10)))

v_table=np.zeros((10,10))
b_table=np.zeros((10,10))

bias_mean=np.zeros((19))
var_mean=np.zeros((19))

for degree in range (1,20):  
    bias_sq=np.zeros((10,80))
    var=np.zeros((10,80))
    out=np.zeros((10,80))
    for i in range (10):   
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        X = poly.fit_transform(x_train[i].reshape(-1, 1))
        X_TEST = poly.fit_transform(x_test.reshape(-1, 1))
        reg = LinearRegression()

        reg.fit(X, y_train[i])
        y_predict = reg.predict(X_TEST)
        
        
        out[i]=y_predict

    point_mean=np.mean(out,axis=0)
    bias_mean[degree-1]=np.mean((point_mean-y_test)**2)

    point_var = np.var(out,axis=0)
    var_mean[degree-1]=np.mean(point_var)


table_bias=pd.DataFrame({'Degree':np.array(range(1,20)),'Bias^2':bias_mean,'Variance': var_mean, 'Variance*100':var_mean[:]*100})
print(table_bias.to_string(index=False))

plot.plot(bias_mean,label='Bias^2', color = 'blue')
plot.plot(var_mean[:],label='Variance * 100', color = 'red')
plot.xlabel('Model Complexity', fontsize='medium')
plot.ylabel('Error', fontsize='medium')
plot.title("Bias vs Variance")
plot.legend()
plot.show()