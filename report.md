# MDL Assignment 1 Report
## Team 48:
### Abhijeeth Singam, Saravanan Senthil

---

## **Task 1:** 
### LinearRegression().fit()
Given test data, LinearRegression().fit() minimizes the square difference between the model and the data to find the weights and bias of a regression that yield the most accurate results. In the case of a simple regression, this happens to be the 'line of best fit'.

![multivariable_reg](./imgs/multivariable_reg.png)

To calculate the aforementioned 'accuracy' of a set of weights and bias, the MSE (Mean Squared Error) is used. LinearRegression().fit() aims to reduce this MSE as much as possible to result in the most accurate set of weights and bias that it can acheive.

---

## **Task 2:**

### Calculating Bias and Variance
To calculate bias and variance, we used the following formulae/codes:  

Bias is one of two ways of measuring the accuracy of an ML model. It represents the difference between the 'expected value' or 'average value' of the model and the actual value we are trying to predict.

Bias is calculated by taking the square root of Bias^2 using the formula:  

![Bias2_formula](./imgs/bias.png)

Code:

```python
bias2 = np.mean( (np.mean(predMatrix, axis = 0) - testData[:, 1] ) ** 2 )
bias = np.sqrt(bias2)
```
where predMatrix is the collection of y_predict from the trained models for each degree

Variance is the other way in which the accuracy of an ML model is measured. It represent the 'variability' of the modell's prediction, i.e. how much the predicted values vary for different realizations of that model.

Variance is calculated using the formula: 

![Variance_formula](./imgs/variance.png)  

Code:  
```python
variance = np.mean(np.var(predMatrix, axis = 0))
```
where predMatrix is the collection of y_predict from the trained models for each degree
Where np.var is numpy's buitin to function compute variance

---

## **Task 3:**
### Calculating Irreducible Error

Irreducible error is a measure of the 'noise' in the supplied data. It is referred to as 'irreducible' as this error arises from the data and not the model and thus cannot be reduced no matter how good the created model is.

To calculate irreducible error, we used the formula:  

![Irreducible_formula](./imgs/irredErr.png)

---

## **Task 4:**
### Plotting Bias^2 - Variance graph

![bias-variance-totalerror](./imgs/bvtgraph.png)

### Understanding the graph:

In this graph we display three different values: bias^2, variance and total error. Total error represents the total

### Tabulating the results:

|   degree |      bias |   variance |
|---------:|----------:|-----------:|
|        1 | 1001.39   |    22550.4 |
|        2 |  978.315  |    37427.8 |
|        3 |   93.0622 |    40282.1 |
|        4 |   89.8518 |    44151   |
|        5 |   87.3383 |    49296.9 |
|        6 |   86.417  |    59239.7 |
|        7 |   94.5778 |    89097.7 |
|        8 |  100.216  |   100876   |
|        9 |   95.924  |   124155   |
|       10 |  107.308  |   134876   |
|       11 |   99.785  |   142946   |
|       12 |  152.416  |   150638   |
|       13 |  117.411  |   153307   |
|       14 |  183.001  |   147355   |
|       15 |  241.782  |   146984   |
|       16 |  260.298  |   151108   |
|       17 |  337.601  |   150028   |
|       18 |  351.453  |   155541   |
|       19 |  440.471  |   155380   |
|       20 |  451.163  |   161849   |

---