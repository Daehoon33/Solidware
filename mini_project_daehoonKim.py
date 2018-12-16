# mini_project_daehoonKim.py
# Daehoon Kim / 2018-12-15
# Language: Python 2.7
# This code demonstrates solution for the mini-project task of Solidware
# Make a simple regression model to predict normalized-loss
#-----------------------------------------------------------

import csv
import sys
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

Data_dir = './data/'
Data_file = 'imports-85.data'

# replace_function()
# this functions replace missing value by mean
# inputs:
# data = list of raw data
# num = index of attribute
# outputs:
# x_list = updated list of data
def replace_function(data, num):
    x_list = []
    for i in data:
        if i[num] == '?':
            x_list.append(np.NaN)
        else:
            x_list.append(float(i[num]))
    x_array = np.array(x_list)
    x_array[np.isnan(x_array)] = round(np.nanmean(x_array), 2)
    x_list = x_array.tolist()
    
    return x_list 

# mse_function()
# this function computes the mean-squares error for this model
# inputs:
# n = number of instances
# x = list of varibables
# y = list of target values
# w = list of weights
# outputs:
# mse = scalar
def mse_function(n, x, y, w):
    sum_error = 0
    for i in range(len(x)):
        y_hat = w[0] + (w[1] * x[i])
        error = (y[i] - y_hat) ** 2
        sum_error += error
    mse = sum_error / len(x)
    return mse

# grad_descent()
# this function solves simple linear regression 
# with gradient descent
# inputs:
# n = number of instances
# x = list of vaiables 
# y = list of target values
# w = list of weights 
# a = learning rate
# output:
# w = uploaded list of parameter values 
def grad_descent(n, x, y, w, a):
    for i in range(n):
        y_hat = w[0] + w[1] * x[i] # compute prediction
        error = y[i] - y_hat # compute prediciton error 
        
        # Update weights
        w[0] = w[0] + a * error * 1 * (1.0 / n)
        w[1] = w[1] + a * error * x[i] * (1.0 / n)
    return w

# ------------
# prepare data
# get data from a file
try: 
    # open data file in csv format and read 
    f = open (Data_dir + Data_file, 'rt')
    raw_data0 = csv.reader(f)
    # parse data in csv format
    raw_data = [rec for rec in raw_data0]
    # handle exception
except Exception as x:
    print 'there was an error:' + str(x)
    sys.exit()    

# remove features 'symboling'
raw_data = [rec[1:] for rec in raw_data]
# skip data sampels with missing values in the target
filtered_data = []
for i in raw_data:
    if i[0] != '?':
        filtered_data.append(i)

# chose x variable and target value 
# x variable = horsepower (column 22)
x = replace_function(filtered_data, 20) # replace missing value(?) by mean 
# y = normalized_losses (column 2)
y = [float(i[0]) for i in filtered_data]

# ------------
# run gradient descent to compute the regression equation
# initialise variables for this model 
a = 0.0001
epsilon = 0.0001
n = len(x)
num_iters = 0
converge = False

# initialise weights with 0
w = [0 for i in range(2)]
# compute initial mean-squared error
curr_mse = 0
prev_mse = mse_function(n, x, y, w)

while(not converge):
    # adjust weights using gradient descent
    w = grad_descent(n, x, y, w, a)

    # compute mean-squared error
    curr_mse = mse_function(n, x, y, w)
    num_iters += 1
    
    # print weights and two sse when num_iters == 10000
    if (num_iters % 10000 == 0):
        print ('iteration: %d previous_mse - current_mse: %6f'
              %(num_iters, prev_mse - curr_mse))
    # iterate until error has not change much from previous iteration
    if (prev_mse - curr_mse <= epsilon):
        converge = True
        print ('my regression equation: y = %f + %f * x' % (w[0], w[1]))
        print ('MSE of this model: %f' %curr_mse)
    else:
        prev_mse = curr_mse

#-----------
# use scikit-learn's linear regression model for comparison
lr = linear_model.LinearRegression()
x_target = [[i] for i in x]
y_target = y
lr.fit(x_target, y_target)
print ('scikit regression equation: y = %f + %f * x' % (lr.intercept_, lr.coef_[0]))
print ('MSE of scikit model: %f' % mean_squared_error(y_target, lr.predict(x_target)))

# compute the prediction of my model and scikit-learn model
y_hat = [0 for i in range(n)]
for j in range(n):
    y_hat[j] = w[0] + (w[1] * x[j])
# compute the prediction of scikit-learn's model
y_hat_sci = list(lr.predict(x_target))

# plot raw data and two results for comparision
plt.figure()
plt.plot(x, y, 'bo', markersize=5)
plt.plot(x, y_hat, 'r', linewidth=2)
plt.plot(x, y_hat_sci, 'g', linewidth=2)
# add plot labels and ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('horse_power', fontsize=14)
plt.ylabel('normalised_loss',fontsize=14)
plt.title('iteration: '+ str(num_iters) + ': y= ' + str(w[0]) + ' + ' + str( w[1]) + 'x')
plt.show()
plt.close()