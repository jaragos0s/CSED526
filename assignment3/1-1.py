import numpy as np
import pandas as pd
import scipy.stats

def myNB(X, Y, X_test, Y_test):    

    x0 = [] # To store '0' labeled data
    x1 = [] # To store '1' labeled data
    
    for i in range(len(Y)):
        if int(Y[i]) == 0: 
            x0.append(X[i]) # To store 0 labeled x data
        elif int(Y[i]) == 1:
            x1.append(X[i]) # To store 1 labeled x data

    x0 = np.array(x0)
    x1 = np.array(x1)
    
    mu0 = np.mean(x0, axis=0) # mean of x0
    mu1 = np.mean(x1, axis=0) # mean of x1
    
    sigma00 = np.var(x0[:, 0]) * len(x0) / (len(x0) - 1)
    sigma01 = np.var(x0[:, 1]) * len(x0) / (len(x0) - 1)
    sigma10 = np.var(x1[:, 0]) * len(x1) / (len(x1) - 1)
    sigma11 = np.var(x1[:, 1]) * len(x1) / (len(x1) - 1)
    # Gaussian distribution on the data
    pdf00 = scipy.stats.multivariate_normal(mu0[0], sigma00).pdf(X_test[:, 0])
    pdf01 = scipy.stats.multivariate_normal(mu0[1], sigma01).pdf(X_test[:, 1])
    pdf10 = scipy.stats.multivariate_normal(mu1[0], sigma10).pdf(X_test[:, 0])
    pdf11 = scipy.stats.multivariate_normal(mu1[1], sigma11).pdf(X_test[:, 1])
    
    # To calculate prior0 for label 0 and prior1 for label 1
    prior0 = pdf00 * pdf01 * len(x0) / len(Y)
    prior1 = pdf10 * pdf11 * len(x0) / len(Y)
    prior = prior0 + prior1
    
    pred = np.argmax([prior0, prior1], axis=0)
    posterior = np.array([prior0 / prior, prior1 / prior]).T
    err = np.mean(pred != Y_test)
    
    return pred, posterior, err

def Testing(Y, pred):
    tp = np.sum((Y == 1) * pred)
    fp = np.sum((Y == 0) * pred)
    tn = np.sum((Y == 0) * pred)
    fn = np.sum((Y == 1) * pred)
    
    accuracy = (Y==pred).mean()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    confusion_matrix = np.array([[tp, fn],[fp, tn]])
    
    return accuracy, precision, recall, confusion_matrix
