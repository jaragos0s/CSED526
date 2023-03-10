def myNBA(X, Y, X_test, Y_test):    
    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    x0 = [] # To store 'C' labeled data
    x1 = [] # To store 'F' labeled data
    x2 = [] # To store 'F-C' labeled data
    x3 = [] # To store 'G' labeled data
    x4 = [] # To store 'G-F' labeled data
    for i in range(len(Y)):
        if Y[i] == 'C':
            x0.append(X[i])
        elif Y[i] == 'F':
            x1.append(X[i])
        elif Y[i] == 'F-C':
            x2.append(X[i])
        elif Y[i] == 'G':
            x3.append(X[i])
        elif Y[i] == 'G-F':
            x4.append(X[i])

    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    
    mu0 = np.mean(x0, axis=0) # mean of x0
    mu1 = np.mean(x1, axis=0) # mean of x1
    mu2 = np.mean(x2, axis=0) # mean of x2
    mu3 = np.mean(x3, axis=0) # mean of x3
    mu4 = np.mean(x4, axis=0) # mean of x4

    sigma00 = np.var(x0[:, 0]) * len(x0) / (len(x0) - 1)
    sigma01 = np.var(x0[:, 1]) * len(x0) / (len(x0) - 1)
    sigma02 = np.var(x0[:, 2]) * len(x0) / (len(x0) - 1)
    sigma03 = np.var(x0[:, 3]) * len(x0) / (len(x0) - 1)
    sigma04 = np.var(x0[:, 4]) * len(x0) / (len(x0) - 1)
    sigma05 = np.var(x0[:, 5]) * len(x0) / (len(x0) - 1)
    sigma06 = np.var(x0[:, 6]) * len(x0) / (len(x0) - 1)
    sigma07 = np.var(x0[:, 7]) * len(x0) / (len(x0) - 1)
    sigma08 = np.var(x0[:, 8]) * len(x0) / (len(x0) - 1)
    
    pdf00 = scipy.stats.multivariate_normal(mu0[0], sigma00).pdf(X_test[:, 0])
    pdf01 = scipy.stats.multivariate_normal(mu0[1], sigma01).pdf(X_test[:, 1])
    pdf02 = scipy.stats.multivariate_normal(mu0[2], sigma02).pdf(X_test[:, 2])
    pdf03 = scipy.stats.multivariate_normal(mu0[3], sigma03).pdf(X_test[:, 3])
    pdf04 = scipy.stats.multivariate_normal(mu0[4], sigma04).pdf(X_test[:, 4])
    pdf05 = scipy.stats.multivariate_normal(mu0[5], sigma05).pdf(X_test[:, 5])
    pdf06 = scipy.stats.multivariate_normal(mu0[6], sigma06).pdf(X_test[:, 6])
    pdf07 = scipy.stats.multivariate_normal(mu0[7], sigma07).pdf(X_test[:, 7])
    pdf08 = scipy.stats.multivariate_normal(mu0[8], sigma08).pdf(X_test[:, 8])
    
    sigma10 = np.var(x1[:, 0]) * len(x1) / (len(x1) - 1)
    sigma11 = np.var(x1[:, 1]) * len(x1) / (len(x1) - 1)
    sigma12 = np.var(x1[:, 2]) * len(x1) / (len(x1) - 1)
    sigma13 = np.var(x1[:, 3]) * len(x1) / (len(x1) - 1)
    sigma14 = np.var(x1[:, 4]) * len(x1) / (len(x1) - 1)
    sigma15 = np.var(x1[:, 5]) * len(x1) / (len(x1) - 1)
    sigma16 = np.var(x1[:, 6]) * len(x1) / (len(x1) - 1)
    sigma17 = np.var(x1[:, 7]) * len(x1) / (len(x1) - 1)
    sigma18 = np.var(x1[:, 8]) * len(x1) / (len(x1) - 1)
    
    pdf10 = scipy.stats.multivariate_normal(mu1[0], sigma10).pdf(X_test[:, 0])
    pdf11 = scipy.stats.multivariate_normal(mu1[1], sigma11).pdf(X_test[:, 1])
    pdf12 = scipy.stats.multivariate_normal(mu1[2], sigma12).pdf(X_test[:, 2])
    pdf13 = scipy.stats.multivariate_normal(mu1[3], sigma13).pdf(X_test[:, 3])
    pdf14 = scipy.stats.multivariate_normal(mu1[4], sigma14).pdf(X_test[:, 4])
    pdf15 = scipy.stats.multivariate_normal(mu1[5], sigma15).pdf(X_test[:, 5])
    pdf16 = scipy.stats.multivariate_normal(mu1[6], sigma16).pdf(X_test[:, 6])
    pdf17 = scipy.stats.multivariate_normal(mu1[7], sigma17).pdf(X_test[:, 7])
    pdf18 = scipy.stats.multivariate_normal(mu1[8], sigma18).pdf(X_test[:, 8])
    
    sigma20 = np.var(x2[:, 0]) * len(x2) / (len(x2) - 1)
    sigma21 = np.var(x2[:, 1]) * len(x2) / (len(x2) - 1)
    sigma22 = np.var(x2[:, 2]) * len(x2) / (len(x2) - 1)
    sigma23 = np.var(x2[:, 3]) * len(x2) / (len(x2) - 1)
    sigma24 = np.var(x2[:, 4]) * len(x2) / (len(x2) - 1)
    sigma25 = np.var(x2[:, 5]) * len(x2) / (len(x2) - 1)
    sigma26 = np.var(x2[:, 6]) * len(x2) / (len(x2) - 1)
    sigma27 = np.var(x2[:, 7]) * len(x2) / (len(x2) - 1)
    sigma28 = np.var(x2[:, 8]) * len(x2) / (len(x2) - 1)
    
    pdf20 = scipy.stats.multivariate_normal(mu2[0], sigma20).pdf(X_test[:, 0])
    pdf21 = scipy.stats.multivariate_normal(mu2[1], sigma21).pdf(X_test[:, 1])
    pdf22 = scipy.stats.multivariate_normal(mu2[2], sigma22).pdf(X_test[:, 2])
    pdf23 = scipy.stats.multivariate_normal(mu2[3], sigma23).pdf(X_test[:, 3])
    pdf24 = scipy.stats.multivariate_normal(mu2[4], sigma24).pdf(X_test[:, 4])
    pdf25 = scipy.stats.multivariate_normal(mu2[5], sigma25).pdf(X_test[:, 5])
    pdf26 = scipy.stats.multivariate_normal(mu2[6], sigma26).pdf(X_test[:, 6])
    pdf27 = scipy.stats.multivariate_normal(mu2[7], sigma27).pdf(X_test[:, 7])
    pdf28 = scipy.stats.multivariate_normal(mu2[8], sigma28).pdf(X_test[:, 8])
    
    sigma30 = np.var(x3[:, 0]) * len(x3) / (len(x3) - 1)
    sigma31 = np.var(x3[:, 1]) * len(x3) / (len(x3) - 1)
    sigma32 = np.var(x3[:, 2]) * len(x3) / (len(x3) - 1)
    sigma33 = np.var(x3[:, 3]) * len(x3) / (len(x3) - 1)
    sigma34 = np.var(x3[:, 4]) * len(x3) / (len(x3) - 1)
    sigma35 = np.var(x3[:, 5]) * len(x3) / (len(x3) - 1)
    sigma36 = np.var(x3[:, 6]) * len(x3) / (len(x3) - 1)
    sigma37 = np.var(x3[:, 7]) * len(x3) / (len(x3) - 1)
    sigma38 = np.var(x3[:, 8]) * len(x3) / (len(x3) - 1)
    
    pdf30 = scipy.stats.multivariate_normal(mu3[0], sigma30).pdf(X_test[:, 0])
    pdf31 = scipy.stats.multivariate_normal(mu3[1], sigma31).pdf(X_test[:, 1])
    pdf32 = scipy.stats.multivariate_normal(mu3[2], sigma32).pdf(X_test[:, 2])
    pdf33 = scipy.stats.multivariate_normal(mu3[3], sigma33).pdf(X_test[:, 3])
    pdf34 = scipy.stats.multivariate_normal(mu3[4], sigma34).pdf(X_test[:, 4])
    pdf35 = scipy.stats.multivariate_normal(mu3[5], sigma35).pdf(X_test[:, 5])
    pdf36 = scipy.stats.multivariate_normal(mu3[6], sigma36).pdf(X_test[:, 6])
    pdf37 = scipy.stats.multivariate_normal(mu3[7], sigma37).pdf(X_test[:, 7])
    pdf38 = scipy.stats.multivariate_normal(mu3[8], sigma38).pdf(X_test[:, 8])
    
    sigma40 = np.var(x4[:, 0]) * len(x4) / (len(x4) - 1)
    sigma41 = np.var(x4[:, 1]) * len(x4) / (len(x4) - 1)
    sigma42 = np.var(x4[:, 2]) * len(x4) / (len(x4) - 1)
    sigma43 = np.var(x4[:, 3]) * len(x4) / (len(x4) - 1)
    sigma44 = np.var(x4[:, 4]) * len(x4) / (len(x4) - 1)
    sigma45 = np.var(x4[:, 5]) * len(x4) / (len(x4) - 1)
    sigma46 = np.var(x4[:, 6]) * len(x4) / (len(x4) - 1)
    sigma47 = np.var(x4[:, 7]) * len(x4) / (len(x4) - 1)
    sigma48 = np.var(x4[:, 8]) * len(x4) / (len(x4) - 1)

    pdf40 = scipy.stats.multivariate_normal(mu4[0], sigma40).pdf(X_test[:, 0])
    pdf41 = scipy.stats.multivariate_normal(mu4[1], sigma41).pdf(X_test[:, 1])
    pdf42 = scipy.stats.multivariate_normal(mu4[2], sigma42).pdf(X_test[:, 2])
    pdf43 = scipy.stats.multivariate_normal(mu4[3], sigma43).pdf(X_test[:, 3])
    pdf44 = scipy.stats.multivariate_normal(mu4[4], sigma44).pdf(X_test[:, 4])
    pdf45 = scipy.stats.multivariate_normal(mu4[5], sigma45).pdf(X_test[:, 5])
    pdf46 = scipy.stats.multivariate_normal(mu4[6], sigma46).pdf(X_test[:, 6])
    pdf47 = scipy.stats.multivariate_normal(mu4[7], sigma47).pdf(X_test[:, 7])
    pdf48 = scipy.stats.multivariate_normal(mu4[8], sigma48).pdf(X_test[:, 8])

    
    prior0 = pdf00 * pdf01 * pdf03 * pdf04 * pdf05 * pdf06 * pdf07 * pdf08 * len(x0) / len(Y)
    prior1 = pdf10 * pdf11 * pdf13 * pdf14 * pdf15 * pdf16 * pdf17 * pdf18 * len(x1) / len(Y)
    prior2 = pdf20 * pdf21 * pdf23 * pdf24 * pdf25 * pdf26 * pdf27 * pdf28 * len(x2) / len(Y)
    prior3 = pdf30 * pdf31 * pdf33 * pdf34 * pdf35 * pdf36 * pdf37 * pdf38 * len(x3) / len(Y)
    prior4 = pdf40 * pdf41 * pdf43 * pdf44 * pdf45 * pdf46 * pdf47 * pdf48 * len(x4) / len(Y)
    prior = prior0 + prior1 + prior2 + prior3 + prior4
    
    pred = np.argmax([prior0, prior1, prior2, prior3, prior4], axis=0)
    pos_dict = {0 : 'C', 1 : 'F', 2 : 'F-C', 3 : 'G', 4 : 'G-F'}
    
    pred_ = []
    for val in pred:
        pred_.append(pos_dict[int(val)])
        
    posterior = np.array([prior0 / prior, prior1 / prior, prior2 / prior, prior3 / prior, prior4 / prior]).T
    err = np.mean(pred != Y_test)
    
    return pred_, posterior, err
