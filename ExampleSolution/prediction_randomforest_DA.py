#!/usr/bin/python
# -*- coding: utf-8 -*-

# SAMPLE SUBMISSION TO THE BIG DATA HACKATHON 13-14 April 2013 'Influencers in a Social Network'
# .... more info on Kaggle and links to go here
#
# written by Ferenc Huszár, PeerIndex

from sklearn import linear_model
from sklearn.metrics import auc_score
import numpy as np
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import scipy

# no transforms
def transform_features(x):
    return np.log(x+1)

def calculatePrediction():
		
        dTr = loadFile('../data/train.csv')
        y_train = dTr[0]
        X_train_A = dTr[1]
        X_train_B = dTr[2]

        dTes = loadFileTest('../data/test.csv')
        X_test_A = dTes[0]
        X_test_B = dTes[1]

        print "train size: {0} {1}".format(X_train_A.shape, X_train_B.shape)
        print "test size: {0} {1}".format(X_test_A.shape, X_test_B.shape)

	#def transform_features(x):
	#    return np.log(1+x)
	
	X_train_minus = transform_features(X_train_A) - transform_features(X_train_B)
	X_train_div = transform_features(X_train_A) / (transform_features(X_train_B) + 1)
	X_train = np.concatenate((X_train_div, X_train_minus),axis=1)

	X_test_minus = transform_features(X_test_A) - transform_features(X_test_B)
	X_test_div = transform_features(X_test_A) / (transform_features(X_test_B) + 1)
	X_test = np.concatenate((X_test_div, X_test_minus),axis=1)
	
        #In this case we'll use a random forest, but this could be any classifier
        cfr = RandomForestClassifier(n_estimators=100, max_features=math.sqrt(X_train.shape[1]), n_jobs=1)

    #Simple K-Fold cross validation. 5 folds.
        cv = cross_validation.KFold(len(X_train), k=10, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
        results = []
        for traincv, testcv in cv:
            probas = cfr.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[testcv])
            p_train = [x[1] for x in probas]
            results.append(auc_score(y_train[testcv].tolist(),p_train))
            #results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
        print "Results: " + str( np.array(results).mean() )

        # Test set prob

        probas = cfr.predict_proba(X_test)
        p_test = [x[1] for x in probas]

	###########################
	# WRITING SUBMISSION FILE
	###########################
	predfile = open('predictions_test.csv','w+')

        print "label size: test - {0} expected {1}".format(len(p_test), X_test_A.shape[0])
	
        for item in p_train:
            print >>predfile, "{0}".format(str(item))
	
	predfile.close()

header = ""

def loadFile(filename):
	
	###########################
	# LOADING TRAINING DATA
	###########################
	
	trainfile = open(filename)
	header = trainfile.next().rstrip().split(',')
	
	y_train = []
	X_train_A = []
	X_train_B = []
	
	for line in trainfile:
	    splitted = line.rstrip().split(',')
	    label = int(splitted[0])
	    A_features = [float(item) for item in splitted[1:12]]
	    B_features = [float(item) for item in splitted[12:]]
	    y_train.append(label)
	    X_train_A.append(A_features)
	    X_train_B.append(B_features)
        trainfile.close()
	
	y_train = np.array(y_train)
	X_train_A = np.array(X_train_A)
	X_train_B = np.array(X_train_B)
	
        return (y_train, X_train_A, X_train_B)

def loadFileTest(filename):
	
	###########################
	# LOADING TRAINING DATA
	###########################
	
	trainfile = open(filename)
	header = trainfile.next().rstrip().split(',')
	
	y_train = []
	X_train_A = []
	X_train_B = []
	
	for line in trainfile:
	    splitted = line.rstrip().split(',')
	    A_features = [float(item) for item in splitted[0:11]]
	    B_features = [float(item) for item in splitted[11:]]
	    X_train_A.append(A_features)
	    X_train_B.append(B_features)
        trainfile.close()
	
	X_train_A = np.array(X_train_A)
	X_train_B = np.array(X_train_B)
	
        return (X_train_A, X_train_B)

	
calculatePrediction()

#
############################
## READING TEST DATA
############################
#
#testfile = open('../data/test.csv')
##ignore the test header
#testfile.next()
#
#X_test_A = []
#X_test_B = []
#for line in testfile:
#    splitted = line.rstrip().split(',')
#    A_features = [float(item) for item in splitted[0:11]]
#    B_features = [float(item) for item in splitted[11:]]
#    X_test_A.append(A_features)
#    X_test_B.append(B_features)
#testfile.close()
#
#X_test_A = np.array(X_test_A)
#X_test_B = np.array(X_test_B)
#
## transform features in the same way as for training to ensure consistency
#X_test = transform_features(X_test_A) - transform_features(X_test_B)
## compute probabilistic predictions
#p_test = model.predict_proba(X_test)
##only need the probability of the 1 class
#p_test = p_test[:,1:2]
#
