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
    return x

def calculatePrediction():
		
        dTr = loadFile('../data/trainSetWithWinsFormulaTest.csv')
        y_train = dTr[0]
        X_train_A = dTr[1]
        X_train_B = dTr[2]
        X_train_C = dTr[3]

        dTes = loadFileTest('../data/testSetWithFormula.csv')
        X_test_A = dTes[0]
        X_test_B = dTes[1]
        X_test_C = dTes[2]

        print "train size: {0} {1} {2}".format(X_train_A.shape, X_train_B.shape, X_train_C.shape)
        print "test size: {0} {1} {2}".format(X_test_A.shape, X_test_B.shape, X_test_C.shape)

	X_train_minus = transform_features(X_train_A) - transform_features(X_train_B)
	X_train_div = transform_features(X_train_A) / (transform_features(X_train_B) + 1)
	X_train = np.hstack((X_train_div, X_train_minus, X_train_C))

	X_test_minus = transform_features(X_test_A) - transform_features(X_test_B)
	X_test_div = transform_features(X_test_A) / (transform_features(X_test_B) + 1)
	X_test = np.hstack((X_test_div, X_test_minus, X_test_C))

        print "train size: {0}".format(X_train.shape)
        print "test size: {0}".format(X_test.shape)
	
        #In this case we'll use a random forest, but this could be any classifier
        cfr = RandomForestClassifier(n_estimators=100, max_features=math.sqrt(X_train.shape[1]), n_jobs=1)

        #Simple K-Fold cross validation.
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
        X_train_C = []
	
	for line in trainfile:
            line = line.replace("-","")

	    splitted = line.rstrip().split(',')
	    label = int(splitted[4])
	    A_features = [float(item) for item in splitted[5:16]]
	    B_features = [float(item) for item in splitted[16:]]
            C_features = [ float(splitted[1]) ]
	    y_train.append(label)
	    X_train_A.append(A_features)
	    X_train_B.append(B_features)
            X_train_C.append(C_features)
        trainfile.close()
	
	y_train = np.array(y_train)
	X_train_A = np.array(X_train_A)
	X_train_B = np.array(X_train_B)
	X_train_C = np.array(X_train_C)
	
        return (y_train, X_train_A, X_train_B, X_train_C)

def loadFileTest(filename):
	
	###########################
	# LOADING TEST DATA
	###########################
	
	trainfile = open(filename)
	header = trainfile.next().rstrip().split(',')
	
	y_train = []
	X_train_A = []
	X_train_B = []
        X_train_C = []
	
	for line in trainfile:
            line = line.replace("-","")
	    splitted = line.rstrip().split(',')
	    A_features = [float(item) for item in splitted[4:15]]
	    B_features = [float(item) for item in splitted[15:]]
            C_features = [float(splitted[1])]
	    X_train_A.append(A_features)
	    X_train_B.append(B_features)
            X_train_C.append(C_features)
        trainfile.close()
	
	X_train_A = np.array(X_train_A)
	X_train_B = np.array(X_train_B)
	X_train_C = np.array(X_train_C)
	
        return (X_train_A, X_train_B, X_train_C)

	
calculatePrediction()
