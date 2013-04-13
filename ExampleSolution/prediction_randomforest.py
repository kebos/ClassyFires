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

def calculatePrediction(y_train, X_train_A, X_train_B, name):
		
	###########################
	# EXAMPLE BASELINE SOLUTION USING SCIKIT-LEARN
	#
	# using scikit-learn LogisticRegression module without fitting intercept
	# to make it more interesting instead of using the raw features we transform them logarithmically
	# the input to the classifier will be the difference between transformed features of A and B
	# the method roughly follows this procedure, except that we already start with pairwise data
	# http://fseoane.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
	###########################
	
	#def transform_features(x):
	#    return np.log(1+x)
	
	X_train_minus = transform_features(X_train_A) - transform_features(X_train_B)
	X_train_div = transform_features(X_train_A) / (transform_features(X_train_B) + 1)
	
	np.set_printoptions(threshold=np.nan)
	
	rowsToPrint = 5
	
#	print "X_train_A"
#	print X_train_A[1:rowsToPrint,]
#	print "X_train_B"
#	print X_train_B[1:rowsToPrint,]
#	
#	print "x_train_minus {0}".format(X_train_minus.shape)
#	print X_train_minus[1:rowsToPrint,]
#	print "x_train_div {0}".format(X_train_div.shape)
#	print X_train_div[1:rowsToPrint,]
#	
	X_train = np.concatenate((X_train_div, X_train_minus),axis=1)
	
        #In this case we'll use a random forest, but this could be any classifier
        cfr = RandomForestClassifier(n_estimators=10, max_features=math.sqrt(X_train.shape[1]), n_jobs=1)

    #Simple K-Fold cross validation. 5 folds.
        cv = cross_validation.KFold(len(X_train), k=5, indices=False)

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
#
#
#
#	print "X_train_concat shape {0}".format(X_train.shape)
#	print X_train[1:rowsToPrint,]
#	
#	# random forest code
#	rf = RandomForestClassifier(n_estimators=10, max_features=math.sqrt(X_train.shape[1]), n_jobs=1)
#	# fit the training data
#	rf.fit(X_train, y_train)
#	# run model against train data
#	# (warning - no x-validation)
#	p_train = rf.predict_proba(X_train)
#	
#	print p_train[0:10]
#	
#	# this is the probability of being 1
#	
#	# highly overfitted
#	aucScore = auc_score(y_train.tolist(),p_train)
#	
#	print 'AuC score on training data: {0}'.format(aucScore)
#
	###########################
	# WRITING SUBMISSION FILE
	###########################
	predfile = open('predictions_{0}.csv'.format(name),'w+')

        print p_train[0:10]
	
        print >>predfile, ','.join([str(item) for item in p_train])
	
	predfile.close()

header = ""

def loadFile(filename):
	
	###########################
	# LOADING TRAINING DATA
	###########################
	
	trainfile = open('../data/train.csv')
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
	
dataTuple = loadFile("../data/train.csv")

y_train = dataTuple[0]
X_train_A = dataTuple[1]
X_train_B = dataTuple[2]

calculatePrediction(y_train, X_train_A, X_train_B, "training")

dataTuple = loadFile("../data/test.csv")

y_train = dataTuple[0]
X_train_A = dataTuple[1]
X_train_B = dataTuple[2]

calculatePrediction(y_train, X_train_A, X_train_B, "test")

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
