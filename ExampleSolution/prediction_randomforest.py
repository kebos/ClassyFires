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
	
	print "X_train_A"
	print X_train_A[1:rowsToPrint,]
	print "X_train_B"
	print X_train_B[1:rowsToPrint,]
	
	print "x_train_minus {0}".format(X_train_minus.shape)
	print X_train_minus[1:rowsToPrint,]
	print "x_train_div {0}".format(X_train_div.shape)
	print X_train_div[1:rowsToPrint,]
	
	X_train = np.concatenate((X_train_div, X_train_minus),axis=1)
	
	print "X_train_concat shape {0}".format(X_train.shape)
	print X_train[1:rowsToPrint,]
	
	#set the training responses
	#target = [x[0] for x in train]
	#set the training features
	#train = [x[1:] for x in train]
	#read in the test file
	#realtest = csv_io.read_data("test.csv")
	
	print "ytrain shape {0}".format(y_train.shape)
	print y_train[1:rowsToPrint,]
	
	X = [[0, 0], [1, 1], [0, 1]]
	Y = [0, 1, 1]
	
	#print "x {0}".format(X.shape)
	#print "y {0}".format(Y.shape)
	
	clf = RandomForestClassifier(n_estimators=10)
	clf = clf.fit(X, Y)
	
	#print clf
	
	print "DONE"
	
	# random forest code
	rf = RandomForestClassifier(n_estimators=10, max_features=math.sqrt(X_train.shape[1]), n_jobs=1)
	# fit the training data
	rf.fit(X_train, y_train)
	# run model against train data
	# (warning - no x-validation)
	p_train = rf.predict_proba(X_train)
	
	print p_train[0:10]
	
	# this is the probability of being 1
	p_train = [x[1] for x in p_train]
	
	#p_train0 = [x[0] for x in p_train]
	#print p_train0
	
	print p_train[0:10]
	#print p_train0[0:10]
	print y_train[0:10]
	
	#csv_io.write_delimited_file("random_forest_solution.csv", p_train)
	
	#print ('Random Forest Complete! You Rock! Submit random_forest_solution.csv to Kaggle')
	
	
	
	
	
	#model = linear_model.LogisticRegression(fit_intercept=False)
	#model.fit(X_train,y_train)
	# compute AuC score on the training data (BTW this is kind of useless due to overfitting, but hey, this is only an example solution)
	#p_train = model.predict_proba(X_train)
	#p_train = p_train[:,1:2]
	
	#print "p_train {0}".format(p_train.shape)
	
	print y_train.tolist().__class__.__name__
	print p_train.__class__.__name__
	
	print y_train[0].__class__.__name__
	print p_train[0].__class__.__name__
	
	# highly overfitted
	aucScore = auc_score(y_train.tolist(),p_train)
	
	
	print 'AuC score on training data: {0}'.format(aucScore)


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
