#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")
#imports
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline
#function  to rescale features and remove any outliers
def rescale_remove_outliers(features,labels,features_list):
	rescale = MinMaxScaler(feature_range=(0,1))
	#rescale features first
        #manual list of index values to keep some outliers in dataset
        manual_keep = [32,47,57,72,89,91,103,119,127,136,0,42,63,
        74,77,114,3,35,65,100,101,108,137,4,73,86,115,116,129,130,
        39,40,79,84,96,59,68,6,19,71,87,34,37,13,123,21,30,88,118,
        28,50,67,69,104]

	to_remove = []

    #manual indexed list of outliers to keep in if any
        features_1=rescale.fit_transform(features)
    #re-shape features for outlier removal
	f_all = [[] for x in xrange(0,len(features[0]))]
	for person in features_1:
		for i in range(0,len(person)):
			f_all[i].append(person[i])
    #remove outliers
	for f in f_all:
        #IQR code removed since too many outliers
        #were being removed
		#qr3 = np.percentile(f,75)
		#qr1 = np.percentile(f,25)
		#iqr =  abs(qr3 - qr1)
		for cnt,el in enumerate(f):
			if (el > np.percentile(f,90) ) and not labels[cnt] and not cnt in manual_keep:

				if cnt not in to_remove:
					to_remove.append(cnt)
        #remove outliers from features,labels
        features_outliers = np.take(features,to_remove,axis=0)
	features = np.delete(features,to_remove,axis=0)
    	labels_outliers = np.take(labels,to_remove)
	labels = np.delete(labels,to_remove)

	return features,labels

#function to produce features and labels from data
def data_to_feature_labels(my_dataset,features_list):
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	return labels,features

#select features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
				 'loan_advances', 'bonus', 'restricted_stock_deferred', \
				 'deferred_income', 'total_stock_value', 'expenses',  \
				 'exercised_stock_options', 'other', 'long_term_incentive',\
				 'restricted_stock', 'director_fees','to_messages', \
				 'from_poi_to_this_person', \
				 'from_messages', 'from_this_person_to_poi',\
				 'shared_receipt_with_poi','frac_from_poi','frac_to_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

#produce the frac_from_poi and frac_to_poi features
for i in my_dataset.keys():
   my_dataset[i]['frac_from_poi'] = float( \
    my_dataset[i]['from_poi_to_this_person']) / \
   float(my_dataset[i]['from_messages'])
   my_dataset[i]['frac_to_poi'] = float( \
    my_dataset[i]['from_this_person_to_poi']) / \
   float(my_dataset[i]['to_messages'])
   if np.isnan(my_dataset[i]['frac_from_poi']):
        my_dataset[i]['frac_from_poi'] = 0
   if np.isnan(my_dataset[i]['frac_to_poi']):
        my_dataset[i]['frac_to_poi'] = 0

### Extract features and labels from dataset for local testing
labels,features = data_to_feature_labels(my_dataset,features_list)
print "Number of POI labels in dataset: " + str(sum(labels))
print "Number of Non-POI labels in dataset: " + str(len(labels)-sum(labels))
print "Total Number of datapoints: " + str(len(labels))
print ''


#create list of features with no poi label
all_features_names = features_list[1:]

#Go through each feature, scale and remove outliers:
features,labels = rescale_remove_outliers(features,labels,all_features_names)
selector = SelectKBest(f_classif,k=13).fit(features,labels)
reduced_features_names = ['poi']
reduced_features_int = selector.get_support(indices=True)

#get reduced features and their f scores
print 'Reduced features and their ANOVA f scores: '
for cnt,i in enumerate(all_features_names):
	 if cnt in reduced_features_int:
	 	reduced_features_names.append(i)
	 	print '	',i,selector.scores_[cnt],'Include'
     	 else:
                print ' ',i,selector.scores_[cnt],'Exclude'
print ''
#It can be seen from the ANOVA f scores that no feature particularly dominates
#and acts as a
#marker for POI's
#Since many outlier points were removed, those that had outliers for the
#features that are no longer
#of interest should be added back in so that there is more data for training.

labels,features = data_to_feature_labels(my_dataset,reduced_features_names)
features,labels = rescale_remove_outliers(features,labels,reduced_features_names)
features = selector.fit_transform(features,labels)


print "New Number of POI labels in dataset: " + str(sum(labels))
print "New Number of Non-POI labels in dataset: " + str(len(labels)-sum(labels))
print "New Total Number of datapoints: " + str(len(labels))
print ''
#Now that the data is properly scaled, only has the K best features,
#and has outliers removed from the dataset,the data should be transformed using
#PCA.

#Note: PCA was removed since it slowed down the algorithm considerably
#and only reduced the results of the metrics.

#n_components='mle'
#pca = PCA(n_components=5)
#features = pca.fit_transform(features)
#print 'Number of features created from PCA: ' + str(pca.n_components_)
#print ''


#After testing the above classifiers with their default settings, it
#seems that using GaussianNB
#provides the best overall accuracy and recall scores while
#DecisionTreeClassifiers and AdaBoost
#are both reasonable alternatives. After performing another test
#with tester.py, the GaussianNB()
# classifier provides very good recall (> 0.8), but very poor precision (~0.1).
# In comparison,
#AdaBoost results in recall and accuracy that are both close to 0.3 and
#is more configurable.
#Because of this, I will choose to use AdaBoost for the classifier.
rescale = MinMaxScaler(feature_range=(0,1))
features_re = rescale.fit_transform(features)
#setting decision tree parameters to test
params_dt = {'criterion':('gini','entropy'),'min_samples_split': \
[i for i in range(2,10)],'max_features':('auto','log2',None)}
clf_dt = DecisionTreeClassifier()

#setting stratifiedshufflesplit to validate
cv_1 = StratifiedShuffleSplit(labels, 1000, random_state = 42)

#optimized decision tree
dt_params = GridSearchCV(DecisionTreeClassifier(),params_dt,cv=cv_1).fit \
(features_re,labels).best_params_
clf_dt.set_params(**dt_params)

#using adaboost to improve results of decisiontree
clf_dt = AdaBoostClassifier(base_estimator=clf_dt)

#optimized gaussiannb
gnb_params = \
GridSearchCV(GaussianNB(),{},cv=cv_1).fit(features_re,labels).best_params_
clf_gnb = GaussianNB()
clf_gnb.set_params(**gnb_params)

#setting voting classifier with equal soft weights
vc = VotingClassifier(estimators=[('dt',clf_dt),('gnb',clf_gnb)],\
    voting='soft',weights=[1,1])

clf = Pipeline([('rescale',rescale),('vc',vc)])
clf.fit(features,labels)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, reduced_features_names)
