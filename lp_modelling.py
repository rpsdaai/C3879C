import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import numpy as np
import matplotlib.pyplot as plt

import logging
import warnings  # To ignore any warnings warnings.filterwarnings("ignore")

import lp_eda

warnings.filterwarnings("ignore")
log = logging.getLogger()

# classifierDict = {
# 	'LR': LogisticRegression(solver='liblinear'),
# 	'DT': DecisionTreeClassifier(),
# 	'RF': RandomForestClassifier(max_depth=10, n_estimators=10),
# 	'XGB': XGBClassifier(),
#
# }

# auc
aucScores_ = {}
classifierlist = []
aucScoresCounter = 0

def do_computeROCScores(model, X, y, classifierName):
	# Ref: https://stackoverflow.com/questions/10851906/python-3-unboundlocalerror-local-variable-referenced-before-assignment/10851939
	global aucScoresCounter
	global aucScores_
	global classifierlist

	proba = model.predict_proba(X)[:,1]
	frp,trp, threshold = roc_curve(y, proba)

	aucScore = dict()

	aucScore['frp'] = frp
	aucScore['trp'] = trp
	aucScore['threshold'] = threshold
	# Computes AUC, returns float
	aucScore['auc'] = auc(frp,trp)

	aucScores_[aucScoresCounter] = aucScore
	aucScoresCounter = aucScoresCounter + 1

	classifierlist.append(classifierName)

def do_LogisticRegression(x_train, y_train):
	# kf = KFold(n_splits=10)
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	model = LogisticRegression(solver='liblinear')
	model.fit(x_train, y_train)
	#
	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	roc_auc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'LR')

	return accuracy, roc_auc, model


def do_DecisionTree(x_train, y_train):
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	model = DecisionTreeClassifier()
	model.fit(x_train, y_train)

	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'DT')
	return accuracy, auc_roc, model


def do_RandomForest(x_train, y_train):
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	model = RandomForestClassifier(max_depth=10, n_estimators=10)
	model.fit(x_train, y_train)

	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'RF')

	return accuracy, auc_roc, model


def do_XGBoost(x_train, y_train):
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	model = XGBClassifier()
	model.fit(x_train, y_train)

	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'XGB')

	return accuracy, auc_roc, model


def do_Bagging_Ensemble(x_train, y_train):
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	cart = DecisionTreeClassifier()
	num_trees = 100
	model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees)
	model.fit(x_train, y_train)
	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'BEN')

	return accuracy, auc_roc, model


def do_Boosting(x_train, y_train, num_trees):
	skf = StratifiedKFold(n_splits=10, shuffle=True)

	model = AdaBoostClassifier(n_estimators=num_trees)
	model.fit(x_train, y_train)

	accuracy = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = cross_val_score(model, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(model, x_train, y_train, 'ADA')

	return accuracy, auc_roc, model


# Ref: https://setscholars.net/2019/02/17/how-to-implement-voting-ensembles-in-python/
def do_VotingEnsemble(x_train, y_train):
	skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

	# create the sub models
	estimators = []
	model1 = LogisticRegression(solver='liblinear')
	estimators.append(('logistic', model1))
	model2 = DecisionTreeClassifier()
	estimators.append(('cart', model2))
	# Ref: https://www.discoverbits.in/371/sklearn-attributeerror-predict_proba-available-probability
	# if probability not set will get the following error: "AttributeError: predict_proba is not available
	# when probability=False" for SVC classifiers. Setting voting to soft drops the accuracy
	model3 = SVC(gamma='auto', probability=True)
	estimators.append(('svm', model3))

	log.debug(estimators)
	# create the ensemble model
	# Have to set voting = 'soft' otherwise get the error: predict_proba is not available when voting='hard'
	# Objective to set voting to soft is so that we can compute roc_auc scoring...
	ensemble = VotingClassifier(estimators, voting='soft')
	# ensemble = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3)], voting='soft')
	ensemble.fit(x_train, y_train)

	accuracy = model_selection.cross_val_score(ensemble, x_train, y_train, scoring='accuracy', cv=skf)
	auc_roc = model_selection.cross_val_score(ensemble, x_train, y_train, scoring='roc_auc', cv=skf)

	do_computeROCScores(ensemble, x_train, y_train, 'VOT')

	return accuracy, auc_roc, ensemble

# No grid searching used
def do_computeModelAccuracy(X, y):
	accuracyList = []
	auc_rocList = []
	modelList = []

	lr_accuracy, lr_aucroc, lr_ = do_LogisticRegression(X, y)
	accuracyList.append(round(100 * lr_accuracy.mean(), 2))
	auc_rocList.append(round(100 * lr_aucroc.mean(), 2))
	modelList.append(lr_)
	log.debug('LR Accuracy: ' + str(round(100 * lr_accuracy.mean(), 2)) + "%")
	log.debug('LR AUC_ROC: ' + str(round(100 * lr_aucroc.mean(), 2)) + "%")

	dt_accuracy, dt_aucroc, dt_ = do_DecisionTree(X, y)
	accuracyList.append(round(100 * dt_accuracy.mean(), 2))
	auc_rocList.append(round(100 * dt_aucroc.mean(), 2))
	modelList.append(dt_)
	log.debug('Decision Tree Accuracy: ' + str(round(100 * dt_accuracy.mean(), 2)) + "%")
	log.debug('Decision Tree AUC ROC: ' + str(round(100 * dt_aucroc.mean(), 2)) + "%")

	rf_accuracy, rf_aucroc, rf_ = do_RandomForest(X, y)
	accuracyList.append(round(100 * rf_accuracy.mean(), 2))
	auc_rocList.append(round(100 * rf_aucroc.mean(), 2))
	modelList.append(rf_)
	log.debug('RF Accuracy: ' + str(round(100 * rf_accuracy.mean(), 2)) + "%")
	log.debug('RF AUC ROC: ' + str(round(100 * rf_aucroc.mean(), 2)) + "%")

	xgb_accuracy, xgb_aucroc, xgb_ = do_XGBoost(X, y)
	accuracyList.append(round(100 * xgb_accuracy.mean(), 2))
	auc_rocList.append(round(100 * xgb_aucroc.mean(), 2))
	modelList.append(xgb_)
	log.debug('XGBoost Accuracy: ' + str(round(100 * xgb_accuracy.mean(), 2)) + "%")
	log.debug('XGBoost AUC ROC: ' + str(round(100 * xgb_aucroc.mean(), 2)) + "%")

	ben_accuracy, ben_aucroc, ben_ = do_Bagging_Ensemble(X, y)
	accuracyList.append(round(100 * ben_accuracy.mean(), 2))
	auc_rocList.append(round(100 * ben_aucroc.mean(), 2))
	modelList.append(ben_)
	log.debug('Bagging Accuracy: ' + str(round(100 * ben_accuracy.mean(), 2)) + "%")
	log.debug('Bagging AUC ROC: ' + str(round(100 * ben_aucroc.mean(), 2)) + "%")

	ada_accuracy, ada_aucroc, ada_ = do_Boosting(X, y, 7)
	accuracyList.append(round(100 * ada_accuracy.mean(), 2))
	auc_rocList.append(round(100 * ada_aucroc.mean(), 2))
	modelList.append(ada_)
	log.debug('Ada Boosting Accuracy: ' + str(round(100 * ada_accuracy.mean(), 2)) + "%")
	log.debug('Ada Boosting AUC ROC: ' + str(round(100 * ada_aucroc.mean(), 2)) + "%")

	ens_accuracy, ens_aucroc, ens_ = do_VotingEnsemble(X, y)
	# ens_accuracy, ens_ = do_VotingEnsemble(X, y)
	accuracyList.append(round(100 * ens_accuracy.mean(), 2))
	auc_rocList.append(round(100 * ens_aucroc.mean(), 2))
	modelList.append(ens_)
	log.debug('Voting Ensemble Accuracy: ' + str(round(100 * ens_accuracy.mean(), 2)) + "%")
	log.debug('Voting Ensemble AUC ROC: ' + str(round(100 * ens_aucroc.mean(), 2)) + "%")

	return (accuracyList, auc_rocList, modelList)

# Ref: https://datascienceplus.com/predict-employee-turnover-with-python/
# Ref: https://www.kaggle.com/sudhirnl7/logistic-regression-with-stratifiedkfold/notebook#Explore-data-set
# Ref: https://medium.com/datadriveninvestor/choosing-the-best-algorithm-for-your-classification-model-7c632c78f38f
def do_computeModelMetrics(model, X, y):
	skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

	# accuracy
	accScores = []
	# roc
	rocScores = []
	# confusion matrix
	cmScores = []
	# auc
	aucScores = {}

	i = 1
	for train_index, test_index in skf.split(X, y):
		log.debug('{} of KFold {}'.format(i, skf.n_splits))

		X_train, X_test = X.loc[train_index], X.loc[test_index]
		y_train, y_test = y.loc[train_index], y.loc[test_index]

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		accScore = accuracy_score(y_test, y_pred)
		accScores.append(accScore)

		# roc score as float returned
		rocScore = roc_auc_score(y_test, y_pred)
		rocScores.append(rocScore)

		log.debug('ROC AUC score:' + str(rocScore))

		cmScore = confusion_matrix(y_test, y_pred)
		cmScores.append(cmScore)

		y_pred_prob = model.predict_proba(X_test)[:,1]
		log.debug(X_test.shape)

		#log.debug('SCORE: ' + str(model.score(xvl,yvl)))
		log.debug('Model SCORE: ' + str(model.score(X_test, y_test)))

		proba = model.predict_proba(X_test)[:,1]
		frp, trp, threshold = roc_curve(y_test, proba)
		aucScore = dict()
		log.debug('frp: ' + str(frp.shape))

		aucScore['frp'] = frp
		aucScore['trp'] = trp
		aucScore['threshold'] = threshold
		# Computes AUC, returns float
		aucScore['auc'] = auc(frp,trp)

		# do this because index 1 starts from 1
		aucScores[i-1] = aucScore

		i += 1

	# On full data
	'''
	proba = model.predict_proba(X)[:,1]
	frp,trp, threshold = roc_curve(y, proba)
	roc_auc_ = auc(frp,trp)
	'''
	return accScores, rocScores, cmScores, aucScores

# Ref: http://queirozf.com/entries/visualizing-machine-learning-models-examples-with-scikit-learn-and-matplotlib
# Ref: https://stackoverflow.com/questions/42894871/how-to-plot-multiple-roc-curves-in-one-plot-with-legend-and-auc-scores-in-python
# Multiple ROC curves on 1 graph that was cross validated for a particular classifier
def do_plotRocCurves(aucScores, filename):
	global classifierlist

	# Now, plot the computed values
	for i in range(len(aucScores)):
		# plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
		roc_label = '{} (AUC={:.2f})'.format(classifierlist[i] + ' ROC ', aucScores[i]['auc'])
		#plt.plot(aucScores[i]['frp'], aucScores[i]['trp'], label='%s ROC ' % (aucScores[i]['auc']))
		plt.plot(aucScores[i]['frp'], aucScores[i]['trp'], label=roc_label)
	# Custom settings for the plot
	plt.plot([0, 1], [0, 1], linestyle='dashed', color='red', linewidth=2, label='random')

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate') # Specificity
	plt.ylabel('True Positive Rate') # Sensitivity
	plt.title('Receiver Operating Characteristic')
	plt.legend(fontsize=10, loc='best')
	plt.tight_layout()

	# Save plot
	plt.savefig(filename)

	# Display
	plt.show()

# No Grid Selection
# Build & train model from best scores
# save the model as filename
# lib2Use - p = pickle, j = joblib
def do_createModel(model, X, y, filename, lib2Use):
	log.debug('Building Model File  ....')
	model.fit(X, y)
	lp_eda.do_saveModel(filename, model, lib2Use)


#
# Grid Searching
#
algo_scores_tracker = {}


def do_GridSearch(classifier, x_train, y_train, params, nfolds):
	# skf = model_selection.StratifiedKFold(n_splits = 3, shuffle=True)

	# grid_search = GridSearchCV(classifier, params, cv=nfolds, iid=True)
	grid_search = GridSearchCV(classifier, params, cv=nfolds, iid=True, scoring=['accuracy', 'roc_auc'], refit='accuracy')
	grid_search.fit(x_train, y_train)  # Fit the grid search model

	return (grid_search)


def update_GridSearch_Scores(grid_search, model, model_suffix):
	global algo_scores_tracker

	log.debug('update_GridSearch_Scores')

	# algo_scores['Estimator'] = grid_search.best_estimator_
	# algo_scores['Score'] = grid_search.best_score_
	# algo_scores['Parameters'] = grid_search.best_params_

	algo_scores = dict()
	algo_scores['Model'] = model
	algo_scores['Estimator'] = grid_search.best_estimator_
	algo_scores['Score'] = grid_search.best_score_
	algo_scores['Parameters'] = grid_search.best_params_

	model_name = model.__class__.__name__
	# if same model but different parameters
	model_name = model_name + model_suffix
	log.debug('Model Name with suffix: ' + model_name)

	algo_scores_tracker[model_name] = algo_scores
	log.debug('Grid Searching ...')
	log.debug("Final Model Name: " + model_name)
	log.debug('Estimator: ' + str(algo_scores_tracker[model_name]['Estimator']))
	log.debug('Score: ' + str(algo_scores_tracker[model_name]['Score']))
	log.debug('Parameters: ' + str(algo_scores_tracker[model_name]['Parameters']))
	log.debug('Model Name: ' + str(algo_scores_tracker[model_name]))

	return (algo_scores_tracker)

def do_useGridSearch2FindBestModel(X, y):
	global algo_scores_tracker

	#
	# Logistic Regression Grid Search
	#
	log.debug('Starting LogisticRegression Grid Searching ....')
	lr_cv_params = {
		'dual': [True, False],
		'C': [1.0, 1.5, 2.0, 2.5],
		'max_iter': [100, 110, 120, 130, 140]
	}
	lr_model = LogisticRegression(solver='liblinear')
	gs_lr = do_GridSearch(lr_model, X, y, lr_cv_params, 10)

	do_computeROCScores(gs_lr, X, y, 'LR')

	algo_scores_tracker = update_GridSearch_Scores(gs_lr, lr_model, '')

	# Decision Tree Grid Search - Warning SLOW!!
	# Ref: https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
	'''
	log.debug('Starting Decision Tree Grid Searching ....')
	dt_cv_params = {
		'criterion': ['gini', 'entropy'],
		'splitter': ['best', 'random'],
		'max_depth': [2, 4, 6, 8, 10],
		'min_samples_leaf': min_samples_leaves.tolist(),
		'min_samples_split': min_samples_splits.tolist(),
		'max_features': max_features
	}
	dt_model = DecisionTreeClassifier()
	gs_dt = do_GridSearch(dt_model, X, y, dt_cv_params, 10)
	
	algo_scores_tracker = update_GridSearch_Scores(gs_dt, dt_model, '')
	'''

	log.debug('Starting Random Forest Grid Searching ....')

	# RandomForest Grid Search
	rf_cv_params = {
		'n_estimators': [4, 6, 9, 10],
		'max_features': ['log2', 'sqrt', 'auto'],
		'criterion': ['entropy', 'gini'],
		'max_depth': [2, 3, 5, 10],
		'min_samples_split': [2, 3, 5],
		'min_samples_leaf': [1, 5, 8]
	}
	rf_model = RandomForestClassifier()
	gs_rf = do_GridSearch(rf_model, X, y, rf_cv_params, 10)

	do_computeROCScores(gs_rf, X, y, 'RF')

	algo_scores_tracker = update_GridSearch_Scores(gs_rf, rf_model, '')

	# XGBoost Grid Search
	log.debug('Starting XGBoosting 1 Grid Searching ....')

	xgb1_cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
	xgb1_ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8,
	                   'objective': 'binary:logistic'}

	xgb_model_1 = XGBClassifier(**xgb1_ind_params)
	gs_xgb_1 = do_GridSearch(xgb_model_1, X, y, xgb1_cv_params, 10)

	do_computeROCScores(gs_xgb_1, X, y, 'XGB_1')

	algo_scores_tracker = update_GridSearch_Scores(gs_xgb_1, xgb_model_1, '_1')

	log.debug('Starting XGBoosting 2 Grid Searching ....')
	xgb2_cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7, 0.8, 0.9]}
	xgb2_ind_params = {'n_estimators': 1000, 'colsample_bytree': 0.8,
	                   'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 3}
	xgb_model_2 = XGBClassifier(**xgb2_ind_params)
	gs_xgb_2 = do_GridSearch(xgb_model_2, X, y, xgb2_cv_params, 10)

	do_computeROCScores(gs_xgb_2, X, y, 'XGB_2')

	algo_scores_tracker = update_GridSearch_Scores(gs_xgb_2, xgb_model_2, '_2')

	log.debug('Starting XGBoosting 3 Grid Searching ....')
	xgb3_cv_params = {
		'max_depth': [3, 5, 7],
		'min_child_weight': [1, 3, 5]
	}
	xgb3_ind_params = {
		'objective': 'binary:logistic',
		'booster': 'gbtree',
		'eval_metric': 'error',
		'n_estimators': 1000,
		'nthread': 4,
		'silent': 1,
		'max_depth': 6,
		'subsample': 0.9,
		'min_child_weight': 3,
		'colsample_bytree': 0.9,
		'eta': 0.1,
		'verbose_eval': True
	}
	xgb_model_3 = XGBClassifier(**xgb3_ind_params)
	gs_xgb_3 = do_GridSearch(xgb_model_3, X, y, xgb3_cv_params, 10)

	do_computeROCScores(gs_xgb_3, X, y, 'XGB_3')

	algo_scores_tracker = update_GridSearch_Scores(gs_xgb_3, xgb_model_3, '_3')

	log.debug('Starting XGBoosting 4 Grid Searching ....')
	xgb4_cv_params = {
		'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
		'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750]
	}
	xgb4_ind_params = {
		'max_depth': 4,
		'min_samples_split': 2,
		'min_samples_leaf': 1,
		'subsample': 1,
		'max_features': 'sqrt'
	}
	xgb_model_4 = XGBClassifier(**xgb4_ind_params)
	gs_xgb_4 = do_GridSearch(xgb_model_4, X, y, xgb4_cv_params, 10)

	do_computeROCScores(gs_xgb_4, X, y, 'XGB_4')

	algo_scores_tracker = update_GridSearch_Scores(gs_xgb_4, xgb_model_4, '_4')

	log.debug(len(algo_scores_tracker))
	log.debug(algo_scores_tracker.keys())


def do_getModelScores_fromGridSearch():
	log.debug('do_getModelScores_fromGridSearch()')
	global algo_scores_tracker

	acc_key = ''
	acc_score = 0
	for k, v in algo_scores_tracker.items():
		log.debug('KEY: ' + k + ' VALUE: ' + str(v))
		score = algo_scores_tracker[k]['Score']
		if score > acc_score:
			acc_score = score
			acc_key = k

	log.debug('BEST Model: ' + acc_key + ' Score: ' + str(acc_score))
	return (algo_scores_tracker[acc_key]['Model'])
