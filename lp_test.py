import pandas as pd
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

import os

import lp_eda
import lp_visualize as vz
import lp_find_impt_features as fip
import lp_modelling as lpm

log = logging.getLogger()

#
# Visualization Tests - START
#
def do_removeFiles(filename):
	if os.path.exists(filename):
		os.remove(filename)
	else:
		log.debug(filename + ' DOES NOT exist')

def do_plotCategoricals_Test(X):
	log.debug('do_plotCategoricals_Test: ')

	vc1, nvc1 = vz.do_countCategoricalValues(X, 'Gender')
	vc2, nvc2 = vz.do_countCategoricalValues(X, 'Married')
	vc3, nvc3 = vz.do_countCategoricalValues(X, 'Self_Employed')
	vc4, nvc4 = vz.do_countCategoricalValues(X, 'Credit_History')

	# Plot categorical features counts - not normalized
	do_removeFiles('cat_not_norm.png')
	vz.do_plotCategorical_Notnormalised(X, ['Gender', 'Married', 'Self_Employed', 'Credit_History'], 'cat_not_norm.png')

	do_removeFiles('cat_norm_1.png')
	vz.do_plotCategorical_Normalised(nvc1, 'Gender', 'cat_norm_1.png')

	do_removeFiles('cat_norm_2.png')
	vz.do_plotCategorical_Normalised(nvc2, 'Married', 'cat_norm_2.png')

	do_removeFiles('cat_norm_3.png')
	vz.do_plotCategorical_Normalised(nvc3, 'Self_Employed', 'cat_norm_3.png')

	do_removeFiles('cat_norm_4.png')
	vz.do_plotCategorical_Normalised(nvc4, 'Credit_History', 'cat_norm_4.png')

def do_plotNumericals_Test(X):
	log.debug('do_plotNumericals_Test: ')

	do_removeFiles('num_1.png')
	vz.do_plotNumericalDataDistribution_OneFeature(X, 'ApplicantIncome', 'num_1.png')

	do_removeFiles('num_2.png')
	vz.do_plotNumericalDataDistribution_OneFeature(X, 'CoapplicantIncome', 'num_2.png')

	do_removeFiles('num_3.png')
	vz.do_plotNumericalDataDistribution_OneFeature(X, 'LoanAmount', 'num_3.png')

	do_removeFiles('num_4.png')
	vz.do_plotDataNumericalDistribution_2Features(X, 'Education', 'ApplicantIncome', 'num_4.png')

def do_plotOrdinals_Test(X):
	log.debug('do_plotOrdinals_Test: ')
	dep_vc1, dep_nvc1 = vz.do_countCategoricalValues(X, 'Dependents')
	edu_vc2, edu_nvc2 = vz.do_countCategoricalValues(X, 'Education')
	prop_vc3, prop_nvc3 = vz.do_countCategoricalValues(X, 'Property_Area')

	ord_vc_list = [dep_vc1, edu_vc2, prop_vc3]
	ord_nvc_list = [dep_nvc1, edu_nvc2, prop_nvc3]

	do_removeFiles('ordinals.png')
	vz.do_plotOrdinals(ord_vc_list, ord_nvc_list, ['Dependents', 'Education', 'Property_Area'], 'ordinals.png')

# BiVariate Plots
def do_plotXTab_Test(full_df):
	log.debug('do_plotXTab: ')
	vz.do_crossTab(full_df, 'Gender', 'Loan_Status')

	do_removeFiles('xtgbp1.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Gender'], 'Loan_Status', 'xtgbp1.png')

	do_removeFiles('xtgbp2.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Married'], 'Loan_Status', 'xtgbp2.png')

	do_removeFiles('xtgbp3.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Self_Employed'], 'Loan_Status', 'xtgbp3.png')

	do_removeFiles('xtgbp4.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Credit_History'], 'Loan_Status', 'xtgbp4.png')

	do_removeFiles('xtgbp5.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Dependents'], 'Loan_Status', 'xtgbp5.png')

	do_removeFiles('xtgbp6.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Education'], 'Loan_Status', 'xtgbp6.png')

	do_removeFiles('xtgbp7.png')
	vz.do_multiple_XTab_GroupedBarPlot(full_df, 1, 1, ['Property_Area'], 'Loan_Status', 'xtgbp7.png')

def do_plotXCut_Test(full_df):
	log.debug('do_plotXCut: ')

	do_removeFiles('xc_1.png')
	vz.do_multiple_XCut_GroupedBarPlot(full_df, 1, 1, ['ApplicantIncome'], 'Loan_Status',
	                                [0, 2500, 4000, 6000, 81000], ['Low', 'Average', 'High', 'Very High'], 'xc_1.png')

	do_removeFiles('xc_2.png')
	vz.do_multiple_XCut_GroupedBarPlot(full_df, 1, 1, ['CoapplicantIncome'], 'Loan_Status',
	                                [0, 1000, 3000, 42000], ['Low', 'Average', 'High'], 'xc_2.png')

	do_removeFiles('xc_3.png')
	vz.do_multiple_XCut_GroupedBarPlot(full_df, 1, 1, ['Total_Income'], 'Loan_Status',
	                                [0, 2500, 4000, 6000, 81000], ['Low', 'Average', 'High', 'Very High'], 'xc_3.png')

	do_removeFiles('xc_4.png')
	vz.do_multiple_XCut_GroupedBarPlot(full_df, 1, 1, ['LoanAmount'], 'Loan_Status',
	                                [0, 100, 200, 700], ['Low', 'Average', 'High'], 'xc_4.png')

	do_removeFiles('grpby.png')
	vz.do_grpByPlot(full_df, 'Loan_Status', 'ApplicantIncome', 'grpby.png')

#
# Visualization Tests - END
#

#
# Plot important features - START
#

# filename - save the important features plot to this
# plot the features ranked according to importance
def do_plotImportantFeatures_Test(filename):
	log.debug('do_plotImportantFeatures_Test: ' + filename)
	X, y = lp_eda.do_processDataset(True, True, True)
	importances = fip.do_findBestFeaturesUsingXtraTreeClassifier(X, y)
	fip.do_plotImportantFeatures(importances, X, filename)

#
# Plot important features - END
#

#
# Modelling - START
#
def do_findBestModel_NoGS():
	log.debug('do_findBestModel_NoGS')
	X, y = lp_eda.do_processDataset(True, True, True)
	log.debug('FEATURE Engineering: ' + str(X.columns))

	lpm.classifierlist.clear()
	lpm.aucScores_.clear()

	accuracyList, aucrocList, modelList = lpm.do_computeModelAccuracy(X, y)
	lpm.do_plotRocCurves(lpm.aucScores_, 'roc_curve_no_gs.png')

	log.debug('Accuracy List Length: ' + str(len(accuracyList)))

	for i in range(len(accuracyList)):
		log.debug(str(i) + ' ' + str(accuracyList[i]))

	for i in range(len(aucrocList)):
		log.debug(str(i) + ' ' + str(aucrocList[i]))

	# Ref: https://www.tutorialspoint.com/python-program-to-find-maximum-and-minimum-element-s-position-in-a-list

	model2UseIndex = accuracyList.index(max(accuracyList))
	log.debug('Max accuracy value: ' + str(max(accuracyList)) + ' Position: ' + str(accuracyList.index(max(accuracyList))) )

	lpm.do_createModel(modelList[model2UseIndex], X, y, 'my_no_fe_gs_model_2.pkl', 'p')

def do_findBestModel_GS():
	log.debug('do_findBestModel_GS')
	X, y = lp_eda.do_processDataset(True, True, True)

	lpm.classifierlist.clear()
	lpm.aucScores_.clear()
	aucScoresCounter = 0

	lpm.do_useGridSearch2FindBestModel(X, y)
	log.debug('Classifierlist Length: ' + str(len(lpm.classifierlist)))
	for i in range(len(lpm.classifierlist)):
		log.debug(lpm.classifierlist[i])


	lpm.do_plotRocCurves(lpm.aucScores_, 'roc_curve_gs.png')
	model = lpm.do_getModelScores_fromGridSearch()
	lpm.do_createModel(model, X, y, 'mygs_fe_model_2.pkl', 'p')
	# do_test('mygsmodel.pkl')
#
# Modelling - END
#

#
# Predict against selected saved model using test data - START
#

#
# Do unit testing: NO Feature Engineering Case
#
def do_positive_test(filename, ohecols_file):
	log.debug('do_positive_test: ' + filename)
	ohecols = lp_eda.do_loadModel(ohecols_file)
	log.debug(ohecols)
	data = [
		{
			'Gender': 'Male',
			'Married': 'Yes',
			'Dependents': 0,
			'Education': 'Not Graduate',
			'Self_Employed': 'Yes',
			'ApplicantIncome': 2165,
			'CoapplicantIncome': 3422,
			'LoanAmount': 152,
			'Loan_Amount_Term': 360,
			'Credit_History': 1,
			'Property_Area': 'Urban'
		}
	]
	test_df = pd.DataFrame.from_dict(data)
	# One Hot Encoding
	test_df = pd.get_dummies(test_df).reindex(columns=ohecols, fill_value=0)

	test_model = lp_eda.do_loadModel(filename)

	# Scale data
	tmp_df = lp_eda.scale_data(test_df, MinMaxScaler(),
	                           ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])
	result = test_model.predict(tmp_df)
	if result == 0:
		log.debug('Loan NOT approved: ' + str(result))
	else:
		log.debug('Loan approved: ' + str(result))

#
# Do unit testing: NO Feature Engineering Case
#
def do_negative_test(filename, ohecols_file):
	log.debug('do_negative_test: ' + filename)
	ohecols = lp_eda.do_loadModel(ohecols_file)
	log.debug(ohecols)
	data = [
		{
			'Gender': 'Male',
			'Married': 'Yes',
			'Dependents': 2,
			'Education': 'Not Graduate',
			'Self_Employed': 'No',
			'ApplicantIncome': 3881,
			'CoapplicantIncome': 0,
			'LoanAmount': 147,
			'Loan_Amount_Term': 360,
			'Credit_History': 0,
			'Property_Area': 'Rural'
		}
	]
	test_df = pd.DataFrame.from_dict(data)
	# One Hot Encoding
	test_df = pd.get_dummies(test_df).reindex(columns=ohecols, fill_value=0)

	test_model = lp_eda.do_loadModel(filename)

	# Scale data
	tmp_df = lp_eda.scale_data(test_df, MinMaxScaler(),
	                           ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])
	result = test_model.predict(tmp_df)
	if result == 0:
		log.debug('Loan NOT approved: ' + str(result))
	else:
		log.debug('Loan approved: ' + str(result))

#
# Do unit testing (Positive Testing)
#
def do_test_with_fe_pos(filename, ohecols_file):
	log.debug('do_test_with_fe_pos: ' + filename)
	ohecols = lp_eda.do_loadModel(ohecols_file)
	log.debug(ohecols)

	data = [
		{
			'Gender': 'Male',
			'Married': 'Yes',
			'Dependents': 0,
			'Education': 'Not Graduate',
			'Self_Employed': 'Yes',
			'ApplicantIncome': 2165,
			'CoapplicantIncome': 3422,
			'LoanAmount': 152,
			'Loan_Amount_Term': 360,
			'Credit_History': 1,
			'Property_Area': 'Urban'
		}
	]

	test_df = pd.DataFrame.from_dict(data)

	lp_eda.do_FeatureEngineering(test_df)
	# One Hot Encoding
	test_df = pd.get_dummies(test_df).reindex(columns=ohecols, fill_value=0)

	test_model = lp_eda.do_loadModel(filename)

	# Scale data
	tmp_df = lp_eda.scale_data(test_df, RobustScaler(), ['LoanAmount', 'Loan_Amount_Term'])

	result = test_model.predict(tmp_df)
	if result == 0:
		log.debug('Loan NOT approved: ' + str(result))
	else:
		log.debug('Loan approved: ' + str(result))

def do_test_with_fe_neg(filename, ohecols_file):
	log.debug('do_test_with_fe_neg: ' + filename)
	ohecols = lp_eda.do_loadModel(ohecols_file)
	log.debug(ohecols)

	data = [
		{
			'Gender': 'Male',
			'Married': 'Yes',
			'Dependents': 2,
			'Education': 'Not Graduate',
			'Self_Employed': 'No',
			'ApplicantIncome': 3881,
			'CoapplicantIncome': 0,
			'LoanAmount': 147,
			'Loan_Amount_Term': 360,
			'Credit_History': 0,
			'Property_Area': 'Rural'
		}
	]

	test_df = pd.DataFrame.from_dict(data)

	lp_eda.do_FeatureEngineering(test_df)
	log.debug('After FE')
	log.debug(test_df.columns)
	# One Hot Encoding
	test_df = pd.get_dummies(test_df).reindex(columns=ohecols, fill_value=0)
	log.debug('After OHE')
	log.debug(test_df.columns)
	test_model = lp_eda.do_loadModel(filename)

	# Scale data
	tmp_df = lp_eda.scale_data(test_df, RobustScaler(), ['LoanAmount', 'Loan_Amount_Term'])

	result = test_model.predict(tmp_df)
	if result == 0:
		log.debug('Loan NOT approved: ' + str(result))
	else:
		log.debug('Loan approved: ' + str(result))

#
# Predict against selected saved model using test data - END
#


if __name__ == '__main__':
	#
	# Visualization Tests - START
	#

	# X, y = lp_eda.do_processDataset(False, False, False)
	# do_plotCategoricals_Test(X)
	# do_plotNumericals_Test(X)
	# do_plotOrdinals_Test(X)
	#
	# full_df = X.copy() # scaled, ohe and feature engineered
	# full_df['Loan_Status'] = y
	# # remap to original categorical value
	# full_df['Loan_Status'] = full_df['Loan_Status'].map({0:'N', 1:'Y'})
	# full_df['Total_Income'] = full_df['ApplicantIncome'] + full_df['CoapplicantIncome']
	#
	# do_plotXCut_Test(full_df)
	# do_plotXTab_Test(full_df)

	#
	# Visualization Tests - END
	#

	#
	# Visualize important features - START
	#

	# do_plotImportantFeatures_Test('rank_features.png')

	#
	# Visualize important features - END
	#

	#
	# Modelling - START
	#
	# do_findBestModel_NoGS()

	#
	# Testing against model found - START
	#
	# do_test_with_fe_neg('my_no_fe_gs_model_2.pkl', 'ohecols_fe.pkl')
	# do_test_with_fe_pos('my_no_fe_gs_model_2.pkl', 'ohecols_fe.pkl')

	#
	# Testing against model found - END
	#

	#
	# Testing against model found - START
	#
	do_findBestModel_GS()
	do_test_with_fe_neg('mygs_fe_model_2.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_pos('mygs_fe_model_2.pkl', 'ohecols_fe.pkl')

	#
	# Testing against model found - END
	#

	#
	# Modelling - END
	#

	'''
	do_test_with_fe_neg('mygs_fe_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_pos('mygs_fe_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_neg('my_no_fe_gs_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_pos('my_no_fe_gs_model.pkl', 'ohecols_fe.pkl')
	'''
