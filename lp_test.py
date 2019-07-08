import pandas as pd
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

import lp_eda

log = logging.getLogger()

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

if __name__ == '__main__':
	do_test_with_fe_neg('mygs_fe_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_pos('mygs_fe_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_neg('my_no_fe_gs_model.pkl', 'ohecols_fe.pkl')
	do_test_with_fe_pos('my_no_fe_gs_model.pkl', 'ohecols_fe.pkl')

