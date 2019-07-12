# Ref: https://www.tutorialspoint.com/flask/flask_url_building.htm
from flask import Flask, request, redirect, render_template, url_for

import random

# import numpy as np
import pandas as pd
# import sklearn
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import ColumnTransformer

import pickle
import ast
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import logging

import lp_eda

log = logging.getLogger()

# https://nearsoft.com/blog/how-to-create-an-api-and-web-applications-with-flask/
app = Flask(__name__)

# load pickled file containing ohe columns
# ohe_cols = lp_eda.do_loadModel("ohe_cols.pkl")
ohe_cols = lp_eda.do_loadModel("ohecols_fe.pkl")
# load the trained model
# model = do_loadModel("lr_no_gs.pkl")
model = lp_eda.do_loadModel("mygs_fe_model.pkl")

# NO Feature Engineering
# columns to scale; values are large compared to the other columns
# cols_2_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Feature Engineering
cols_2_scale = ['LoanAmount', 'Loan_Amount_Term']

def do_setType(df):
	log.debug('do_setType()')
	# using dictionary to convert specific columns
	# Ref: https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/
	convert_dict = {
        'ApplicantIncome': float,
		'CoapplicantIncome': float,
		'LoanAmount': float,
		'Loan_Amount_Term': int,
		'Credit_History': float
    }
	df = df.astype(convert_dict)
	log.debug('Converted TYPES: ' + str(df.dtypes))
	return df

@app.route('/success/<name>')
def success(name):
	#logging.info("success()")
	return 'welcome %s ' % name

@app.route('/')
def index():
	log.debug("--> index(): Entry point")
	return render_template('index.html')

# return "Web Service is running!"
@app.route('/webhook', methods=['POST'])
def webhook():
	log.debug("webhook(): test that web service is running")
	return "Web Service is running!"

# Ref: https://www.quora.com/How-do-I-run-python-flask-file-when-click-a-HTML-button
@app.route('/test_form_service', methods=['GET', 'POST'])
def test_form_service():
	log.debug("--> test_form_service()")
	if request.method == 'POST':
		log.debug("POST")
		fname = request.form['firstname']
		lname = request.form['lastname']
		gender = request.form['gender']
		maritalStatus = request.form['MaritalStatus']
		dependents = request.form["Dependents"]
		education = request.form['Education']
		employment = request.form['SelfEmployment']
		property = request.form.get("PropertyAreaList")
		applicantIncome = request.form.get("ApplicantIncome")
		coapplicantIncome = request.form.get("CoApplicantIncome")
		loanAmt = request.form.get("LoanAmount")
		loanTerm = request.form.get("LoanTerm")
	else:
		log.debug("GET")
		fname = request.args.get('firstname')
		lname = request.args.get('lastname')
		gender = request.args.get('gender')
		maritalStatus = request.args.get['MaritalStatus']
		dependents = request.args.get["Dependents"]
		education = request.args.get['Education']
		employment = request.args.get['SelfEmployment']
		property = request.args.get("PropertyAreaList")
		applicantIncome = request.args.get("ApplicantIncome")
		coapplicantIncome = request.args.get("CoApplicantIncome")
		loanAmt = request.args.get("LoanAmount")
		loanTerm = request.args.get("LoanTerm")
	# return redirect(url_for('success',name = fname+' '+lname))

	# Make a list of dictionary items from user input
	mydata = [
		{
			'Gender': gender,
			'Married': maritalStatus,
			'Dependents': dependents,
			'Education': education,
			'Self_Employed': employment,
			'ApplicantIncome': applicantIncome,
			'CoapplicantIncome': coapplicantIncome,
			'LoanAmount': loanAmt,
			'Loan_Amount_Term': loanTerm,
			'Credit_History': random.randint(0, 1),
			'Property_Area': property
		}
	]

	log.debug('DATA: ' + str(mydata))

	# return redirect(url_for('success', name=fname, data=mydata))
	# redirect to next url for loan prediction using trained model
	return redirect(url_for('loan_approval_service', name=fname+' '+lname, data=mydata))


@app.route('/loan_approval_service/<name>/<data>', methods=['GET', 'POST'])
def loan_approval_service(name, data):
	# Read back PICKLE file
	# loaded_model = pickle.load(open('skf_with_lr', 'rb'))
	# print(loaded_model.predict(x_test))
	# print(x_test.head())
	# result = loaded_model.score(x_test, y_test)
	# print(result)

	log.debug('loan_approval_service()' + str(type(data)))
	log.debug('Name: ' + name)
	log.debug('data contents: ' + data)

	# convert the string representation to a dict
	dict = ast.literal_eval(data)
	# and use it as the input
	df = pd.DataFrame(data=(dict))

	# ADDED for Feature Engineering
	# Have to set the types properly otherwise will get error when computing new feature engineered dataframes
	df = do_setType(df)

	log.debug('df = ' + df.to_string() + ' type = ' + str(type(df)))

	lp_eda.do_FeatureEngineering(df)

	# Do One Hot Encoding before scaling data
	df = pd.get_dummies(df).reindex(columns = ohe_cols, fill_value = 0)

	scaled_df = lp_eda.scale_data(df, RobustScaler(), cols_2_scale)
	log.debug('scaled_df: ' + scaled_df.to_string())

	log.debug('Running prediction ....')
	result = model.predict(scaled_df)
	log.debug('Loan Approved RESULTS: ' + str(result))

	if result[0] == 0:
		app_status = "Hello " + name + " Sorry, your Loan is DENIED"
	else:
		app_status = "Hello " + name + " Congratulations, your Loan is APPROVED"

	log.debug('<-- loan approval service')
	return app_status

if __name__ == '__main__':
	app.run(debug=True)
