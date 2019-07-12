import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

import warnings
# Ref: https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings/32389270
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.compose import ColumnTransformer
import pickle
import joblib

import sys
import logging

# Log to both console + file
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%Y-%m-%d %H:%M:%S',
                    handlers = [
	                    logging.FileHandler('loan_prediction.log', 'w', 'utf-8'),
	                    logging.StreamHandler(sys.stdout)
                    ])

log = logging.getLogger()

# ## Data Analysis Functions
def impute_mode(df, column):
	log.debug('--> impute_mode()' + column)
	df[column].fillna(df[column].mode()[0], inplace=True)


def impute_mean(df, column):
	log.debug('--> impute_mean()' + column)
	df[column].fillna(df[column].mean()[0], inplace=True)


def impute_median(df, column):
	log.debug('--> impute_median()' + column)
	df[column].fillna(df[column].median(), inplace=True)

def prepare_data(df):
	log.debug('--> prepare_data()')
	# There are empty cells in these columns based on above analysis. So replace these empty cells by modal values
	impute_mode(df, 'Gender')
	impute_mode(df, 'Married')
	impute_mode(df, 'Dependents')
	impute_mode(df, 'Self_Employed')
	impute_mode(df, 'Credit_History')

	# LoanAmount has outliers, so if we use the mean to fill in the missing values, results will be skewed
	# (affected by outliers)
	# We will use 'median' instead (middle value in list of numbers)
	impute_median(df, 'LoanAmount')
	impute_median(df, 'Loan_Amount_Term')


# Convert categorical Loan_status values of Y, N to numbers
def prepare_dependent_data(df, dep_cols):
	log.debug('--> prepare_dependent_data()' + dep_cols)
	# train['Loan_Status'].replace('N', 0, inplace = True)
	# train['Loan_Status'].replace('Y', 1, inplace = True)

	# Ref: https://stackoverflow.com/questions/4148375/is-there-an-in-place-equivalent-to-map-in-python?lq=1
	# Alternative is to use map. However no inplace
	df[dep_cols] = df[dep_cols].map({'N': 0, 'Y': 1})


# Example, drop 'Loan_ID' variable as it does not have any effect on the loan status
def remove_useless_data(df, col_2_remove):
	log.debug('--> remove_useless_data()' + str(col_2_remove))
	# Ref: https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe-by-column-name
	# axis = 0 for rows and axis = 1 for columns
	df.drop(col_2_remove, axis=1, inplace=True)


def split_dataset(df, dependent_col):
	log.debug('--> split_dataset()' + dependent_col)
	# Sklearn requires target variable in separate dataset. Therefore, we drop our target variable from
	# train dataset & save in another dataset
	X = df.drop(dependent_col, axis=1)
	y = df[dependent_col]

	return X, y

# Feature Engineering
def do_FeatureEngineering(df):
	log.debug('--> lp_eda: do_FeatureEngineering()')

	# New Feature: Total Income
	df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
	df['Total_Income_log'] = np.log(df['Total_Income'])

	# EMI is the monthly amount to be paid by the applicant to repay the loan
	# Calculated by taking the ratio of loan amount with respect to loan amount term.
	df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
	df['EMI_log'] = np.log(df['EMI'])
	# EMI_log has null values
	impute_median(df, 'EMI_log')
	# Income left after the EMI has been paid. If this value is high, the chances are high that a person
	# will repay the loan and hence increasing the chances of loan approval
	# Multiply by 1000 to make units equal
	df['Balance_Income'] = df['Total_Income']  - (df['EMI']*1000)
	df['Balance_Income_log'] = np.log(df['Balance_Income'])
	# Balance_Income_log has null values
	impute_median(df, 'Balance_Income_log')

	cols2drop = ['ApplicantIncome', 'CoapplicantIncome', 'Total_Income', 'Balance_Income', 'EMI']
	remove_useless_data(df, cols2drop)

def do_EDA(df, oheFlag, feFlag):
	log.debug('--> do_EDA()')

	if feFlag == True:
		do_FeatureEngineering(df)

	prepare_data(df)

	prepare_dependent_data(df, 'Loan_Status')

	remove_useless_data(df, 'Loan_ID')

	X, y = split_dataset(df, 'Loan_Status')

	if oheFlag == True:
		# do ONE HOT ENCODING
		X = pd.get_dummies(X)

	return X, y


# df contains df will full features already converted to numbers (excl. Loan_ID, Loan_Status) - 11 cols
def scale_data(df, scale_fn, cols_2_scale):
	log.debug('--> scale_data(): ' + ' , '.join(cols_2_scale))
	# Ref: https://stackoverflow.com/questions/38420847/apply-standardscaler-on-a-partial-part-of-a-data-set

	'''
	scaler = MinMaxScaler()
	col_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
				 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
				 'Credit_History', 'Property_Area']
	features = X[col_names]
	'''
	ct = ColumnTransformer([
		('scaler', scale_fn, cols_2_scale)
	], remainder='passthrough')

	# Ref: https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
	# Ref: https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
	log.debug('scale_data(): df.index')
	log.debug(df.columns)
	tmp_df = pd.DataFrame(ct.fit_transform(df), index=df.index, columns=df.columns)
	# log.debug('ColumnTransformer: ')
	# log.debug(tmp_df['Property_Area'].head())
	return (tmp_df)


# Save model to disk
def do_saveModel(filename, model, which_library):
	log.debug('--> do_saveModel(): ' + filename + ' library to use: ' + which_library)
	with open(filename, 'wb') as f:
		if which_library == 'p':
			pickle.dump(model, f)
		else:
			joblib.dump(model, f)


def do_loadModel(filename):
	log.debug('--> do_loadModel(): ' + filename)
	with open(filename, 'rb') as f:
		model = pickle.load(f)
	return (model)

# oheFlag=False, scaleFlag=False, feFlag=True
def do_processDataset(oheFlag, scaleFlag, feFlag):
	log.debug('--> do_processDataset()')

	train = pd.read_csv("data/train.csv")

	X, y = do_EDA(train, oheFlag, feFlag)

	# Dont scale it before doing feature engineering
	if scaleFlag == True:
		if feFlag == True:
			# After including feature engineering the following columns have been dropped: ApplicantIncome',
			# 'CoapplicantIncome'and replaced by 'Total_income'
			X = scale_data(X, RobustScaler(), ['LoanAmount', 'Loan_Amount_Term'])
		else:
			X = scale_data(X, RobustScaler(), ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])

	return X, y

# def main():
#    X, y = do_processDataset()

if __name__ == '__main__':
	log.debug('--> Starting ....')
	# oheFlag=False, scaleFlag=False, feFlag=True
	# X, y = do_processDataset(True, False, True)
	X, y = do_processDataset(False, False, True)
	log.debug(X.columns)
	log.debug(X['Property_Area'].head())
	log.debug(X.isnull().sum())

	'''
	do_saveModel('ohecols_fe.pkl', X.columns, 'p')
	contents = do_loadModel('ohecols_fe.pkl')
	log.debug('Read back ohecols_fe.pkl: ' + contents)
	'''
# xs_train, xs_test, ys_train, ys_test = train_test_split(X, y, test_size = 0.2)
