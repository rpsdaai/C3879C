import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

import warnings
# Ref: https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings/32389270
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import logging

import lp_eda

log = logging.getLogger()

def swlib_versions():
	log.debug('--> swlib_versions()')
	log.debug('Matplotlib Version: ' + format(mpl.__version__))
	log.debug('Seaborn Version: ' + format(sns.__version__))
	log.debug('Numpy Version: ' + format(np.__version__))
	log.debug('Pandas Version: ' + format(pd.__version__))

# !!!!Feature Engineering - Repeated in lp_eda.py
def do_FeatureEngineering(df):
	log.debug('--> do_FeatureEngineering()')
	# New Feature: Total Income
	df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
	log.debug('Total_Income: ' + str(df['Total_Income']))
	# log.debug('Total_Income: ' + str(df['Total_Income']))
	log.debug('Total_Income_log: ' + str(np.log(df['Total_Income'])))
	df['Total_Income_log'] = np.log(df['Total_Income'])

	# EMI is the monthly amount to be paid by the applicant to repay the loan
	# Calculated by taking the ratio of loan amount with respect to loan amount term.
	df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
	df['EMI_log'] = np.log(df['EMI'])
	# Income left after the EMI has been paid. If this value is high, the chances are high that a person
	# will repay the loan and hence increasing the chances of loan approval
	# Multiply by 1000 to make units equal
	df['Balance_Income'] = df['Total_Income']  - (df['EMI']*1000)
	df['Balance_Income_log'] = np.log(df['Balance_Income'])
	# Balance_Income_log has null values
	lp_eda.impute_median(df, 'Balance_Income_log')

def do_plot(df, col, filename):
	log.debug('--> do_plot()' + filename)
	ax_ = sns.distplot(df[col])
	figure_ = ax_.get_figure()
	figure_.savefig(filename)
	plt.close(figure_)

def do_findCorrelation(df, filename, cols_2_drop):
	log.debug('--> do_findCorrelation(): ' + ' , '.join(cols_2_drop))
	corr_df = df.drop(cols_2_drop, axis = 1)
	log.debug('DF after dropping: ')
	log.debug(corr_df.columns)
	correlationMatrix = corr_df.corr()
	#log.debug('Correlation Matrix Shape: ' + correlationMatrix.shape)
	ax = plt.subplots(figsize = (11, 8)) # width = 10, height = 8 inches
	plt.xticks(rotation = 60)
	ax_ = sns.heatmap(correlationMatrix, vmax = .8, square = True, cmap = 'coolwarm')
	figure_ = ax_.get_figure()
	plt.tight_layout()
	figure_.savefig(filename)
	plt.close(figure_)

def do_plotNewFeatures(df):
	log.debug('--> do_plotNewFeatures()')
	do_plot(df, 'Total_Income', 'total_income.png')
	do_plot(df, 'Total_Income_log', 'total_income_log.png')
	do_plot(df, 'EMI', 'EMI.png')
	do_plot(df, 'EMI_log', 'EMI_log.png')
	do_plot(df, 'Balance_Income', 'Balance_Income.png')
	do_plot(df, 'Balance_Income_log', 'Balance_Income_log.png')

# cols2drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'EMI_log']
def do_dropAfterFeatureEngineering(df, cols2drop):
	log.debug('--> do_dropAfterFeatureEngineering()')
	# axis = 1 -> column
	df.drop(cols2drop, axis = 1, inplace=True)

# Ref: https://stackoverflow.com/questions/46927545/get-feature-names-of-selectkbest-function-python
def do_findBestFeaturesUsingChi2(X, y):
	chi2_selector = SelectKBest(score_func=f_classif, k=5)
	chi2_selector.fit(X, y)

	X_new = chi2_selector.transform(X)
	log.debug(X_new.shape)

	log.debug(X.columns[chi2_selector.get_support(indices = True)])

	# 1st way to get the list
	vector_names = list(X.columns[chi2_selector.get_support(indices = True)])
	return (vector_names)

	# 2nd way
	# X.columns[chi2_selector.get_support(indices = True)].tolist()

# Ref: https://machinelearningmastery.com/feature-selection-machine-learning-python/
def do_findBestFeaturesUsingXtraTreeClassifier(X, y):
	# feature extraction
	# model = ExtraTreesClassifier(max_depth=3, n_estimators=41)
	model = ExtraTreesClassifier()
	model.fit(X, y)
	log.debug(model.feature_importances_)
	return (model.feature_importances_)

def do_plotImportantFeatures(featureImportances, X, filename):
	fig = plt.figure()
	fig.suptitle('Features Ranking', y = 1.0)

	importances=pd.Series(featureImportances, index=X.columns)
	importances.plot(kind='barh', figsize=(12,8))

	plt.tight_layout()
	fig.savefig(filename)
	plt.show()
	plt.close(fig)

#train.columns
# cols = ['Credit_History', 'LoanAmount_Log', 'Total_Income', 'EMI', 'Education_Graduate']
#sns.pairplot(train, height=2.5)
#sns.pairplot(data=train,
#             y_vars=['Credit_History', 'LoanAmount_Log', 'Total_Income', 'EMI', 'Education_Graduate'],
#             x_vars=['Loan_Status'])
if __name__ == '__main__':
	swlib_versions()
	# oheFlag=False, scaleFlag=False, feFlag=True
	# X, y = lp_eda.do_processDataset(oheFlag=True, scaleFlag=False)
	X, y = lp_eda.do_processDataset(True, False, False)
	log.debug(X.columns)

	log.debug(X.columns)
	do_FeatureEngineering(X)
	do_plotNewFeatures(X)

	# cols2drop = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Balance_Income', 'EMI']

	cols2drop = ['Total_Income_log', 'Balance_Income_log', 'EMI_log']
	cols2Scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Balance_Income', 'EMI']

	Xcopy = X.copy()

	xcopy_cols2drop = ['Total_Income_log', 'Balance_Income_log', 'EMI_log', 'ApplicantIncome', 'CoapplicantIncome']
	Xcopy.drop(xcopy_cols2drop, axis = 1, inplace=True)

	X = lp_eda.scale_data(X, RobustScaler(), cols2Scale)

	log.debug(X.head())
	do_findCorrelation(X, 'lp_heatmap.png', cols2drop)

	# do_findBestFeaturesUsingXtraTreeClassifier(X, y)
	# dont scale otherwise important features comes out wrong

	importances = do_findBestFeaturesUsingXtraTreeClassifier(Xcopy, y)
	do_plotImportantFeatures(importances, Xcopy, 'rank_features.png')

