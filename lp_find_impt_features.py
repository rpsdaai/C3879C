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
	# FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
	# if no argument supplied to ExtraTreesClassifier()
	model = ExtraTreesClassifier(n_estimators=100)
	model.fit(X, y)
	log.debug(model.feature_importances_)
	return (model.feature_importances_)

# Ref: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
def do_horizontalLabelPlacement(ax):
	# For each bar: Place a label
	for rect in ax.patches:
	    # Get X and Y placement of label from rect.
	    x_value = rect.get_width()
	    y_value = rect.get_y() + rect.get_height() / 2

	    # Number of points between bar and label. Change to your liking.
	    space = 5
	    # Vertical alignment for positive values
	    ha = 'left'

	    # If value of bar is negative: Place label left of bar
	    if x_value < 0:
	        # Invert space to place label to the left
	        space *= -1
	        # Horizontally align label at right
	        ha = 'right'

	    # Use X value as label and format number with one decimal place
	    label = "{:.3f}".format(x_value)

	    # Create annotation
	    plt.annotate(
	        label,                      # Use `label` as label
	        (x_value, y_value),         # Place label at end of the bar
	        xytext=(space, 0),          # Horizontally shift label by `space`
	        textcoords="offset points", # Interpret `xytext` as offset in points
	        va='center',                # Vertically center label
	        ha=ha)                      # Horizontally align label differently for
	                                    # positive and negative values.

def do_plotImportantFeatures(featureImportances, X, filename):
	fig = plt.figure()
	fig.suptitle('Features Ranking', y = 1.0)

	# Ref: https://stackoverflow.com/questions/18973404/setting-different-bar-color-in-matplotlib-python
	# Ref: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
	my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c',
	             'peru', 'tomato', 'maroon', 'darkgoldenrod', 'olive', 'aqua', 'deepskyblue',
	             'blueviolet', 'crimson', 'orange', 'mediumspringgreen', 'steelblue', 'fuchsia', 'deeppink']  # red, green, blue, black, etc.

	importances=pd.Series(featureImportances, index=X.columns)
	bh_ax = importances.plot(kind='barh', figsize=(12,8), color=my_colors)

	# Ref: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
	do_horizontalLabelPlacement(bh_ax)

	plt.tight_layout()
	fig.savefig(filename)
	plt.show()
	plt.close(fig)
