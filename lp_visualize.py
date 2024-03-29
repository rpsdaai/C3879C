import pandas as pd
import numpy as np  # For mathematical calculations
import seaborn as sns  # For data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import logging

import warnings  # To ignore any warnings warnings.filterwarnings("ignore")

import lp_eda

from matplotlib import rcParams
# Ref: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
rcParams.update({'figure.autolayout': True})

warnings.filterwarnings("ignore")
log = logging.getLogger()

# Draw labels above the bar plots
# Ref: https://matplotlib.org/examples/api/barchart_demo.html
# Ref: https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot
# def autolabel(rects, ax):
def autolabel(ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    #for rect in rects:
    for rect in ax.patches:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        # if p_height > 0.95: arbitrary; 95% looked good to me.
        if p_height > 0.90:
            # label_position = height - (y_height * 0.05)
            label_position = height - (y_height * 0.1)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                # '%d' % int(height),
                '%.2f' % height,
                ha='center', va='bottom', color='red', fontweight='bold')

#
# UNIVARIATE Data Analysis, returns count of categorical values (both normalized and un-normalized)
#
def do_countCategoricalValues(df, column):
	vc = df[column].value_counts()
	nvc = df[column].value_counts(normalize=True)
	return vc, nvc


# UNIVARIATE Categorical Data Analysis
#
# plot loan_status side by side - one normalized and the other not
#
def do_plotCategorical(df, column, filename):
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.set_title('Un-Normalised')
	plt.subplots_adjust(wspace=0.5)  # set spacing between plots

	cp_cx = sns.countplot(df[column], ax=ax1)

	autolabel(cp_cx)

	ax2 = fig.add_subplot(122)
	ax2.set_title('Normalised')
	# train = pd.read_csv("data/train")
	train = df.copy()
	ser = train['Loan_Status'].value_counts(normalize=True)

	df1 = pd.DataFrame(ser).reset_index()
	df1.columns = ['Loan_Status', 'count']

	bp_cx = sns.barplot(df1['Loan_Status'], df1['count'])
	autolabel(bp_cx)

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


#
# UNIVARIATE Categorical Data Analysis - un-normalized
#
def do_plotCategorical_Notnormalised(df, columns, filename):
	#fig = plt.figure(figsize=(6, 6))
	fig = plt.figure()
	#plt.subplots_adjust(wspace=0.5, hspace=0.5)
	# Ref: https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot/28132929
	fig.suptitle('Un-normalized Plots', y = 1.0)  # or plt.suptitle('Main title')
	# fig = plt.figure(1)
	for i in range(len(columns)):
		log.info('columns[' + str(i) + ']' + columns[i])
		ax_tmp = fig.add_subplot(2, 2, i + 1)
		plt.xlabel(columns[i])  # Must come after add_subplot otherwise x-axis screwed up
		cp_ax = sns.countplot(df[columns[i]], ax=ax_tmp)

		autolabel(cp_ax)

	plt.tight_layout()
	fig.savefig(filename)
	plt.show()
	plt.close(fig)


#
# UNIVARIATE Categorical Data Analysis - normalized
#
def do_plotCategorical_Normalised(norm_val_count_series, column, filename):
	fig = plt.figure()

	# Ref: https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot/28132929
	fig.suptitle('Normalized Plots: ' + column + ' vs. Count', y = 1.0)
	df1 = pd.DataFrame(norm_val_count_series).reset_index()
	df1.columns = [column, 'Count']
	bp_ax = sns.barplot(df1[column], df1['Count'])

	autolabel(bp_ax)

	fig.savefig(filename)
	plt.show()
	plt.close(fig)

#
# UNIVARIATE Numerical Data Analysis e.g. ApplicantIncome, CoapplicantIncome, LoanAmount
#
def do_plotNumericalDataDistribution_OneFeature(df, column, filename):
	fig = plt.figure()
	# fig = plt.figure(figsize=(11, 8))
	ax1 = plt.subplot(121)
	plt.subplots_adjust(wspace=0.5)  # set spacing between plots
	ax1.set_title(column + ' distribution plot')
	sns.distplot(df[column])

	ax2 = plt.subplot(122)
	ax2.set_title(column + ' box plot')
	sns.boxplot(y=df[column])

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


#
# UNIVARIATE Numerical Data Analysis e.g. Graduate, NonGraduate vs ApplicantIncome
#
def do_plotDataNumericalDistribution_2Features(df, column1, column2, filename):
	fig = plt.figure(1)
	sns.set(style="whitegrid")
	sns.boxplot(x=df[column1], y=df[column2])
	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# To FIX!!! the titles
# UNIVARIATE Ordinal Data Analysis: With ordinal scales, the order of the values is what’s important and significant,
# but the differences between each one is not really known. Ordinal scales are typically measures of non-numeric
# concepts like satisfaction, happiness, discomfort, etc.
# Ref: https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
# To FIX!!! the titles
def do_plotOrdinals(vclist_series, nvclist_series, columns, filename):
	fig = plt.figure(figsize=(20, 10))
	#fig = plt.figure()
	# Ref: https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot/28132929
	fig.suptitle('Ordinal Plots', y = 1.0)  # or plt.suptitle('Main title')
	log.debug('COL Length: ' + str(len(columns)))
	plt.subplots_adjust(hspace=0.5)
	for i in range(len(columns)):
		ax = plt.subplot(2, 3, i + 1)
		df1 = pd.DataFrame(vclist_series[i]).reset_index()
		df1.columns = [columns[i], 'Count']
		# sns.barplot(df1['gender'], df1['count'])
		bp_ax = sns.barplot(df1[columns[i]], df1['Count'])

		autolabel(bp_ax)
	# plt.suptitle('Ordinal Normalized Plots')
	for i in range(len(columns)):
		ax = plt.subplot(2, 3, i + 4)
		df1 = pd.DataFrame(nvclist_series[i]).reset_index()
		df1.columns = [columns[i], 'Count']
		# sns.barplot(df1['gender'], df1['count'])
		bp_ax = sns.barplot(df1[columns[i]], df1['Count'])

		autolabel(bp_ax)

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


#
# BIVARIATE Analysis
#
def do_crossTab(df, column1, column2):
	# Ref: http://www.datasciencemadesimple.com/cross-tab-cross-table-python-pandas/
	# Ref: https://pbpython.com/pandas-crosstab.html
	# The margins keyword instructed pandas to add a total for each row as well as a total at the bottom.
	# I also passed a value to margins_name in the function call because I wanted to label the results “Total”
	# instead of the default “All”.
	# gender_loanstatus = pd.crosstab(train['Gender'], train['Loan_Status'], margins=True, margins_name="Total")

	# Ref: https://stackoverflow.com/questions/21247203/how-to-make-a-pandas-crosstab-with-percentages
	ct_df = pd.crosstab(df[column1], df[column2], normalize='index')

	stacked_data = ct_df.stack().reset_index().rename(columns={0: 'value'})

	# Ref: https://stackoverflow.com/questions/43544694/using-pandas-crosstab-with-seaborn-stacked-barplots
	# Plot grouped bar char
	bp_ax = sns.barplot(x=stacked_data[column1], y=stacked_data.value, hue=stacked_data[column2])
	autolabel(bp_ax)

	plt.show()


# BIVARIATE Categorical Analysis
def do_multiple_XTab_GroupedBarPlot(df, nrows, ncols, columns, column2, filename):
	#fig = plt.figure(figsize=(11, 8))
	fig = plt.figure()
	fig.suptitle('Categorical Bivariate Data Analysis', y = 1.0)  # or plt.suptitle('Main title')
	for i in range(len(columns)):
		ct_df = pd.crosstab(df[columns[i]], df[column2], normalize='index')
		stacked_data = ct_df.stack().reset_index().rename(columns={0: 'value'})
		ax_ = fig.add_subplot(nrows, ncols, i + 1)
		bp_ax = sns.barplot(x=stacked_data[columns[i]], y=stacked_data.value, hue=stacked_data[column2], ax=ax_)
		autolabel(bp_ax)

		# Ref: https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot/34579525
		# Legend blocking original graph, move it outside
		# resize figure box to -> put the legend out of the figure
		###
		box = bp_ax.get_position() # get position of figure
		bp_ax.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

		# Put a legend to the right side
		bp_ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
		###

		plt.subplots_adjust(wspace=0.5, hspace=0.5)
		plt.xticks(rotation=60)
		plt.xlabel(columns[i])
		plt.ylabel('Percentage')

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# BIVARIATE Categorical Analysis
def do_multiple_XTab_Stacked(df, nrows, ncols, columns, column2, filename):
	#fig = plt.figure(figsize=(11, 8))
	fig = plt.figure()
	fig.suptitle('Categorical Bivariate Data Analysis', y = 1.0)  # or plt.suptitle('Main title')
	for i in range(len(columns)):
		ax_ = fig.add_subplot(nrows, ncols, i + 1)

		ct_df = pd.crosstab(df[columns[i]], df[column2], normalize='index')
		ct_df.plot(kind='bar', stacked=True, ax=ax_)

		plt.subplots_adjust(wspace=0.5, hspace=0.5)
		plt.xticks(rotation=60)
		plt.xlabel(columns[i])
		plt.ylabel('Percentage')

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# BIVARIATE Numerical Analysis
def do_multiple_XCut_GroupedBarPlot(df, nrows, ncols, columns, column2, groupByBins, groupByCategories, filename):
	#fig = plt.figure(figsize=(11, 8))
	fig = plt.figure()
	fig.suptitle('Numerical Bivariate Data Analysis', y = 1.0)  # or plt.suptitle('Main title')
	for i in range(len(columns)):
		# Ref: https://stackoverflow.com/questions/45751390/pandas-how-to-use-pd-cut
		cut_df = pd.cut(df[columns[i]], groupByBins, labels=groupByCategories)
		ct_df = pd.crosstab(cut_df, df[column2], normalize='index')
		# now stack and reset
		stacked_data = ct_df.stack().reset_index().rename(columns={0: 'value'})

		ax_ = fig.add_subplot(nrows, ncols, i + 1)
		bp_ax = sns.barplot(x=stacked_data[columns[i]], y=stacked_data.value, hue=stacked_data[column2], ax=ax_)
		autolabel(bp_ax)

		# Ref: https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot/34579525
		# Legend blocking original graph, move it outside
		# resize figure box to -> put the legend out of the figure
		###
		box = bp_ax.get_position() # get position of figure
		bp_ax.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

		# Put a legend to the right side
		bp_ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
		###

		plt.subplots_adjust(wspace=0.5, hspace=0.5)
		plt.xticks(rotation=60)
		plt.xlabel(columns[i])
		plt.ylabel('Percentage')

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# BIVARIATE Numerical Analysis
def do_multiple_XCut_Stacked(df, nrows, ncols, columns, column2, groupByBins, groupByCategories, filename):
	#fig = plt.figure(figsize=(11, 8))
	fig = plt.figure()
	fig.suptitle('Numerical Bivariate Data Analysis', y = 1.0)  # or plt.suptitle('Main title')
	for i in range(len(columns)):
		ax_ = fig.add_subplot(nrows, ncols, i + 1)

		# Ref: https://stackoverflow.com/questions/45751390/pandas-how-to-use-pd-cut
		cut_df = pd.cut(df[columns[i]], groupByBins, labels=groupByCategories)
		ct_df = pd.crosstab(cut_df, df[column2], normalize='index')
		ct_df.plot(kind='bar', stacked=True, ax=ax_)

		plt.subplots_adjust(wspace=0.5, hspace=0.5)
		plt.xticks(rotation=60)
		plt.xlabel(columns[i])
		plt.ylabel('Percentage')

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# BIVARIATE Numerical Analysis
def do_grpByPlot(df, Xaxis_col, Yaxis_col, filename):
	# fig = plt.figure(figsize=(20, 10))
	fig = plt.figure()
	fig.suptitle('Numerical Bivariate Data Analysis: ' + Xaxis_col + ' vs. ' + Yaxis_col, y = 1.0)

	# df.groupby(Xaxis_col)[Yaxis_col].mean().plot.bar()
	tmp = df.groupby(Xaxis_col)[Yaxis_col].mean()
	log.info(type(tmp))
	# Seaborn equivalent
	bp = sns.barplot(x=tmp.index, y=tmp.values)

	# Annotate the barplot
	# Ref: https://github.com/mwaskom/seaborn/issues/1582
	for p in bp.patches:
		bp.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
		            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

	fig.savefig(filename)
	plt.show()
	plt.close(fig)


# BIVARIATE Numerical Analysis
# Plots the correlation between ALL numerical variables. The darker the oolor the higher the correlation
def do_correlationPlot(df, filename):
	matrix = df.corr()

	fig, ax_ = plt.subplots(figsize=(9, 6))
	fig.suptitle('Correlation Plot', y = 1.0)
	sns.heatmap(matrix, vmax=.8, square=True, cmap='BuPu', ax=ax_)

	plt.xticks(rotation=60)

	fig.savefig(filename)
	plt.show()
	plt.close(fig)
