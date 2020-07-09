from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import patsy
from patsy import dmatrices
import statsmodels.api as sm

from sklearn.metrics import precision_recall_curve,confusion_matrix,mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, make_scorer, roc_curve, roc_auc_score, log_loss
from sklearn.linear_model import LinearRegression, Ridge #ordinary linear regression + w/ ridge regularization
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV, Ridge, RidgeCV 
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split,learning_curve,cross_val_score, cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.plotting import plot_decision_regions


def drop_columns(names, dataframe):
	for name in names:
		dataframe.drop(name,axis=1,inplace=True)


def precision_recall_scores (why_predict, ex_test, why_test, model, threshold):
	why_predict = (model.predict_proba(ex_test)[:,1] > threshold)
	print("Threshold of "+ str(threshold) +":")
	print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(why_test, why_predict),recall_score(why_test, why_predict)))


# def precision_recall_graph (model, why_test, ex_test):
# 	precision_curve, recall_curve, threshold_curve = precision_recall_curve(why_test, model.predict_proba(ex_test)[:,1])
# 	plt.figure(dpi=80)
# 	plt.plot(threshold_curve, precision_curve[1:],label='precision')
# 	plt.plot(threshold_curve, recall_curve[1:], label='recall')
# 	plt.legend(loc='lower left')
# 	plt.xlabel('Threshold (above this probability, label as "good wine"")');
# 	plt.title('Precision and Recall Curves');


# def make_confusion_matrix(why_predict, ex_test, why_test, model, threshold):
# 	# Predict class 1 if probability of being in class 1 is greater than threshold
# 	# (model.predict(X_test) does this automatically with a threshold of 0.5)
# 	why_predict = (model.predict_proba(ex_test)[:, 1] >= threshold)
# 	wine_confusion = confusion_matrix(why_test, why_predict)
# 	plt.figure(dpi=80)
# 	sns.heatmap(wine_confusion, cmap=plt.cm.Reds, annot=True, square=True, fmt='d',xticklabels=['meh wine', 'good wine'], yticklabels=['meh wine', 'good wine']);
# 	plt.xlabel('prediction')
# 	plt.ylabel('actual')


# def print_roc_auc (ex_test, why_test, model):
# 	model_auc = roc_auc_score(y_score = model.predict_proba(ex_test)[:,1],y_true = why_test)
# 	print (f'ROC/AUC score for {model}: {model_auc}')