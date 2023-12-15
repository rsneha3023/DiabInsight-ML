#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:49:18 2023

@author: sneharavi
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (confusion_matrix, accuracy_score)
import dmba as dmba
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from xgboost import XGBClassifier



os.chdir(r'/Users/sneharavi/Desktop/Quantitaive_Methods/Final Project')
os.getcwd()

#https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
df = pd.read_csv('diabetes_data.csv')
df.isna().sum()
df.info()

sns.histplot(data=df, x='Age',palette="dark")
plt.title('Distribution of Age')

%matplotlib inline
sns.countplot(data=df, x='Gender')

# Male = 1, Female = 0
gender = LabelEncoder()
df['Gender'] = gender.fit_transform(df['Gender'])

# Yes = 1, No = 0
polyuria = LabelEncoder()
df['Polyuria']= polyuria.fit_transform(df['Polyuria'])

# Yes = 1, No = 0
polydipsia = LabelEncoder()
df['Polydipsia']= polydipsia.fit_transform(df['Polydipsia'])

# Yes = 1, No = 0
df.rename(columns = {'sudden weight loss':'sudden_weight_loss'}, inplace= True)
sudden_weight_loss = LabelEncoder()
df['sudden_weight_loss']= sudden_weight_loss.fit_transform(df['sudden_weight_loss'])

# Yes = 1, No = 0
weakness = LabelEncoder()
df['weakness']= weakness.fit_transform(df['weakness'])

# Yes = 1, No = 0
polyphagia = LabelEncoder()
df['Polyphagia']= polyphagia.fit_transform(df['Polyphagia'])

# Yes = 1, No = 0
df.rename(columns = {'Genital thrush':'Genital_thrush'}, inplace= True)
genital_thrush = LabelEncoder()
df['Genital_thrush']= genital_thrush.fit_transform(df['Genital_thrush'])

# Yes = 1, No = 0
df.rename(columns = {'visual blurring':'visual_blurring'}, inplace= True)
visual_blurring = LabelEncoder()
df['visual_blurring']= visual_blurring.fit_transform(df['visual_blurring'])

# Yes = 1, No = 0
Itching = LabelEncoder()
df['Itching']= Itching.fit_transform(df['Itching'])

# Yes = 1, No = 0
Irritability = LabelEncoder()
df['Irritability']= Irritability.fit_transform(df['Irritability'])

# Yes = 1, No = 0
df.rename(columns = {'delayed healing':'delayed_healing'}, inplace= True)
delayed_healing = LabelEncoder()
df['delayed_healing']= delayed_healing.fit_transform(df['delayed_healing'])

# Yes = 1, No = 0
df.rename(columns = {'partial paresis':'partial_paresis'}, inplace= True)
partial_paresis = LabelEncoder()
df['partial_paresis']= partial_paresis.fit_transform(df['partial_paresis'])

# Yes = 1, No = 0
df.rename(columns = {'muscle stiffness':'muscle_stiffness'}, inplace= True)
muscle_stiffness = LabelEncoder()
df['muscle_stiffness']= muscle_stiffness.fit_transform(df['muscle_stiffness'])

# Yes = 1, No = 0
alopecia = LabelEncoder()
df['Alopecia']= alopecia.fit_transform(df['Alopecia'])

# Yes = 1, No = 0
obesity = LabelEncoder()
df['Obesity']= obesity.fit_transform(df['Obesity'])

# Yes = 1, No = 0
df.rename(columns = {'class':'Diabetes'}, inplace= True)
output = LabelEncoder()
df['Diabetes']= output.fit_transform(df['Diabetes'])

X = df.drop(columns='Diabetes')
Y = df['Diabetes']
X.columns
df.columns

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=100 )

print('X Train Shape:',X_train.shape) 
print('Y Train Shape:',y_train.shape) 
print('X Test Shape:',X_test.shape)
print('Y Test Shape:',y_test.shape)

#-------------------------------------------------------------------------------------------------------------------------------------

# Random Forest
model_gini = RandomForestClassifier(n_estimators=35, criterion='gini', max_depth=5, oob_score=True)
model_gini.fit(X_train, y_train)
model_gini.score(X_train, y_train)
model_gini.score(X_test, y_test)

print('Random Forest Gini Model Training Accuracy Score:' ,round(model_gini.score(X_train, y_train)*100,3))
print('Random Forest Gini Model Test Accuracy Score:', round(model_gini.score(X_test, y_test)*100,3))

# RandomForest: Accuracy
pred_gini = model_gini.predict(X_test)
pred_gini

cm_gini = confusion_matrix(y_test, pred_gini)
cm_gini

print('Random Forest Gini Model Training Accuracy Score:' ,round(model_gini.score(X_train, y_train)*100,3))
print('Random Forest Gini Model Test Accuracy Score:', round(model_gini.score(X_test, y_test)*100,3))
print('Precision', round((cm_gini[0, 0] / sum(cm_gini[:, 0]))*100,3))
print('Recall', round((cm_gini[0, 0] / sum(cm_gini[0, :]))*100,3))
print('Specificity', round((cm_gini[1, 1] / sum(cm_gini[1, :]))*100,3))

confusion_gini = dmba.classificationSummary(y_test, pred_gini, class_names=model_gini.classes_)
labels = ([['True Negative: %s'%cm_gini[0,0], 'False Positive: %s'%cm_gini[0,1]],['False Negative: %s'%cm_gini[1,0], 'True Positive: %s'%cm_gini[1,1]]])
fig, ax = plt.subplots()
sns.heatmap(cm_gini, annot= labels, fmt = '', cmap= 'Blues')
plt.title('Confusion Matrix: Random Forest Classifier (Gini)')

pred_prob_gin = model_gini.predict(X_test)
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
auc_score = roc_auc_score(y_test, pred_gini)

# Plot the ROC curve
plt.title('Random Forest Gini Model ROC Curve')
plt.plot(fpr, tpr, color='purple', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Recall (True Positive Rate)')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.legend(loc='lower right')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------

# Feature Importance Gini
X = df.drop(columns='Diabetes')
Y = df['Diabetes']
scores_gini = defaultdict(list)
for _ in range(3):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    model_gini = RandomForestClassifier(criterion='gini')
    model_gini.fit(x_train, y_train)
    acc = metrics.accuracy_score(y_test, model_gini.predict(x_test))
    for column in X.columns:
        X_t = x_test.copy()
        X_t[column] = np.random.permutation(X_t[column].values)
        shuff_acc = metrics.accuracy_score(y_test, model_gini.predict(X_t))
        scores_gini[column].append((acc - shuff_acc) / acc)

print('Gini Features sorted by their score:')
print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores_gini.items()], reverse=True))

importances_gini = model_gini.feature_importances_

df_gini = pd.DataFrame({'feature': X.columns, 'Accuracy decrease': [np.mean(scores_gini[column]) for column in X.columns], 'Gini decrease': importances_gini})

# Sorting DataFrames based on Accuracy Decrease
df_gini = df_gini.sort_values('Accuracy decrease', ascending=False)

# Plotting the Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(df_gini['feature'], df_gini['Gini decrease'], color='lightblue')
plt.xlabel('Gini Importance')
plt.ylabel('Features')
plt.title('Gini Feature Importance for Early Diabetes Detection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------

# PCA

sp_pca = PCA()  
sp_pca.fit(df)
sp_pca.components_
df_components_ = pd.DataFrame(sp_pca.components_,columns=df.columns,index = ['PC-'+str(i) for i in range(sp_pca.n_components_)])
sp_pca.explained_variance_
sp_pca.explained_variance_ratio_
explained_variance = pd.DataFrame(sp_pca.explained_variance_ratio_)
explained_variance.plot()
cum_sum_explained_variance = np.cumsum(sp_pca.explained_variance_ratio_)
cum_sum_explained_variance
cum_sum_explained_variance_df = pd.DataFrame(cum_sum_explained_variance)
cum_sum_explained_variance_df.rename(columns = {'0':'Cumulative Exp_Variance'}, inplace= True)

num_of_steps = range(0,len(cum_sum_explained_variance_df))

%matplotlib inline
ax = explained_variance.head(20).plot.bar(legend=False, figsize=(12, 10))
ax.set_xlabel('Principal Component', size = 15)
ax.set_ylabel('Explained Variance Ratio', size=15)
plt.step(num_of_steps, cum_sum_explained_variance_df, where='mid',label='Cumulative explained variance', color='red')
plt.title('Explained Variance Ratio by Principal Components', size=20)
plt.show()


loadings = pd.DataFrame(sp_pca.components_[0:5, :], columns=df.columns)
print(loadings)
maxPC = 1.01 * loadings.loc[0:5, :].abs().to_numpy().max()    # this is for automatically setting the y-axis scale, so that you can always zoom into the principal components no matter how small their values are

f, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
for i, ax in enumerate(axes):
    pc_loadings = loadings.loc[i, :]
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    ax.axhline(color='#888888')
    pc_loadings.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'PC{i}')
    ax.set_ylim(-maxPC, maxPC)
plt.tight_layout()
plt.show()

# PCA
df1 = df.drop(columns='Age')
sp_pca = PCA(n_components=10)  
sp_pca.fit(df1)
sp_pca.components_
df_components_ = pd.DataFrame(sp_pca.components_,columns=df1.columns,index = ['PC-'+str(i) for i in range(sp_pca.n_components_)])
sp_pca.explained_variance_
sp_pca.explained_variance_ratio_
explained_variance = pd.DataFrame(sp_pca.explained_variance_ratio_)
explained_variance.plot()
cum_sum_explained_variance = np.cumsum(sp_pca.explained_variance_ratio_)
cum_sum_explained_variance
cum_sum_explained_variance_df = pd.DataFrame(cum_sum_explained_variance)
cum_sum_explained_variance_df.rename(columns = {'0':'Cumulative Exp_Variance'}, inplace= True)

num_of_steps = range(0,len(cum_sum_explained_variance_df))

%matplotlib inline
ax = explained_variance.head(20).plot.bar(legend=False, figsize=(12, 10))
ax.set_xlabel('Principal Component', size = 15)
ax.set_ylabel('Explained Variance Ratio', size=15)
plt.step(num_of_steps, cum_sum_explained_variance_df, where='mid',label='Cumulative explained variance', color='red')
plt.title('Explained Variance Ratio by Principal Components', size=20)
plt.show()


loadings = pd.DataFrame(sp_pca.components_[0:10, :], columns=df1.columns)
print(loadings)
maxPC = 1.01 * loadings.loc[0:10, :].abs().to_numpy().max()    # this is for automatically setting the y-axis scale, so that you can always zoom into the principal components no matter how small their values are

f, axes = plt.subplots(10, 1, figsize=(10, 10.5), sharex=True)
for i, ax in enumerate(axes):
    pc_loadings = loadings.loc[i, :]
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    ax.axhline(color='#888888')
    pc_loadings.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'PC{i}')
    ax.set_ylim(-maxPC, maxPC)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------

# XGBoost Classifier

X_train.info()
y_train.info()


xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, subsample = 1, max_depth = 2,eval_metric = 'error',  random_state=100)
xgb.fit(X_train, y_train)

print('XG Boost Model Training Accuracy Score:',round((xgb.score(X_train, y_train))*100,3))
print('XG Boost Model Testing Accuracy Score:',round((xgb.score(X_test, y_test))*100,3))

xgb_predict = xgb.predict(X_test)
xgb_predict

cm_xgb = confusion_matrix(y_test, xgb_predict)
cm_xgb

print('XG Boost Model Training Accuracy Score:',round((xgb.score(X_train, y_train))*100,3))
print('XG Boost Model Testing Accuracy Score:',round((xgb.score(X_test, y_test))*100,3))
print('Precision:', round((cm_xgb[0, 0] / sum(cm_xgb[:, 0]))*100,3))
print('Recall:', round((cm_xgb[0, 0] / sum(cm_xgb[0, :]))*100,3))
print('Specificity:', round((cm_xgb[1, 1] / sum(cm_xgb[1, :]))*100,3))

%matplotlib inline
confusion_xg = dmba.classificationSummary(y_test, xgb_predict, class_names=xgb.classes_)
labels = ([['True Negative: %s'%cm_xgb[0,0], 'False Positive: %s'%cm_xgb[0,1]],['False Negative: %s'%cm_xgb[1,0], 'True Positive: %s'%cm_xgb[1,1]]])
fig, ax = plt.subplots()
sns.heatmap(cm_xgb, annot= labels, fmt = '', cmap="BuPu") 
plt.title('Confusion Matrix: XGB Classifier')

fpr, tpr, thresholds = roc_curve(y_test, xgb.predict_proba(X_test)[:, 0])
auc_score = roc_auc_score(y_test, xgb_predict)

# Plot the ROC curve
plt.title('XG Boost ROC Curve')
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Recall (True Positive Rate)')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.legend(loc='upper left')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression

logit_model = smf.logit(formula=('Diabetes ~ Age + Gender + Polyuria + Polydipsia + sudden_weight_loss + weakness + Polyphagia + Genital_thrush + visual_blurring + Itching + Irritability + delayed_healing +partial_paresis +  muscle_stiffness+ Alopecia + Obesity'),data=pd.concat([X_train, y_train], axis=1)).fit()
logit_model.summary()

yhat = round(logit_model.predict(X_train[['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss',
       'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring',
       'Itching', 'Irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'Alopecia', 'Obesity']]))
pred = round(logit_model.predict(X_test[['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss',
       'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring',
       'Itching', 'Irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'Alopecia', 'Obesity']]))


cm = confusion_matrix(y_test,pred)
accuracy_score(yhat, y_train)
accuracy_score(pred, y_test)


print('Logistic Regression Model Training Accuracy Score:',round(accuracy_score(yhat, y_train)*100,3))
print('Logistic Regression Model Testing Accuracy Score:',round(accuracy_score(pred, y_test)*100,3))
print('Precision', round((cm[0, 0] / sum(cm[:, 0]))*100,3))
print('Recall', round((cm[0, 0] / sum(cm[0, :]))*100,3))
print('Specificity', round((cm[1, 1] / sum(cm[1, :]))*100,3))


%matplotlib inline
labels = ([['True Negative: %s'%cm[0,0], 'False Positive: %s'%cm[0,1]],['False Negative: %s'%cm[1,0], 'True Positive: %s'%cm[1,1]]])
fig, ax = plt.subplots()
sns.heatmap(cm, annot= labels, fmt = '', cmap='BuPu')
plt.title('Confusion Matrix: Logistic Regression')
plt.show()

pred_prob = logit_model.predict(X_test)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
auc_score = roc_auc_score(y_test, pred)

# Plot the ROC curve
plt.title('Logistic Regression ROC Curve')
plt.plot(fpr, tpr, color='green', label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Recall (True Positive Rate)')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.legend(loc='lower right')
plt.show()

def oddsratio(x):
    import numpy as np
    odds_ratios = pd.DataFrame(
    {
        "OddsRatio": x.params,
        "Lower CI": x.conf_int()[0],
        "Upper CI": x.conf_int()[1],
    }
    )
    odds_ratios = np.exp(odds_ratios)
    return print(odds_ratios)

oddsratio(logit_model)

