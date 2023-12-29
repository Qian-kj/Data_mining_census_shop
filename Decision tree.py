import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DEBUGGING = True
DATA_DIR  = 'C:/Users/qiankj/Desktop/Coursework-DM/data/'
DATA_FILE = 'adult.csv'

#import dataset
dataset = pd.read_csv(DATA_DIR + DATA_FILE)

#drop the attribute fnlwgt
dataset.drop(['fnlwgt'], inplace = True, axis = 1)
adult_dat = dataset.copy()
print(adult_dat.head())

#1
#1.1 number of instances
num_ins = adult_dat.shape[0]
print('(i) number of instances:', num_ins)

#1.2 the number of missing value
num_null = adult_dat.isnull().sum()
all_nan = 0
for i in num_null:
  all_nan = all_nan+i
print('(ii) number of missing value:',all_nan)

#1.3 fraction of missing values over all attribute values
all_values = adult_dat.shape[0]*(adult_dat.shape[1] - 1)
fraction = all_nan / all_values
print('(iii) fraction of missing values over all attribute values:', round(fraction,4))

#1.4 number of instances with missing values
# any rows which contain nan
num_null_row = adult_dat[adult_dat.isna().any(axis=1)].shape[0]
print('(iv) number of instances with missing values:', num_null_row)

#1.5 fraction of instances with missing values over all instances
fraction2 = num_null_row / num_ins
print('(v) fraction of missing values over all attribute value', round(fraction2,4))

#2
#drop instance with missing value
adult_dat.dropna(inplace=True)
print(adult_dat.isnull().sum())

#Label encoding the attributes and target
le = LabelEncoder()
for i in adult_dat:
    adult_dat[i] = le.fit_transform(adult_dat[i])
    print('the set of all possible discrete values for {}:'.format(i),adult_dat[i].unique())

#3
#training data & test data
y = adult_dat.pop('class')
X = adult_dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
M_train = len(X_train)
M_test = len(X_test)
if (DEBUGGING):
    print('number of training instances:', M_train)
    print('number of test instances:', M_test)

#initialise the decision tree
clf = DecisionTreeClassifier(random_state = 0)
#fit the tree model to the training data
clf.fit(X_train, y_train)

#predict the labels for the test set
y_hat = clf.predict(X_test)
#calculate error rate
n = 0
for i in range(M_test):
    if y_hat[i] != y_test.iloc[i]:
        n += 1
error_rate = n / M_test
print('the error rate of the resulting tree:',round(error_rate,4))

#4
#all instances with at least one missing value
adult_dat2 = dataset.copy()


#(i) all instances with at least one missing value
nan_dat = adult_dat2[adult_dat2.isnull().T.any()]

#(ii) an equal number of randomly selected instances without missing values
#number of instances with missing value
num_nan_row = nan_dat.shape[0]
#instances without missing values
adult_dat2.dropna(inplace = True)
#an equal number of randomly selected instances without missing values
selected_dat = adult_dat2.sample(n=num_nan_row, axis=0)

#smaller data set D'
other_D = nan_dat.append(selected_dat)
#order randomly
other_D = other_D.sample(frac=1)
#Index of other_D dataset
index_other_D = other_D.index.tolist()
#Original dataset D
dataset_D = dataset.copy()
#Drop the instances which are contained in other_D dataset
dataset_D.drop(index = index_other_D, inplace=True)

#4.1 construct D'_1
#handle missing values with method(i)
other_D1 = other_D.copy()
dataset_D1 = dataset_D.copy()
for i in other_D1:
    other_D1[i] = other_D1[i].replace(np.nan, 'missing')
    dataset_D1 = dataset_D1.replace(np.nan, 'missing')

#Label encoding the attributes and target
le2 = LabelEncoder()
le3 = LabelEncoder()
for i in other_D1:
    other_D1[i] = le2.fit_transform(other_D1[i])
    dataset_D1[i] = le3.fit_transform(dataset_D1[i])

#training data & test data for D'_1
y1_train = other_D1.pop('class')
X1_train = other_D1
y1_test = dataset_D1.pop('class')
X1_test = dataset_D1

M1_train = len(X1_train)
M1_test = len(X1_test)
if (DEBUGGING):
    print('number of training instances:', M1_train)
    print('number of test instances:', M1_test)

#initialise the decision tree
clf1 = DecisionTreeClassifier()
#fit the tree model to the training data
clf1.fit(X1_train, y1_train)

#predict the labels for the test set
y1_hat = clf1.predict(X1_test)
#calculate error rate
n1 = 0
for i in range(M1_test):
    if y1_hat[i] != y1_test.iloc[i]:
        n1 += 1
error_rate1 = n1 / M1_test

#4.2 construct D'_2
other_D2 = other_D.copy()
dataset_D2 = dataset_D.copy()
for i in other_D2:
    other_D2[i] = other_D2[i].fillna(other_D2[i].mode().iloc[0])
    dataset_D2[i] = dataset_D2[i].fillna(dataset_D2[i].mode().iloc[0])

#Label encoding the attributes and target
le2_2 = LabelEncoder()
le3_2 = LabelEncoder()
for i in other_D2:
    other_D2[i] = le2_2.fit_transform(other_D2[i])
    dataset_D2[i] = le3_2.fit_transform(dataset_D2[i])

#training data & test data for D'_1
y2_train = other_D2.pop('class')
X2_train = other_D2
y2_test =dataset_D2.pop('class')
X2_test = dataset_D2
M2_train = len(X2_train)
M2_test = len(X2_test)
if (DEBUGGING):
    print('number of training instances:', M2_train)
    print('number of test instances:', M2_test)

#initialise the decision tree
clf2 = DecisionTreeClassifier()
#fit the tree model to the training data
clf2.fit(X2_train, y2_train)

#predict the labels for the test set
y2_hat = clf2.predict(X2_test)
#calculate error rate
n2 = 0
for i in range(M2_test):
    if y2_hat[i] != y2_test.iloc[i]:
        n2 += 1
error_rate2 = n2 / M2_test

#compare error rate for D'_1 with error rate for D'_2
print('the error rate for D\'_1 :',round(error_rate1,4),
    '\nthe error rate for D\'_2:',round(error_rate2,4))