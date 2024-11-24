import pandas as ps
import numpy as np
from sklearn import tree

data = ps.read_csv("student-por.csv", sep=';')

#### creates a entry in every row called pass that is 
#### 0 if they failed (grade below 35) or
#### 1 if they passed (grade above 35)
data['pass'] = data.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)

#### deletes all entries of the grades in each row
data = data.drop(['G1','G2','G3'], axis=1)

data = ps.get_dummies(data, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

data = data.sample(frac=1)

data_train = data[:500]
data_test = data[500:]

data_train_att = data_train.drop(['pass'], axis=1)
data_train_pass = data_train['pass']

data_test_att = data_test.drop(['pass'], axis=1)
data_test_pass = data_test['pass']

data_att = data.drop(['pass'], axis=1)
data_pass = data['pass']

print("Passing: %d out of %d" % (np.sum(data_pass), len(data_pass)))

t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
t = t.fit(data_train_att, data_train_pass)