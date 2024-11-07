import pandas as ps

data = ps.read_csv("student-por.csv", sep=';')

#### creates a entry in every row called pass that is 
#### 0 if they failed (grade below 35) or
#### 1 if they passed (grade above 35)
data['pass'] = data.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)

#### deletes all entries of the grades in each row
data = data.drop(['G1','G2','G3'], axis=1)

#### .head() is first 5 rows
print(data.head())
