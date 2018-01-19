# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:33:49 2018

@author: lixud
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset and drop unnecessary data
titanic_df=pd.read_csv('train.csv', dtype={"Age": np.float64}, )
test_df=pd.read_csv('test.csv', dtype={"Age": np.float64}, )
titanic_df = titanic_df.drop(['PassengerId'], axis=1)


# Variable "Name"
# The longer the name, the higher the survival rate
titanic_df.groupby(titanic_df.Name.apply(lambda x: len(x)))['Survived'].mean().plot()

titanic_df['Name_Length'] = titanic_df['Name'].apply(lambda x: len(x))
test_df['Name_Length'] = test_df['Name'].apply(lambda x: len(x))

titanic_df['Name_Length'] = pd.qcut(titanic_df['Name_Length'],5)
test_df['Name_Length'] = pd.qcut(test_df['Name_Length'],5)

# get a title feature from the names
titanic_df['Title'] = titanic_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
titanic_df['Title'] = titanic_df['Title'].replace(['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr'],'Mr')
titanic_df['Title'] = titanic_df['Title'].replace(['Mlle','Ms'], 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(titanic_df['Title'],prefix='Title')
titanic_df = pd.concat([titanic_df,df],axis=1)
titanic_df = titanic_df.drop(['Title'], axis=1)


test_df['Title'] = test_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
test_df['Title'] = test_df['Title'].replace(['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr'],'Mr')
test_df['Title'] = test_df['Title'].replace(['Mlle','Ms'], 'Miss')
test_df['Title'] = test_df['Title'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(test_df['Title'],prefix='Title')
test_df = pd.concat([test_df,df],axis=1)
test_df = test_df.drop(['Title'], axis=1)


# Get a feature on family. ie. if a family member survived, the rest of family member also has a high chance survive
titanic_df['Fname'] = titanic_df['Name'].apply(lambda x:x.split(',')[0])
titanic_df['Familysize'] = titanic_df['SibSp']+titanic_df['Parch']
dead_female_Fname = list(set(titanic_df[(titanic_df.Sex=='female') & (titanic_df.Age>=12)
                              & (titanic_df.Survived==0) & (titanic_df.Familysize>1)]['Fname'].values))
survive_male_Fname = list(set(titanic_df[(titanic_df.Sex=='male') & (titanic_df.Age>=12)
                              & (titanic_df.Survived==1) & (titanic_df.Familysize>1)]['Fname'].values))
titanic_df['Dead_female_family'] = np.where(titanic_df['Fname'].isin(dead_female_Fname),1,0)
titanic_df['Survive_male_family'] = np.where(titanic_df['Fname'].isin(survive_male_Fname),1,0)

test_df['Fname'] = test_df['Name'].apply(lambda x:x.split(',')[0])
test_df['Familysize'] = test_df['SibSp']+test_df['Parch']
test_df['Dead_female_family'] = np.where(test_df['Fname'].isin(dead_female_Fname),1,0)
test_df['Survive_male_family'] = np.where(test_df['Fname'].isin(survive_male_Fname),1,0)

titanic_df = titanic_df.drop(['Name','Fname'],axis=1)
test_df = test_df.drop(['Name','Fname'],axis=1)

# Change Family Size to small, normal and big
titanic_df['Familysize'] = np.where(titanic_df['Familysize']==0, 'solo',
                                    np.where(titanic_df['Familysize']<=3, 'normal', 'big'))
df = pd.get_dummies(titanic_df['Familysize'],prefix='Familysize')
titanic_df = pd.concat([titanic_df,df],axis=1)
titanic_df = titanic_df.drop(['SibSp','Parch', 'Familysize'], axis=1)

test_df['Familysize'] = np.where(test_df['Familysize']==0, 'solo',
                                    np.where(test_df['Familysize']<=3, 'normal', 'big'))
df = pd.get_dummies(test_df['Familysize'],prefix='Familysize')
test_df = pd.concat([test_df,df],axis=1)
test_df = test_df.drop(['SibSp','Parch', 'Familysize'], axis=1)

# Variable "Ticket"
titanic_df['Ticket_Lett'] = titanic_df['Ticket'].apply(lambda x: str(x)[0])
titanic_df['Ticket_Lett'] = titanic_df['Ticket_Lett'].apply(lambda x: str(x))

titanic_df['High_Survival_Ticket'] = np.where(titanic_df['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
titanic_df['Low_Survival_Ticket'] = np.where(titanic_df['Ticket_Lett'].isin(['A','W','3','7']),1,0)
titanic_df = titanic_df.drop(['Ticket','Ticket_Lett'],axis=1)

test_df['Ticket_Lett'] = test_df['Ticket'].apply(lambda x: str(x)[0])
test_df['Ticket_Lett'] = test_df['Ticket_Lett'].apply(lambda x: str(x))

test_df['High_Survival_Ticket'] = np.where(test_df['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
test_df['Low_Survival_Ticket'] = np.where(test_df['Ticket_Lett'].isin(['A','W','3','7']),1,0)
test_df = test_df.drop(['Ticket','Ticket_Lett'],axis=1)



# Variable "Embarked", drop "S" and keep "C" and "Q"
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

# Variable "Fare"
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
# Convert float to int for "Fare"
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# Variable "Age"
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)

# Add a feature for children
titanic_df['IsChild'] = np.where(titanic_df['Age']<=14,1,0)
titanic_df['Age'] = pd.cut(titanic_df['Age'],5)

test_df['IsChild'] = np.where(test_df['Age']<=14,1,0)
test_df['Age'] = pd.cut(test_df['Age'],5)

# Variable Cabin, Drop
titanic_df = titanic_df.drop('Cabin',axis=1)
test_df = test_df.drop('Cabin',axis=1)

# Variable "Sex"
sex_dummies_titanic = pd.get_dummies(titanic_df['Sex'])
sex_dummies_test = pd.get_dummies(test_df['Sex'])

titanic_df = titanic_df.join(sex_dummies_titanic)
test_df    = test_df.join(sex_dummies_test)

titanic_df.drop(['Sex'], axis=1,inplace=True)
test_df.drop(['Sex'], axis=1,inplace=True)

titanic_df.drop(['female'], axis=1,inplace=True)
test_df.drop(['female'], axis=1,inplace=True)

# Variable "Pclass"
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

##################################
from sklearn.preprocessing import LabelEncoder
features = titanic_df.drop(["Survived"], axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(titanic_df[feature])
    titanic_df[feature] = le.transform(titanic_df[feature])

features = test_df.drop(["PassengerId"], axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(test_df[feature])
    test_df[feature] = le.transform(test_df[feature])
# train and test
X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()
'''
# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
'''
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
'''
# GBDT
from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier(n_estimators=500,learning_rate=0.03,max_depth=3)

GBDT.fit(X_train, Y_train)

Y_pred = GBDT.predict(X_test)

GBDT.score(X_train, Y_train)
'''
# Submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('titanic.csv', index=False)
