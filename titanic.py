# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:37:35 2017

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn import tree

def get_data():
    train = pd.read_csv("train.csv")
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    test=pd.read_csv("test.csv")
    train = train.append(test) 
    return train,targets
def process_age():
    grouped = train.groupby(['Sex','Pclass','Title'])
    print grouped.median()
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    train.Age = train.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

    
def getTitles():
    train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    train['Title'] = train.Title.map(Title_Dictionary)

def process_titles():
    train["Title_Mr"] = float(0)
    train["Title_Mr"][train["Title"]=="Mr"]=1
    train["Title_Of"] = float(0)
    train["Title_Of"][train["Title"]=="Officer"]=1
    train["Title_Ro"] = float(0)
    train["Title_Ro"][train["Title"]=="Royalty"]=1
    train["Title_Mrs"] = float(0)
    train["Title_Mrs"][train["Title"]=="Mrs"]=1
    train["Title_Miss"] = float(0)
    train["Title_Miss"][train["Title"]=="Miss"]=1
    train["Title_Mas"] = float(0)
    train["Title_Mas"][train["Title"]=="Master"]=1
    
def process_sex():
    train["Sex"][train["Sex"]=="female"]=1
    train["Sex"][train["Sex"]=="male"]=0
    
def process_embarkment():
    train["Embarked"][train["Embarked"]=="S"]=0
    train["Embarked"][train["Embarked"]=="C"]=1
    train["Embarked"][train["Embarked"]=="Q"]=2
    train["Embarked"] = train["Embarked"].fillna(0)

def process_family():
    train["Family_size"] = float(0)
    train["Family_size"]=train["Parch"]+train["SibSp"]+1
    train['Singleton'] = train['Family_size'].map(lambda s : 1 if s == 1 else 0)
    train['SmallFamily'] = train['Family_size'].map(lambda s : 1 if 2<=s<=4 else 0)
    train['LargeFamily'] = train['Family_size'].map(lambda s : 1 if 5<=s else 0)
    
def process_age_group():
    train["AgeGroup"]=float(0)
    train["AgeGroup"][train["Age"]>5]=1
    train["AgeGroup"][train["Age"]>18]=2
    train["AgeGroup"][train["Age"]>30]=3
    train["AgeGroup"][train["Age"]>50]=4
    train["AgeGroup"][train["Age"]>65]=5
    
def process_fares():
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    
def get_features():
    features = list(train.columns)
    features.remove('PassengerId')
    features.remove('Name')
    features.remove('Ticket')
    features.remove('Title')
    features.remove('Cabin')
    features.remove('Singleton')
    features.remove('Title_Of')
    features.remove('Title_Ro')
    #features.remove('Title_Mas')
    return features
    
def get_test_train():
    train1 = train[0:891]
    test = train[891:]
    return train1,test
    
train,target=get_data()
getTitles()
process_age()
process_sex()
process_embarkment()
process_fares()
process_family()
process_age_group()
process_titles()
train1,test=get_test_train()
features = get_features()
print features
from sklearn.ensemble import RandomForestClassifier

features_forest = train1[features].values


print features_forest

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 5, min_samples_split=2, n_estimators = 250, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

featur = pd.DataFrame()
featur['feature'] = features
featur['importance'] = my_forest.feature_importances_
print featur.sort(['importance'],ascending=False)

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[features].values
pred_forest = my_forest.predict(test_features)
#print(len(pred_forest))
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
#print(my_solution)
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
