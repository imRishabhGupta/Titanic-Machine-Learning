# Titanic-Machine-Learning
This project contains script for the famous Titanic:Machine Learning from Disaster competition on Kaggle.
It uses RandomForestClassifier supervised machine learning algorithm and was able to achieve accuracy of 0.8(approx.).
In total, 17 features have been used by this project among which 11 have engineered.

## Getting Started
Before running the code make sure that the project folder contains all the three files -
train.csv
test.csv
titanic.py

### Prerequisites 
This project uses only three famous libraries - 
pandas
numpy
sklearn
One can download these using command 

```
pip install pandas
```

If you are using Anaconda, then you will already have pandas and numpy installed. And you can install sklearn using command

```
conda install sklearn
```
## Running 
Now, we ready and we can run the file titanic.py. We can also select which features we want to use to train our classifier in the function

```
get_features()
```

Features currently this code is using are -
```
'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Family_size', 'SmallFamily', 'LargeFamily', 'AgeGroup', 'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Mas'
```

