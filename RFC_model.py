import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report , accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('Titanic-Dataset.csv')
df = df.drop(['Ticket', 'Fare', 'Cabin','Name'], axis=1)


df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


LabEnc = LabelEncoder()
df["Sex"] = LabEnc.fit_transform(df["Sex"])
df["Embarked"] = LabEnc.fit_transform(df["Embarked"])


#FEATURE ENGINEERING
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Pclass"] = df["Pclass"].astype(str)


X = df.drop('Survived',axis = 1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RFC_model = RandomForestClassifier(n_estimators=100, random_state=42)
RFC_model.fit(X_train,y_train)

prediction = RFC_model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)


print(accuracy)
print(classification_report(y_test,prediction))