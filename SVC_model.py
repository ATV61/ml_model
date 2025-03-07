import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report


df = pd.read_csv('Titanic-Dataset.csv')
df = df.drop(['Ticket', 'Fare', 'Cabin','Name'], axis=1)


df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Pclass"] = df["Pclass"].astype(str)


df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=True)
X = df.drop('Survived',axis = 1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))
print(f"SVC Model Accuracy: {accuracy:4f}")