import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(['Ticket', 'Fare', 'Cabin'], axis=1)  # Ticket ID has no correlation to all other columns ; Fare column is equivalent to Pclass ; Cabin column has too many missing values 


print("--> First 5 Rows of the Dataset:")
print(df.head())
print("\n" + "-"*50 + "\n")
print("--> Summary Statistics:")
print(df.describe())





#Reviewing the data after dropping , only two columns have missing values --> Age , Embarked 
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median())) # Fills the missing values by grouping Pclass and Sex and using it to calculate median
df['Embarked'].fillna(df['Embarked'].mode()[0])



plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.text(0.5, -0.1, 'This Graph shows us that passengers from Class-1 had the highest chances of survival', 
         horizontalalignment='center', 
         verticalalignment='center', 
         transform=plt.gca().transAxes)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.text(0.5, -0.1, 'This Graph shows us that there were more female survivors', 
         horizontalalignment='center', 
         verticalalignment='center', 
         transform=plt.gca().transAxes)
plt.show()         


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df[df['Survived'] == 1]['Age'], label='Survived', color='green', ax=axes[0])
axes[0].set_title("Age Distribution of Survivors")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].text(0.5, -0.1, "This graph shows us that most survivors were between the age 20-40", ha='center', va='center', fontsize=12, transform=axes[0].transAxes)

sns.histplot(df[df['Survived'] == 0]['Age'], label='Not Survived', color='red', ax=axes[1])
axes[1].set_title("Age Distribution of Non-Survivors")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].text(1.7, -0.1, "This graph shows us that most casualties were between the age 20-40", ha='center', va='center', fontsize=12, transform=axes[0].transAxes)
plt.show()

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  
plt.figure(figsize=(8, 6))
sns.barplot(x=df['FamilySize'], y=df['Survived'], errorbar = None ,  palette='Oranges')
plt.xlabel('Family Size (All members of a family)')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Family Size')
plt.ylim(0, 1)
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(x=df['Embarked'], y=df['Survived'], errorbar = None ,  palette='Purples')
plt.xlabel('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Embarkation Port')
plt.ylim(0, 1)
plt.show()

