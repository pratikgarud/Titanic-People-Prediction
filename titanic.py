import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

y = df.Survived.map({1:'Survived', 0:'Dead'})

label = LabelEncoder()
X['Sex'] = label.fit_transform(X['Sex'])

X = X.fillna(X.mean())
X['Fare'] = X['Fare'].astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=42, random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred

print(model.score(X_test,y_test)*100)

with open('Titanic_Model.pkl', 'wb') as f:
    pickle.dump(model,f)
