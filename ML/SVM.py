import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

bankdata = pd.read_csv("C:/Users/ABC/Desktop/New folder/TDAI/Data/bill_authentication.csv")
#print(bankdata.shape)
#print(bankdata.head(x)) x is row which u want to see
#data processing
#divide data for two part
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

svcclassifer = SVC(kernel='linear')
svcclassifer.fit(X_train, y_train)

y_predict = svcclassifer.predict(X_test)

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

plt.plot(y_train)
plt.ylabel("class")
plt.show()