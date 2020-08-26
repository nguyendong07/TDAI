
weather= ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy', 'Sunny', 'Overcast','Overcast','Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

le = preprocessing.LabelEncoder()
weatherEncoder = le.fit_transform(weather)
tempEncoder = le.fit_transform(temp)
playEncoder = le.fit_transform(play)

features = list(zip(weatherEncoder,tempEncoder))
model = KNeighborsClassifier()
model.fit(features, playEncoder)
predict = model.predict([[0,2]])
print(predict)
