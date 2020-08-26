
# Assigning features and label variables
weather= ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy', 'Sunny', 'Overcast','Overcast','Rainy'] 
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

lb = preprocessing.LabelEncoder()
weather_encoder = lb.fit_transform(weather)
print(weather_encoder)
temp_encoded = lb.fit_transform(temp)
print(temp_encoded)
label = lb.fit_transform(play)
print(label)
##Combine both the features
features = list(zip(weather_encoder, temp_encoded))
#create model
model = GaussianNB()
#fit the data
model.fit(features, label)
#predict with the new data
predict = model.predict([[0, 2]])
print(predict)