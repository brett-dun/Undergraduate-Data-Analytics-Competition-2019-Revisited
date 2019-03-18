
import pandas
import numpy as np
from sklearn import tree

'''
Note:
This is a poor way of going about making a predictions. While the accuracy is somwhere around 70%,
this model takes too many variables into account. A better implementation of this would use the logistic
regression model from the original competition to choose which variables to use and then use sklearn to
create a model that only uses the variables with the highest correlation to attitrion status.
'''

df = pandas.read_csv('cleaned_data.csv')

print(df.dtypes)
print(df.head())

x_values = df[4:].values
print(x_values)

y_values = df['AttritionStatus'].values
print(y_values)

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x_values[:750], y_values[:750]) # use the first 3/4 of the data for training

predictions = classifier.predict(x_values[750:])

correct = 0
for i,p in enumerate(predictions):
	#print('Prediction:', p, 'Actual:', y_values[750+i], '[Correct]' if p == y_values[750+i] else '')
	correct += 1 if p == y_values[750+i] else 0

print('Accuracy:', correct/250)