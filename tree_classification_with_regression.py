
import pandas
import numpy as np
from sklearn import tree

'''
Note:
By looking at the p-values from the logistic regression model and only using the variables with a value
at or below alpha, results are improved. In my testing, the following alpha values resulted in different
levels of accuracy (original tree classification was ~70% and logistic regression model was ~71% accurate):
alpha = 0.05 -> 78.4-80.0% accuracy
alpha = 0.01 -> 74-77% accuracy
alpha = 0.001 -> 75.6-78.8% accuracy
alpha = 0.0001 -> 86.8-87.2% accuracy
alpha = 0.00001 -> 88.4% accuracy
This model still isn't very good because it remains poor at predicting whether an employee will leave,
rather the model tends to predict that employees won't leave and since so many don't, it becomes more accurate.
'''

df = pandas.read_csv('cleaned_data.csv') # testing + training data
df2 = pandas.read_csv('logisticRegressionOutput.csv') # p-values from logistic regression model

print(df.dtypes)
print(df.head())

variables = []

alpha = 0.01

for i, row in df2.iterrows():

	if row[0] != '(Intercept)' and row['pValues'] <= alpha:
		variables.append(row[0])

print(variables)

assert len(variables) > 0, 'No variables above alpha level.' # make sure that there is at least one variable being used

x_values = df[variables].values
print(x_values)

y_values = df['AttritionStatus'].values
print(y_values)

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x_values[:750], y_values[:750]) # use the first 3/4 of the data for training

predictions = classifier.predict(x_values[750:])

correct = 0
#[correct (yes), incorrect (prediction = yes), incorrect (prediction = no), correct (no)]
prediction_distribution = [0,0,0,0]
for i,p in enumerate(predictions):
	#print('Prediction:', p, 'Actual:', y_values[750+i], '[Correct]' if p == y_values[750+i] else '')
	correct += 1 if p == y_values[750+i] else 0
	if p == 1:
		if y_values[750+i] == 1:
			prediction_distribution[0] += 1
		else:
			prediction_distribution[1] += 1
	else:
		if y_values[750+i] == 1:
			prediction_distribution[2] += 1
		else:
			prediction_distribution[3] += 1

print('Accuracy:', correct/250)
print(prediction_distribution)
