# Predicting-Molecular-Solubility-LogS-
This Python program uses a linear regression model to predict the aqueous solubility (logS) of molecules from molecular descriptors such as molecular weight, aromatic proportion, number of rotatable bonds, and MolLogP. Includes data exploration, training/testing split, and visualization of predicted versus actual values.
```

import pandas as pd

# load the data set into variable called df
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
print(df)

"""about the dataset:
we are predicting solubility - logS
higher logP values means lower solubility

1. the first column MolLogP is a measure of
hydrophobicity and higher values means more hydrophobic so 
less soluble and lower MolLogP values means more soluble in water
How does it relate to our predicting variable logS: water solubility
is strongly negatively correlated with LogP

2. second column MOlWt is molecular weight which is 
total mass of molecule in g/mol
higher MolWt = lower logS

3. NumRotatableBonds is teh third column and indicates molecular felxibility
basically determined by the number of single bonds that can rotate freely
higher number means more flexibility and higher solubility

4. aromatic proportion = ratio of aromatic atoms/total heavy atoms
aromatics are like atoms in rings (benzene)

5. logS is a measure of a molecule's solubility on a 
base 10 log scale measured in mol/L. logS >=-1 mean high solubility
logS <= -9 means rlly low solubility
higher LogS means more solubility
#

if we have high MolLogP, high MolWt and high aromatic = low logS
"""

# data set is abt molecules we are using first 4
#columns to predict 5th column logS = solubility

# we will now separate data into x and y

# create a separate data frame for just y-value
y = df["logS"]
#print(y)

# create a dataframe for our x-variables (first 4 columns) by
# removing the logS column from original df
X = df.drop("logS", axis=1)
#print(X)

# now we need to split into our training and testing set
from sklearn.model_selection import train_test_split

# training set X_train has 80% of data and testing set X_test has 20% of data
#train_test_split basically splits data for us in an unbiased way and setting "random_state" with integer
# makes sure it splits the data at same spot everytime we run program
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
#print(X_test)

# model building now
# import the linear regression model
from sklearn.linear_model import LinearRegression

# lr stands for linear reg, we create an empty model object
#it has the algorithm for performing lin reg but no data
lr = LinearRegression()

# this is where we feed it the data or where the "learning" happens
# model looks at the training data and adjusts coeff to best match the
# y_train values
lr.fit(X_train, y_train)

#now we will apply the model to make a prediction bc we finished training it
# first check training set so that the model applies what it has been trained on namely the X_train
y_lr_train_pred = lr.predict(X_train)

# now we run the model on our testing data and check how accurate it is for
# new data it's never seen before bc not in tarining set
y_lr_test_pred = lr.predict(X_test)



# now we analyze the performance of the model by using a graph
# and checking by how much the values deviate

# these are used to check how good the model is
from sklearn.metrics import mean_squared_error, r2_score

# for the training set we calcuate mse and r^2
# mse = mean squared error and is the avg squared diff btwn prediction and truth
#lower mse = better. r^2 is coeff of determination tells u how much variability in data is captured by model
# higher r^2 close to 1 means better accuracy/better fit
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
# print("train mse:", lr_train_mse)
# print("train r2:", lr_train_mse)

# do same for testing set (unseen data) if train score > test score then overfitting
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
# print("test mse:", lr_test_mse)
# print("test r2:", lr_test_r2)


# now we move onto data visualization
# matplotlib for graphs
# numpy for math tools
import matplotlib.pyplot as plt
import numpy as np

# create a 5x5 inch square plot
plt.figure(figsize=(5,5))

#we want a scatterplot with x-axis having actual LogS values (data we used for training)
# y-axis has the predicted logS values. each dot is a single molecule, alpha makes points transparent
# c changes color. for a perfect model all dots would lie in diagonal
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha =0.3)

# add a line of best fit, degree 1 means straight line
# polyfit means find best-fitting polynomial btwn the x and y
# it finds the best y=mx+b for the data basically, z = [slope, intercept] a numpy array
z = np.polyfit(y_train, y_lr_train_pred, 1)

# this turns the [slope, y-int] into a callable function on data x
# creates a p(x) function used to get the output of some x
# if x = 2 then output = m(2) + b, where it has the m and b values alr
p = np.poly1d(z)

# plot the regression line onto the graph
plt.plot(y_train, p(y_train), "#F8766D")

# label the axes + title
plt.ylabel("Predict LogS")
plt.xlabel("Actual LogS")
plt.title("Actual vs Predicted logS")

# show the plot
#plt.show()

# for practicing data visualization I will plot the testing data
plt.figure(figsize = (5,5))
plt.scatter(x=y_test, y=y_lr_test_pred, alpha=0.3)
z2 = np.polyfit(y_test, y_lr_test_pred, 1)
g = np.poly1d(z2)

plt.plot(y_test, g(y_test))

plt.xlabel("Actual LogS (test)")
plt.ylabel("Predicted LogS (test)")
plt.title("Actual vs Predicted LogS (Testing set)")

# this line will show both graphs
plt.show()
