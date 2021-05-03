# Categorize the data
#
# ---------------Labels---------------
# 1. Building windows float processed
# 2. Building windows non-float processed
# 3. Vehicle windows float processed
# 4. Vehicle windows non-float processed
# 5. Containers
# 6. Tableware
# 7. Headlamps

# Features: RI Na Mg Al Si K Ca Ba Fe


import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
# This function converts a list to a string
def convert_list_to_string(list_, seperator=' '):
  return seperator.join(list_)
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# This function randomly shuffles the dataset and returns with feature-set and label-set
def randomly_shuffle(dataset):
  np.random.shuffle(dataset)
  feature_set = dataset[:,1:np.shape(dataset)[1]]
  label_set = dataset[:,0].astype(int)
  return feature_set, label_set
#-------------------------------------------------------------------------------------

#KNN_label = KNN(valid_feature_set, valid_label_set, train_feature_set, train_label_set, k_KNN)
#-------------------------------------------------------------------------------------
# This function returns with results with the KNN classification 
def KNN(sample_array, sample_label, train_feature, train_label, k_value):
    l_d = np.shape(sample_array)[0] #validating set dimension
    f_d = np.shape(sample_array)[1] #feature vector dimension
    
    KNN_result = [] # Initialize an empty vector for KNN_result
    
    for i in range(l_d):
        sample_vector = sample_array[i][:] # Each row for the sample array
        
        #Euclidean Distance Equation is used for defining distance between two points
        Euclidean = np.sum(np.square(sample_vector - train_feature),axis=1)
        
        # The first k (for KNN) closest indices from the sample  
        closest_indices = np.argsort(Euclidean)[0:k_value]

        # A vector for the closest points with k(for KNN) number
        closest = []
        for j in range(k_value):
            # Find closest points using closest indices
            closest.append(train_label[closest_indices[j]])
        
        
        # *******************Vote Process******************* 
        # The class that has a majority vote will be chosen
        # For example... 
        # when "closest" has [2,2,3], it will be considered as 2
        # When "closest" has [1,4,4], it will be considered as 4
        if max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
        closest.count(6),closest.count(7))==closest.count(1):
            KNN_result.append(1)
        elif max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
          closest.count(6),closest.count(7))==closest.count(2):
            KNN_result.append(2)
        elif max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
          closest.count(6),closest.count(7))==closest.count(3):
            KNN_result.append(3)
        elif max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
          closest.count(6),closest.count(7))==closest.count(5):
            KNN_result.append(5)
        elif max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
          closest.count(6),closest.count(7))==closest.count(6):
            KNN_result.append(6)
        elif max(closest.count(1),closest.count(2),closest.count(3),closest.count(5),\
          closest.count(6),closest.count(7))==closest.count(7):
            KNN_result.append(7)
    
    
    # KNN_result returns with a result from the KNN classification
    return KNN_result
#-------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# This function computes ACCURACY
# --------------------------------------------------------------------------------------
def calc_accuracy(trained_guess, real_label):
    
    number_total = np.shape(trained_guess)[0] # Total Number of the elements for KNN label
    
    number_correct_labelled = 0 # Initialize number of correct labelled

    for i in range(number_total):
        # If my trained label is correct, number_correct_labelled increases by 1
        if trained_guess[i] == real_label[i]:
            number_correct_labelled += 1
    
    accuracy = number_correct_labelled / number_total
    return accuracy
# --------------------------------------------------------------------------------------

# Empty array for feature set and label set
feature_set=[] 
label_set=[]
label_set2 =[] # Another label set for making each array for concatenating later

# Reading the csv file
with open('glass_dataset.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    feature_set.append([float(row['RI']),float(row['Na']),float(row['Mg']),float(row['Al']),\
        float(row['Si']),float(row['K']),float(row['Ca']),\
            float(row['Ba']),float(row['Fe'])])
    
    label_set.append(int(row['Type']))
    label_set2.append([int(row['Type'])]) # One more bracket makes it array

# Convert lists to NumPy arrays
feature_set = np.array(feature_set)
label_set = np.array(label_set)
label_set2 = np.array(label_set2)


all_together_set = np.concatenate((feature_set,label_set2), axis=1)


# Analyzing and Processing the dataset
# Mean, std, min, and max
feature_mean = []
feature_std = []
feature_min = []
feature_max = []
for i in range(9):
  feature_mean.append( np.round(np.mean(feature_set[:,i]),decimals=5) )
  feature_std.append( np.round(np.std(feature_set[:,i],ddof=1),decimals=5) )
  feature_min.append( np.round(np.min(feature_set[:,i]),decimals=5))
  feature_max.append( np.round(np.max(feature_set[:,i]),decimals=5))

num_instances = np.shape(feature_set)[0]
one = []
two = []
three = []
five = []
six = []
seven = []

for i in range(num_instances):
    if all_together_set[i,9] == 1:
      one.append(feature_set[i])
    if all_together_set[i,9] == 2:
      two.append(feature_set[i])
    if all_together_set[i,9] == 3:
      three.append(feature_set[i])
    if all_together_set[i,9] == 5:
      five.append(feature_set[i])
    if all_together_set[i,9] == 6:
      six.append(feature_set[i])
    if all_together_set[i,9] == 7:
      seven.append(feature_set[i])    

# Converting lists to NumPy arrays
one = np.array(one)
two = np.array(two)
three = np.array(three)
five = np.array(five)
six = np.array(six)
seven = np.array(seven)


# store number of instances in corresponding to each class
n_one = np.shape(one)[0]
n_two = np.shape(two)[0]
n_three = np.shape(three)[0]
n_five = np.shape(five)[0]
n_six = np.shape(six)[0]
n_seven = np.shape(seven)[0]

#
# Intuitive plot for visualizing the 
#
ii = 0 # Change this value for checking each feature for visualizing plot
y = [one[:,ii],two[:,ii],three[:,ii],five[:,ii],six[:,ii],seven[:,ii]]
x = [1,2,3,4,5,6]

for xe, ye in zip(x,y):
  plt.scatter([xe]*len(ye), ye)

# plt.xticks([1,2,3,4,5,6])
# plt.axes().set_xticklabels(['Class 1','Class 2','Class 3','Class 5','Class 6','Class 7'])
# plt.title('RI (Refractive Index)') # Change title
# plt.show()


# Vectors of Mean and std
mu = []
std = []
for i in range(np.shape(feature_set)[1]):
    mu.append(np.mean(feature_set[:,i]))
    std.append(np.std(feature_set[:,i], ddof=1))

#-----------------------------------------------------
#-------------------STANDARDIZATION-------------------
#-----------------------------------------------------
feature_norm = (feature_set-mu)/std
# print("Standardized feature=")
# print(feature_norm)

#-----------------------------------------------------
#-------------Principle Component Analysis-------------
#-----------------------------------------------------
# Cx of the normalized 
Cx_norm = np.cov(np.transpose(feature_norm))
# print('Normalized Cx=',Cx_norm)                                               ### Check


# Eigenvalues and eigenspace
Eigenvalues,P=LA.eig(Cx_norm)

# print('eigenvalues',Eigenvalues)                                              ### Check
# print('eigenspace=',P)

# Points project to Y=P^T * X
Y = np.transpose(P).dot(np.transpose(feature_norm))
Y = np.transpose(Y)
# print("number",np.shape(Y))
# print('Y (points project to Y=P^T * X) =',Y)                                  ### Check

# New covariance matrix
Cy = P.dot(Cx_norm).dot(np.transpose(P))
# print('Cy=',Cy)

combined_data_set_only = np.concatenate((label_set2, feature_norm),axis=1)
combined_data_set = np.concatenate((label_set2, Y),axis=1)

#-----------------------------------------------------
#------------- KNN with Cross Validation -------------
#-----------------------------------------------------
# Randomly shuffle the dataset
feature_set,label_set = randomly_shuffle(combined_data_set_only)


# Number of columns for all data (Label column + feature columns)
number_cols_all = np.shape(combined_data_set_only)[1]

# K value for the k-fold cross validation
K = 3
split_data = np.array_split(combined_data_set_only, K) # Split array into K folders

# Usually make k an odd number to have a tiebreaker
k_KNN = 1

accuracy = []
for i in range(K): # Validation set
    size=[]
    valid_split_data = split_data[i]
    valid_split_data = np.array(valid_split_data)

    valid_feature_set = valid_split_data[:, 1:np.shape(valid_split_data)[1]]
    valid_label_set = valid_split_data[:,0].astype(int) # int type
    
    number_cols_valid = np.shape(valid_feature_set)[1]

    
    a = np.empty((0,number_cols_all),int) # Initialize an empty numpy array for append later
    
    for j in range(K): # Training set
            if i==j:
                continue
            else:
                temp = split_data[j] # Temporarily store split data
                train_split_data = np.append(a,temp,axis=0) # Append all the training split data in an array
                a = temp
    

    train_feature_set = train_split_data[:, 1:np.shape(train_split_data)[1]]
    train_label_set = train_split_data[:,0].astype(int)

    
    KNN_label = KNN(valid_feature_set,valid_label_set,train_feature_set,train_label_set,k_KNN)
    accu = calc_accuracy(KNN_label, valid_label_set)
    
    accuracy.append(accu)

ave_accuracy = sum(accuracy)/len(accuracy)

print("Accuracy from kNN = ", ave_accuracy)
#
#
# RANDOM FOREST
#
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydot
from sklearn.ensemble import RandomForestRegressor
import os

df = pd.read_csv("glass_dataset.csv")

# Count the numbers of instances for each class
# sizes = df['Type'].value_counts(sort=1)
# print(sizes)

# Define dependent variable
y = df['Type'].values

# Define independent variables
x = df.drop(labels='Type', axis=1)

# Split data into train and test datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 50)


# Instantiate model with decision trees
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the model on training data
model.fit(x_train, y_train)

prediction_test = model.predict(x_test)


print("Accuracy from Random Forest = ", metrics.accuracy_score(y_test, prediction_test))

# It knows, what parameters are contributing the best
# We can extract the importance of the features in order

feature_list = list(x.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
# print(feature_imp)

# Get numerical feature importances
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance (%)'); plt.xlabel('Features'); plt.title('Feature Importances')
plt.ylim([0,1])
# plt.show()


# Decision Tree

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Pull out one tree from the forest
tree = model.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = "tree.dot", feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to creat a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

#
#
# Limit depth of tree to 3 levels



rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(x_train, y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')