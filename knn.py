import random
import csv
import math 
from collections import Counter

# flower parameters are part of train set, labels are part of test set. Labels are the classes for the classification
x_train, x_test, y_train, y_test = [], [], [], []
k = 19

# Euclidean distance function for 2 lists p and q containing a set of points (4 points in this case)
def euclidean_distance(p, q):
	# The formula for euclidean distance is ... 
	distanceSq = 0
	for i in range(3):
		distanceSq += (float(q[i]) - float(p[i]))**2
	return math.sqrt(distanceSq)

# Load data into a list of lists
with open('iris.csv', newline='') as irisCSV:
	dataset = []
	reader = csv.reader(irisCSV)
	# Each row is a flower
	for row in reader:
		dataset.append(row)

# Split into training and test
for row in dataset:
	# Randomly generate a number 0.0 <= n <= 1.0. It has 0.5 probability of being under 0.5. This gives us a random split which is close to 50-50. It may not always be a 50% but it is random.  
	if random.random() < 0.9:
		# Append all but last elements of the row to x_train. Last element is the label so it gets added to y_train. Indixes are preserved this way. so for index i y_train[i] will be the label for x_train[i]
		x_train.append(row[:-1])
		y_train.append(row[-1])
	else:
		x_test.append(row[:-1])
		y_test.append(row[-1])

# Create list of distances to all neighbors
for index, query in enumerate(x_test):
	distances = []
	for neighbor in x_train:
		distances.append(euclidean_distance(query, neighbor))

	# Get a list of indixes in sorted order of their corresponding distance. Sort distances but return indices of each distance in the original list. To do this, we call enumerate() whch gives us index, value for a given list. Then we sort the enumeration according to the output of a simple function (lambda function) which retrieves the value element from the enumeration (stored at index 1). The value of the key parameter should be a function (or other callable) that takes a single argument and returns a key to use for sorting purposes. This technique is fast because the key function is called exactly once for each input record.
	sorted_indices = [index for index, value in sorted(enumerate(distances), key=lambda value: value[1])]
	# Get the labels for the first k nearest neighbors. First select the top k indices from the sorted list of indices. Then get the label for that index.
	knn = [y_train[i] for i in sorted_indices[:k]]
	# Get the label which appears the most in the k-nearest neighbors. Counter function creates a dictionary of elements with their counts. We select the first most common element and [0] selects the string value and omits the count of times it appeared in the list
	y_prediction = Counter(knn).most_common(1)[0][0]


	print("Predicted nearest neighbors: {0} ; Actual: {1}".format(y_prediction, y_test[index]))
