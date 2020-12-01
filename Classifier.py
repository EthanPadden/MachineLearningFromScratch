import random
import numpy as np

# open file and read in formatted content to text array
def process_file(filename) :
	file = open(filename, "r")
	raw_text = file.read()
	file.close()

	data = np.empty((0, 9), float)
	labels = np.array([])

	lines = raw_text.split('\n')
	for line in lines:
		attributes = np.array(line.split('\t'))
		labels = np.append(labels, attributes[3]) # index 3 is the style, append to labels
		attributes = np.delete(attributes, 3) # and then delete it from the data
		attributes = attributes.astype(np.float) # all other attributes should be floats
		data = np.append(data, np.array([attributes]), axis=0)

	return data, labels

# check percentage similarity of two arrays
# if inputs are incompatible (eg different array lengths) returns -1
def array_similarity(array_1, array_2):
	identical = 0
	different = 0

	if (len(array_1) == len(array_2) and len(array_1) > 0):
		for x in range(len(array_1)):
			if array_1[x] == array_2[x]:
				identical += 1
			else:
				different += 1
	else :
		return -1

	# round % to 2 decimal places
	return round((identical / (identical + different) * 100), 2)

def split_data_set(data_array, label_array, percentage_for_training) :
	# takes data and label arrays and splits them into training and test based on percentage provided
	training_data = data_array
	training_labels = label_array

	testing_data = []
	testing_labels = []

	training_data_length = (len(data_array) * (percentage_for_training/100))

	i = 0

	while (i < training_data_length) :
		chance = (random.randint(1, len(data_array)))
		if (chance < training_data_length):
			# put the randomly selected data instance into the testing array
			testing_data.insert(1, training_data[i])
			testing_labels.insert(1, training_labels[i])

			# then delete it from the training array
			np.delete(training_data, i)
			np.delete(training_labels, i)

			i += 1

	return training_data, training_labels, testing_data, testing_labels

class Classifier:
	def __init__(self):
		### TODO
		a = 5

	def fit(self, X, y):
		### TODO
		a = 5

	def predict(self, X):
		### TODO
		return []

if __name__ == '__main__':
	data, labels = process_file("beer.txt")
	training_data, training_labels, testing_data, testing_labels = split_data_set(data, labels, 33.33)

	classifier = Classifier()
	classifier.fit(training_data, training_labels)
	prediction = classifier.predict(testing_data)

	print("Accuracy: ", array_similarity(prediction, testing_labels), "%")
