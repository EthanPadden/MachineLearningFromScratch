import random

# open file and read in formatted content to text array
def process_file(filename) :
	file = open(filename, "r")
	raw_text = file.read()
	file.close()

	data = []
	labels = []

	lines = raw_text.split('\n')
	for line in lines:
		words = line.split('\t')
		for word in words:
			if word != words[3]:  # don't try convert style string to float!
				word = float(word)
			else:
				labels.insert(1, words.pop(3))  # instead remove style and set as label
		data.insert(1, words)

	return data, labels

# check percentage similarity of two arrays
# if inputs are incompatible (eg different array lengths) returns -1
def array_similarity(array_1, array_2):
	identical = 0
	different = 0

	if len(array_1) == len(array_2):
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
		if chance < (training_data_length):
			testing_data.insert(1, training_data.pop(i))
			testing_labels.insert(1, training_labels.pop(i))

			i += 1

	return training_data, training_labels, testing_data, testing_labels

class Classifier:
	def __init__():
		### TODO

	def fit(self, X, y):
		### TODO

	def predict(self, X):
		### TODO

if __name__ == '__main__':
	data, labels = process_file("beer.txt")
	training_data, training_labels, testing_data, testing_labels = split_data_set(data, labels, 33.33)

	classifier = Classifier()
	classifier.fit(training_data, training_labels)
	prediction = classifier.predict(testing_data)

	print("Accuracy: ", array_similarity(prediction, testing_labels), "%")
