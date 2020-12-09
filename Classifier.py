"""
C4.5 Implementation
https://github.com/liamroddy/MachineLearningFromScratch

Authors:
Ethan Padden (17744021)
Liam Roddy (17738889)
"""

import random
import numpy as np
import math


# open file and read in formatted content to text array
def process_file(filename):
	file = open(filename, "r")
	raw_text = file.read()
	file.close()

	data = np.empty((0, 9), float)
	labels = np.empty((304), dtype='object')

	lines = raw_text.split('\n')
	random.shuffle(lines)
	for line in lines:
		attributes = np.array(line.split('\t'))

		classification = str(attributes[3])
		index = int(attributes[7])
		labels[index] = classification

		attributes = np.delete(attributes, 3)  # delete classification from the data
		attributes = attributes.astype(np.float)  # all other attributes should be floats
		data = np.append(data, np.array([attributes]), axis=0)

	return data, labels


def array_similarity(array_1, array_2):
	# check percentage similarity of two arrays
	# if inputs are incompatible (eg different array lengths) returns -1
	identical = 0
	different = 0

	if (len(array_1) == len(array_2) and len(array_1) > 0):
		for x in range(len(array_1)):
			if array_1[x] == array_2[x]:
				identical += 1
			else:
				different += 1
	else:
		return -1

	# round % to 2 decimal places
	return round((identical / (identical + different) * 100), 2)


def split_data_set(data_array, label_array, percentage_for_training):
	# takes data and label arrays and splits them into training and test based on percentage provided
	training_data = data_array
	training_labels = label_array

	testing_data = np.empty((0, 9), float)
	testing_labels = np.empty((0), dtype='<U5')

	training_data_length = (len(data_array) * (percentage_for_training / 100))

	i = 0

	while (i < training_data_length):
		chance = (random.randint(1, len(data_array)))
		if (chance < training_data_length):
			# put the randomly selected data instance into the testing array
			testing_data = np.append(testing_data, np.array([training_data[i]]), axis=0)

			id_int = int(np.array([training_data[i][6]])) # get the beer id
			label_string = str(training_labels[id_int]) # get the corresponding label for that beer id
			testing_labels = np.append(testing_labels, np.array([label_string]), axis=0) # and add it to the testing_labels

			# then delete the whole data instance from the training array
			training_data = np.delete(training_data, i, axis=0)

			i += 1

	return training_data, training_labels, testing_data, testing_labels


class TreeNode(object):
	def __init__(self):
		self.left = None
		self.right = None

		self.attribute = None
		self.threshold = None
		self.classification = None


class Classifier:
	def __init__(self):
		self.tree_root = None
		self.already_chosen_attributes = np.empty([0], dtype=int)

	def fit(self, X, y):
		self._build_tree(X, y)

	def predict(self, X):
		predicted_labels = []
		current_node = self.tree_root

		for x_row in X:
			current_node = self.tree_root

			while current_node.attribute is not None and current_node.threshold is not None:
				if x_row[current_node.attribute] <= current_node.threshold:
					if current_node.left is not None:
						current_node = current_node.left
				else:  # > threshold
					if current_node.right is not None:
						current_node = current_node.right
				if current_node.right is None and current_node.left is None: # if leaf node
					break

			predicted_labels = np.append(predicted_labels, current_node.classification)

		return predicted_labels

	# PRIVATE FUNCTIONS:

	def _build_tree(self, data_copy, labels, tree=None):
		# Build tree up recursively

		# Initialise the tree
		if tree is None:
			tree = TreeNode()
			self.tree_root = tree

		best_gain = 0
		a_best = -1

		if len(data_copy.shape) > 1:
			threshold = None

			for i in range(data_copy.shape[1]):
				# if this attribute is not an ID and has not already been chosen
				if i != 6 and np.where(self.already_chosen_attributes == i)[0].size == 0:
					threshold, info_gain = self._get_best_threshold(data_copy, i, labels)
					if info_gain > best_gain:  # and i is not in already_chosen_attributes
						best_gain = info_gain
						a_best = i
						# add i to already_chosen_attributes
						self.already_chosen_attributes = np.append(self.already_chosen_attributes, i)

			if best_gain != 0:  # if 0 we must have split on all the attributes; no more info_gain to be gotten
				leftChild, rightChild = self._split_data_according_to_attribute(data_copy, a_best, threshold=threshold)

				tree.attribute = a_best
				tree.threshold = threshold

				if leftChild.size > 0:
					tree.left = TreeNode()
					tree.left.classification = self._get_dominant_classification(leftChild, labels)
					leftChild[:, a_best] = -1  # blank out entire attribute column so we don't split on it again
					self._build_tree(leftChild, labels, tree.left)
				if rightChild.size > 0:
					tree.right = TreeNode()
					tree.right.classification = self._get_dominant_classification(rightChild, labels)
					rightChild[:, a_best] = -1
					self._build_tree(rightChild, labels, tree.right)

	def _get_dominant_classification(self, data, labels):
		labels_for_data = self._get_corresponding_labels(data, labels)

		dominant_class = None

		ale_count = repr(labels_for_data).count("ale")
		stout_count = repr(labels_for_data).count("stout")
		lager_count = repr(labels_for_data).count("lager")

		if (ale_count >= stout_count) and (ale_count >= lager_count):
			dominant_class = "ale"
		elif (stout_count >= ale_count) and (stout_count >= lager_count):
			dominant_class = "stout"
		else:
			dominant_class = "lager"

		return dominant_class

	def _calculate_entropy(self, labels):
		sum = 0.0

		# remove blank entries
		labels = np.delete(labels, np.where(labels == None))
		labels = np.delete(labels, np.where(labels == ""))

		labels_unique = np.unique(labels.astype('str'), axis=0)

		for label in labels_unique:
			labels_copy = labels.copy()
			label_count = len(np.where(labels == label)[0])
			label_proportion = float(label_count) / float(labels.size)
			sum += -1 * label_proportion * (math.log(label_proportion, 2))
		return sum

	def _get_candidate_thresholds(self, data, labels, attr_index):
		# Make a copy of the data for sorting without affecting the original sort order
		data_for_sorting = data.copy()

		# Sort the data according to the attribute at the index
		data_for_sorting = data_for_sorting[data_for_sorting[:, attr_index].argsort()]

		# Create a list to hold candidate thresholds
		candidate_thresholds = []

		for sorted_index in range(len(data_for_sorting) - 1):
			current_unsorted_index = int(data_for_sorting[sorted_index][6])
			next_unsorted_index = int(data_for_sorting[sorted_index + 1][6])
			current_label_value = labels[current_unsorted_index]
			next_label_value = labels[next_unsorted_index]

			# Check if the classification changes
			if (current_label_value != next_label_value):
				# Get the midpoint attribute value between this row and the next
				current_attr_value = data_for_sorting[sorted_index][attr_index]
				next_attr_value = data_for_sorting[sorted_index + 1][attr_index]
				midpoint = (current_attr_value + next_attr_value) / 2

				# Add this to the list
				candidate_thresholds.append(midpoint)

		# Return the candidate list
		return candidate_thresholds

	def _split_data_according_to_attribute(self, data, attr_index, threshold=None):
		# Create LHS and RHS of dataset
		left_child = []
		right_child = []

		if (threshold != None):
			for row in data:
				if (row[attr_index] > threshold):
					np.delete(row, attr_index)  # remove the attribute itself
					right_child.append(row)
				else:
					np.delete(row, attr_index)  # remove the attribute itself
					left_child.append(row)

		return np.array(left_child), np.array(right_child)


	def _get_corresponding_labels(self, dataset, labels_whole_dataset):
		labels_dataset = []
		for row in dataset:
			index = int(row[6])
			labels_dataset.append(labels_whole_dataset[index])

		return labels_dataset

	def _get_best_threshold(self, data, attr_index, labels):
		candidate_thresholds = self._get_candidate_thresholds(data, labels, attr_index)
		info_gains = []
		for threshold in candidate_thresholds:
			info_gain, info_gain_ratio = self._calculate_info_gain(data, attr_index, labels, threshold)
			info_gains.append(info_gain)
		if len(info_gains) > 0:
			max_info_gain = max(float(sub) for sub in info_gains)
			corresponding_index = info_gains.index(max_info_gain)
			return candidate_thresholds[corresponding_index], max_info_gain
		else:
			return -1, 0

	def _calculate_info_gain(self, whole_dataset, attr_index, labels, threshold):
		datasets = self._split_data_according_to_attribute(whole_dataset, attr_index, threshold)

		entropy_whole_dataset = self._calculate_entropy(labels)

		sum = 0
		sum_entropy = 0
		for dataset in datasets:
			ratio = len(dataset) / len(whole_dataset)
			current_dataset_labels = self._get_corresponding_labels(dataset, labels)
			entropy_dataset = self._calculate_entropy(current_dataset_labels)
			sum_entropy += entropy_dataset
			sum += (ratio * entropy_dataset)
		info_gain = entropy_whole_dataset - sum
		info_gain_ratio = info_gain/sum_entropy
		return info_gain, info_gain_ratio


if __name__ == '__main__':
	number_of_runs = 10
	total_accuracy = 0

	for i in range(number_of_runs):
		data, labels = process_file("beer.txt")
		training_data, training_labels, testing_data, testing_labels = split_data_set(data, labels, 33.33)

		classifier = Classifier()
		classifier.fit(training_data, training_labels)
		prediction = classifier.predict(testing_data)

		run_accuracy = array_similarity(prediction, testing_labels)
		print("Accuracy for run", i + 1, ":", run_accuracy, "%")
		total_accuracy += run_accuracy

	total_accuracy /= number_of_runs

	print("Total accuracy of", number_of_runs, "runs:", round(total_accuracy, 2), "%")