import math
import numpy as np


def calculate_entropy(labels):
    sum = 0.0
    labels_unique = np.unique(labels, axis=0)
    print("Calculating entropy...")
    print("Labels:")
    print(labels_unique)
    for label in labels_unique:
        label_count = len(labels[labels == label])
        label_proportion = float(label_count) / float(len(labels))
        print(label + " proportion = " + str(label_count) + "/" + str(len(labels)) + " = " + str(label_proportion))
        sum += -1 * label_proportion * (math.log(label_proportion, 2))
    print("Entropy = " + str(sum))
    return sum


def get_candidate_thresholds(data, labels, attr_index):
    # Make a copy of the data for sorting without affecting the original sort order
    data_for_sorting = data.copy()

    # Sort the data according to the attribute at the index
    data_for_sorting = data_for_sorting[data_for_sorting[:, attr_index].argsort()]

    # Create a list to hold candidate thresholds
    candidate_thresholds = []

    for sorted_index in range(len(data_for_sorting) - 1):
        current_unsorted_index = int(data_for_sorting[sorted_index][9])
        next_unsorted_index = int(data_for_sorting[sorted_index + 1][9])
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

def split_data(data, attr_index, threshold=None):
    # Create LHS and RHS of dataset
    new_data_2 = []
    new_data_1 = []

    if (threshold != None):
        for row in data:
            if(row[attr_index] > threshold):
                new_data_2.append(row)
            else:
                new_data_1.append(row)
    ### TODO: For discrete values

    return [np.array(new_data_1), np.array(new_data_2)]

def get_corresponding_labels(dataset, labes_whole_dataset):
    labels_dataset = []
    for row in dataset:
        index = int(row[9])
        try:
            labels_dataset.append(labes_whole_dataset[index])
        except IndexError:
            pass

    return labels_dataset

def calculate_info_gain(whole_dataset, attr_index, labels):
    ### TODO: For discrete values
    ### TODO: Remove hard coded value and iterate over candidate thresholds
    datasets = split_data(whole_dataset,attr_index, 40)

    entropy_whole_dataset = calculate_entropy(labels)

    sum = 0
    for dataset in datasets:
        ratio = len(dataset)/len(whole_dataset)
        labels = get_corresponding_labels(dataset, labels)
        entropy_dataset = calculate_entropy(labels)
        sum += (ratio*entropy_dataset)

    return entropy_whole_dataset - sum