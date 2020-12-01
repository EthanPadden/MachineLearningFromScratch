import math
import numpy as np

def calculate_entropy(data, labels):
    sum = 0.0
    labels_unique = np.unique(labels, axis=0)
    print("Calculating entropy...")
    print("Labels:")
    print(labels_unique)
    for label in labels_unique:
        label_count = len(labels[labels == label])
        label_proportion = float(label_count)/float(len(labels))
        print(label + " proportion = " + str(label_count) + "/" + str(len(labels)) + " = " + str(label_proportion))
        sum += -1*label_proportion*(math.log(label_proportion, 2))
    print("Entropy = " + str(sum))
    return sum

def get_candidate_thresholds(mappings, data, labels, attr_index):
    pass
    ### TODO
    # # Sort the data and leb according to the attribute at the index
    # print("Getting candidate thresholds for attribute at index " + str(attr_index))
    # sorted(data, key=lambda x: x[1])
    # data = mappings[0]
    # data = data[data[:, attr_index].argsort()]

    # Iterate through the rows:
    #   If the classification changes:
    #       Get the midpoint attribute value between this row and the last
    #       Add it to the candidate list

    # for row in data:


    # Return the candidate list