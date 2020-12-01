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