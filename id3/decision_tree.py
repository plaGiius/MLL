import pandas as pd
import math
import numpy as np

# Load the PlayTennis dataset
data = pd.read_csv("/Users/girishkrithik/Desktop/ML-LKF/id3/PlayTennis.csv")

# Define the ID3 algorithm function


def id3(data, target_attribute_name, attribute_names, default_class=None):
    # Count the number of each target class in the data
    classes, class_counts = np.unique(
        data[target_attribute_name], return_counts=True)

    # If all the data belongs to a single class, return that class
    if len(classes) == 1:
        return classes[0]

    # If there are no attributes left to split on, return the default class
    if len(attribute_names) == 0:
        return default_class

    # Calculate the entropy of the current data
    entropy = calculate_entropy(data[target_attribute_name], class_counts)

    # Initialize variables for tracking the best attribute and its information gain
    best_info_gain = -1
    best_attribute = None

    # Loop over all attributes and calculate their information gain
    for attribute in attribute_names:
        attribute_values, attribute_counts = np.unique(
            data[attribute], return_counts=True)

        # Calculate the weighted entropy of each possible value of the attribute
        weighted_entropy = 0
        for i in range(len(attribute_values)):
            subset = data[data[attribute] == attribute_values[i]]
            subset_classes, subset_class_counts = np.unique(
                subset[target_attribute_name], return_counts=True)
            weighted_entropy += (attribute_counts[i] / len(data)) * calculate_entropy(
                subset[target_attribute_name], subset_class_counts)

        # Calculate the information gain of the attribute
        info_gain = entropy - weighted_entropy

        # Update the best attribute and its information gain if this one is better
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute

    # Create a new decision tree node with the best attribute
    tree = {best_attribute: {}}

    # Remove the best attribute from the list of attribute names
    attribute_names = [
        attr for attr in attribute_names if attr != best_attribute]

    # Loop over the possible values of the best attribute and create a subtree for each one
    for value in np.unique(data[best_attribute]):
        # Recursively call the ID3 algorithm on the subset of data with this value of the best attribute
        subtree = id3(data[data[best_attribute] == value],
                      target_attribute_name, attribute_names, default_class)

        # Add this subtree to the main decision tree node
        tree[best_attribute][value] = subtree

    # If the default class is not None, add a subtree for missing attribute values
    if default_class is not None:
        tree["default"] = default_class
    return tree

# Define a function to calculate the entropy of a set of target classes


def calculate_entropy(target_attribute, class_counts):
    entropy = 0
    total = sum(class_counts)
    for count in class_counts:
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


# Define the attribute names and target attribute for the PlayTennis dataset
attribute_names = ["Outlook", "Temperature", "Humidity", "Windy"]
target_attribute_name = "PlayTennis"

# Run the ID3 algorithm on the PlayTennis dataset
decision_tree = id3(data, target_attribute_name, attribute_names)

# Print the resulting decision tree
print(decision_tree)
