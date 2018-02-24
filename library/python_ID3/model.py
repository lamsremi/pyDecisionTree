"""Implementation of a Decision Tree Classifier in pure python.

Ref:
* https://en.wikipedia.org/wiki/ID3_algorithm
* https://www.youtube.com/watch?v=LDRbO9a6XPU
"""
import pickle
import os
import math


class Model():
    """Decision tree model
    """

    def __init__(self):
        """Initialize an instance.
        Attributes:
            _tree (Decision node Object)
        """
        self._tree = None
        self._header = None
        self._attributes = None
        self._params_path = "library/python_ID3/params/"
        # self._params_path = "params/"

    def predict(self, inputs_data=None):
        """Classify inputs.
        Args:
            inputs_data (list): list of inputs to classify.
        Return:
            outputs_data (list): list of predicted labels.
        """
        outputs_data = []
        for input_data in inputs_data:
            # Base case: we've reached a leaf
            predictions = classify(input_data, self._tree)
            # Get the arg of the max
            outputs_data.append(max(predictions, key=predictions.get))
            # outputs_data.append(predictions)
        return outputs_data

    def fit(self, train_data, header):
        """Build the tree.
        Args:
            train_data (list): inputs data.
        """
        self._header = header
        self._attributes = [attr for attr in range(len(header)-1)]
        # print("attributes : {}".format(self._attributes))
        self._tree = build_tree(train_data, self._attributes, self._header, 0)

    def persist_parameters(self, model_version):
        """Store the parameters of a version of model.
        """
        version_path = self._params_path + model_version
        # Create folder
        if not os.path.exists(version_path):
            os.mkdir(version_path)
        # Store the tree
        with open(version_path + "/tree.pkl","wb") as handle:
            pickle.dump(self._tree, handle)
        # Store the header
        with open(version_path + "/header.pkl","wb") as handle:
            pickle.dump(self._header, handle)

    def load_parameters(self, model_version):
        """Load the parameters of a version of model.
        """
        if model_version is not None:
            version_path = self._params_path + model_version
            # Load the tree
            with open(version_path + "/tree.pkl","rb") as handle:
                self._tree = pickle.load(handle)
            # Load the header
            with open(version_path + "/header.pkl","rb") as handle:
                self._header = pickle.load(handle)

    def display_tree(self):
        """Dispay a tree."""
        print_md_tree(self._tree, self._header, spacing="")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Val
    value = row[node.attribute]
    # Decide which child branch to follow
    return classify(row, node.child_branches["{}".format(value)])

def build_tree(rows, initial_attributes, header, value):
    """Builds the tree recursively.
    """
    # If :
    # * all example are positive
    # * all example are negative
    # * number of predicting attributes is empty
    # Return a leaf
    if len(class_counts(rows)) == 1 or initial_attributes == []:
        return Leaf(rows, value=value)

    # Find best attribute and the resulting set of values
    attribute = find_best_attribute(rows, initial_attributes, header)
    # Set attributes
    attributes = initial_attributes.copy()
    # Remove the attribute
    attributes.remove(attribute)
    # Then partition using the attribute that best classify
    rows_sets, values = partition(rows, attribute)

    # Initialize child branches
    child_branches = {}
    # For each branch
    for rows_set, val in zip(rows_sets, values):
        # Recursively build the tree
        child_branches[val] = build_tree(rows_set, attributes, header, val)

    # Return a Decision node.
    return Decision_Node(attribute, child_branches, value=value)


def find_best_attribute(rows, attributes, header):
    """Find the best attributes to partition the rows.
    Args:
        rows (list): list of records.
    Return:
        best_gain (float): the best gain.
        best_question (Question): the best question to ask.
    """
    best_gain = 0  # keep track of the best information gain
    best_attribute = None # keep train of the feature / value that produced it
    # Current uncertainty
    current_uncertainty = entropy(rows)
    # Iterate through each attribute
    for col in attributes:
        # print("col : {}".format(col))
        # # Get the list of unique values
        # vals = unique_vals(rows, col)  # unique values in the column
        # print("vals : {}".format(vals))
        # Parition
        rows_sets, vals = partition(rows, col)
        # print("rows_sets : {}".format(rows_sets))
        # Compute the gain of information
        gain = info_gain(rows_sets, current_uncertainty)
        # If greater than the best set it
        if gain >= best_gain:
            best_gain, best_attribute = gain, col
    # Return
    return best_attribute


def partition(rows, attribute):
    """Partitions a dataset.
    """
    vals = unique_vals(rows, attribute)
    # print("vals : {}".format(vals))
    rows_sets = []
    for val in vals:
        # print("val : {}".format(val))
        rows_set = []
        for row in rows:
            if row[attribute] == val:
                rows_set.append(row)
        # print("rows_set : {}".format(rows_set))
        rows_sets.append(rows_set)
    return rows_sets, vals


def info_gain(childs, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    child nodes.
    Args:
        childs (list): list of sets of rows.
        current_incertainty (float): current uncertainty.
    """
    # Total nuùber of element
    number_element = sum([len(child) for child in childs])
    # Gain
    gain = current_uncertainty
    for child in childs:
        gain -= float(len(child) / number_element) * entropy(child)
    return gain


def entropy(rows):
    """Calculate the Entropy Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl*math.log2(prob_of_lbl)
    return impurity


def class_counts(rows):
    """Counts the number of each type of example in a dataset
    Agrs:
        rows (list): list of list of inputs.
    Returns:
        counts (dict): counts of labels.
    """
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # print("row : {}".format(row))
        # in our dataset format, the label is always the last column
        label = row[-1]
        # print("counts : {}".format(counts))
        # print("label : {}".format(label))
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, value):
        # print("leaf created")
        self.predictions = class_counts(rows)
        self.value = value


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 attribute,
                 child_branches,
                 value):
        # print("attribute: {}".format(attribute))
        self.attribute = attribute
        self.value = value
        self.child_branches = child_branches


def print_md_tree(node, header, spacing=""):
    """World's most elegant tree printing function."""
    # spacing
    # print("spacing length : {}".format(len(spacing)))
    # print("spacing value : '{}'".format(spacing))
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        spacing_pred = spacing[:-3]
        print(spacing_pred + "└-> predict", node.predictions)
        return
    # Print the attribute
    print(spacing + "{}".format(header[node.attribute].upper()))

    # Compute the length of the dict
    n_keys = len(list(node.child_branches.keys()))

    # Mardowns
    md, md_end = "├", "└"
    md_line, md_empty = "│", " "
    count = 1
    # Print each children
    for val, child in node.child_branches.items():
        # Set md val
        md_val = md if count != n_keys else md_end
        # Print the value
        print(spacing + md_val + "- {}:".format(child.value))
        # Set markdown for predict
        md_pred = md_line if count != n_keys else md_empty
        # Call this function recursively on the true branch
        print_md_tree(child, header, spacing + md_pred + "     ")
        # Increment count
        count += 1



# if __name__ == '__main__':

#     my_model = Model()

#     train_data = [
#         ['Square', 'Blue', '3', 'Apple'],
#         ['Square', 'Yellow', '3', 'Apple'],
#         ['Square', 'Red', '1', 'Grape'],
#         ['Oval', 'Red', '1', 'Cider'],
#         ['Round', 'Green', '3', 'Lemon'],
#         ['Round', 'Red', '1', 'Grape'],
#         ['Round', 'Green', '2', 'Grape'],
#         ['Triangle', 'Yellow', '3', 'Lemon']
#     ]

#     header = ["Shape", "Color", "Diameter", "Label"]

#     my_model.fit(train_data, header)

#     print_md_tree(my_model._tree, header)

#     my_model.persist_parameters("X")

#     testing_data = [data[:-1] for data in train_data]

#     my_loaded_model = Model()

#     my_loaded_model.load_parameters("X")

#     prediction_data = my_loaded_model.predict(testing_data)

#     print(prediction_data)