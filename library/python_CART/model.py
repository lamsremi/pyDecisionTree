"""Implementation of a Decision Tree Classifier in pure python.

This implementation code is directly taken from Josh Gordon's implementation
in Machine Learning Recipes #8.

Ref:
* https://www.youtube.com/watch?v=LDRbO9a6XPU
"""
import os
import pickle

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
        self._params_path = "library/python_CART/params/"

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
        self._tree = build_tree(train_data, self._header)


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
        print_md_tree(self._tree, spacing="")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def build_tree(rows, header):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows, header)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def find_best_split(rows, header):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    Args:
        rows (list): list of records.
    Return:
        best_gain (float): the best gain.
        best_question (Question): the best question to ask.
    """
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val, header)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def class_counts(rows):
    """Counts the number of each type of example in a dataset
    Agrs:
        rows (list): list of list of inputs.
    Returns:
        counts (dict): counts of labels.
    """
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_md_tree(node, spacing=""):
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
    print(spacing + "{}".format(str(node.question).upper()))

    # Mardowns
    md, md_end = "├", "└"
    md_line, md_empty = "│", " "

    # Call this function recursively on the true branch
    print(spacing + md + '- True:')
    print_md_tree(node.true_branch, spacing + md_line + "     ")

    # Call this function recursively on the false branch
    print (spacing + md_end + '--> False:')
    print_md_tree(node.false_branch, spacing + md_empty + "     ")

# if __name__ == '__main__':

#     my_model = Model()

#     train_data = [
#         ['Green', 3, 'Apple'],
#         ['Yellow', 3, 'Apple'],
#         ['Red', 1, 'Grape'],
#         ['Red', 1, 'Grape'],
#         ['Yellow', 3, 'Lemon'],
#     ]

#     header = ["color", "diameter", "label"]

#     my_model.fit(train_data, header)

#     print_tree(my_model._tree)

#     my_model.persist_parameters("X")

#     testing_data = [
#         ['Green', 3],
#         ['Yellow', 3],
#         ['Red', 2],
#         ['Red', 1],
#         ['Yellow', 3],
#     ]

#     my_loaded_model = Model()

#     my_loaded_model.load_parameters("X")

#     prediction_data = my_loaded_model.predict(testing_data)

#     print(prediction_data)
