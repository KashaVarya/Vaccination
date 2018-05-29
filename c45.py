import math


class C45:
    """Creates a decision tree with C4.5 algorithm"""

    def __init__(self, path_to_data, path_to_names):
        self.filePathToData = path_to_data
        self.filePathToNames = path_to_names
        self.data = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.tree = None

    def fetch_data(self):
        with open(self.filePathToNames, "r") as file:
            classes = file.readline()
            self.classes = [x.strip() for x in classes.split(",")]
            # add attributes
            for line in file:
                [attribute, values] = [x.strip() for x in line.split(":")]
                values = [x.strip() for x in values.split(",")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.filePathToData, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)

    def preprocess_data(self):
        for index, row in enumerate(self.data):
            for attr_index in range(self.numAttributes):
                if not self.is_attr_discrete(self.attributes[attr_index]):
                    self.data[index][attr_index] = float(self.data[index][attr_index])

    def print_tree(self):
        self.print_node(self.tree)

    def print_node(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + self.attributes[index] + " : ")
                        self.print_node(child, indent + "	")
            else:
                # numerical
                left_child = node.children[0]
                right_child = node.children[1]
                if left_child.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + left_child.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold) + " : ")
                    self.print_node(left_child, indent + "	")

                if right_child.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + right_child.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.print_node(right_child, indent + "	")

    def generate_tree(self):
        self.tree = self.recursive_generate_tree(self.data, self.attributes)

    def recursive_generate_tree(self, cur_data, cur_attributes):
        all_same = self.all_same_class(cur_data)

        if len(cur_data) == 0:
            # Fail
            return Node(True, "Fail", None)
        elif all_same is not False:
            # return a node with that class
            return Node(True, all_same, None)
        elif len(cur_attributes) == 0:
            # return a node with the majority class
            maj_class = self.get_maj_class(cur_data)
            return Node(True, maj_class, None)
        else:
            (best, best_threshold, splitted) = self.split_attribute(cur_data, cur_attributes)
            remaining_attributes = cur_attributes[:]
            remaining_attributes.remove(best)
            node = Node(False, best, best_threshold)
            node.children = [self.recursive_generate_tree(subset, remaining_attributes) for subset in splitted]
            return node

    def get_maj_class(self, cur_data):
        freq = [0] * len(self.classes)
        for row in cur_data:
            index = self.classes.index(row[-1])
            freq[index] += 1
        max_ind = freq.index(max(freq))
        return self.classes[max_ind]

    def all_same_class(self, data):
        for row in data:
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def is_attr_discrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def split_attribute(self, cur_data, cur_attributes):
        splitted = []
        max_ent = -1 * float("inf")
        best_attribute = -1
        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        for attribute in cur_attributes:
            index_of_attribute = self.attributes.index(attribute)
            if self.is_attr_discrete(attribute):
                # split cur_data into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                values_for_attribute = self.attrValues[attribute]
                subsets = [[] for _ in values_for_attribute]
                for row in cur_data:
                    for index in range(len(values_for_attribute)):
                        if row[i] == values_for_attribute[index]:
                            subsets[index].append(row)
                            break
                e = self.gain(cur_data, subsets)
                if e > max_ent:
                    max_ent = e
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
            else:
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                cur_data.sort(key=lambda x: x[index_of_attribute])
                for j in range(0, len(cur_data) - 1):
                    if cur_data[j][index_of_attribute] != cur_data[j + 1][index_of_attribute]:
                        threshold = (cur_data[j][index_of_attribute] + cur_data[j + 1][index_of_attribute]) / 2
                        less = []
                        greater = []
                        for row in cur_data:
                            if row[index_of_attribute] > threshold:
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(cur_data, [less, greater])
                        if e >= max_ent:
                            splitted = [less, greater]
                            max_ent = e
                            best_attribute = attribute
                            best_threshold = threshold
        return best_attribute, best_threshold, splitted

    def gain(self, union_set, subsets):
        # input : data and disjoint subsets of it
        # output : information gain
        s = len(union_set)
        # calculate impurity before split
        impurity_before_split = self.entropy(union_set)
        # calculate impurity after split
        weights = [len(subset) / s for subset in subsets]
        impurity_after_split = 0
        for i in range(len(subsets)):
            impurity_after_split += weights[i] * self.entropy(subsets[i])
        # calculate total gain
        total_gain = impurity_before_split - impurity_after_split
        return total_gain

    def entropy(self, data_set):
        s = len(data_set)
        if s == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in data_set:
            class_index = list(self.classes).index(row[-1])
            num_classes[class_index] += 1
        num_classes = [x / s for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num * self.log(num)
        return ent * -1

    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)


class Node:
    def __init__(self, is_leaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = is_leaf
        self.children = []
