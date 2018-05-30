import math


# DataSet class to store the csv data
class DataSet:
    def __init__(self, classifier):
        self.rows = []
        self.attributes = []
        self.attribute_types = []
        self.classifier = classifier
        self.class_col_index = None

    def preprocessing(self):
        # convert attributes that are numeric to floats
        for example in self.rows:
            for x in range(len(self.rows[0])):
                if self.attributes[x] == 'True':
                    example[x] = float(example[x])

    # Calculate the entropy of the current dataset
    def calculate_entropy(self):
        # get count of all the rows with classification 1
        ones = self.count_positives()

        # get the count of all the rows in the dataset.
        total_rows = len(self.rows)
        # from the above two we can get the count of rows with classification 0 too

        # Entropy formula is sum of p*log2(p). Referred the slides. P is the probability
        entropy = 0

        # probability p of classification 1 in total data
        p = ones / total_rows
        if p != 0:
            entropy += p * math.log(p, 2)
        # probability p of classification 0 in total data
        p = (total_rows - ones) / total_rows
        if p != 0:
            entropy += p * math.log(p, 2)

        # from the formula
        entropy = -entropy
        return entropy

    # Calculate the gain of a particular attribute split
    def calculate_information_gain(self, attr_index, val, entropy):
        classifier = self.attributes[attr_index]
        attr_entropy = 0
        total_rows = len(self.rows)
        gain_upper_dataset = DataSet(classifier)
        gain_lower_dataset = DataSet(classifier)
        gain_upper_dataset.attributes = self.attributes
        gain_lower_dataset.attributes = self.attributes
        gain_upper_dataset.attribute_types = self.attribute_types
        gain_lower_dataset.attribute_types = self.attribute_types

        for example in self.rows:
            if example[attr_index] >= val:
                gain_upper_dataset.rows.append(example)
            elif example[attr_index] < val:
                gain_lower_dataset.rows.append(example)

        if len(gain_upper_dataset.rows) == 0 or len(gain_lower_dataset.rows) == 0:
            return -1

        attr_entropy += gain_upper_dataset.calculate_entropy() * len(gain_upper_dataset.rows) / total_rows
        attr_entropy += gain_lower_dataset.calculate_entropy() * len(gain_lower_dataset.rows) / total_rows

        return entropy - attr_entropy

    # count number of rows with classification "1"
    def count_positives(self):
        count = 0
        class_col_index = None

        # find the index of classifier
        for a in range(len(self.attributes)):
            if self.attributes[a] == self.classifier:
                class_col_index = a
            else:
                class_col_index = len(self.attributes) - 1
        for i in self.rows:
            if i[class_col_index] == "1":
                count += 1
        return count


# the node class that will make up the tree
class DecisionTreeNode:
    def __init__(self, parent):
        self.classification = None
        self.attribute_split = None
        self.attribute_split_index = None
        self.attribute_split_value = None
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.height = None
        self.is_leaf_node = True

    # compute the decision tree recursively
    def compute_decision_tree(self, dataset, parent_node):
        node = DecisionTreeNode(parent_node)
        if parent_node is None:
            node.height = 0
        else:
            node.height = node.parent.height + 1

        # count_positives() will count the number of rows with classification "1"
        ones = dataset.count_positives()

        if len(dataset.rows) == ones:
            node.classification = 1
            node.is_leaf_node = True
            return node
        elif ones == 0:
            node.is_leaf_node = True
            node.classification = 0
            return node
        else:
            node.is_leaf_node = False

        # The index of the attribute we will split on
        splitting_attribute = None

        # The information gain given by the best attribute
        maximum_info_gain = 0

        split_val = None
        minimum_info_gain = 0.01

        entropy = dataset.calculate_entropy()

        # for each column of data
        for attr_index in range(len(dataset.rows[0])):
            if dataset.attributes[attr_index] != dataset.classifier:
                local_max_gain = 0
                local_split_val = None

                # these are the values we can split on, now we must find the best one
                attr_value_list = [example[attr_index] for example in dataset.rows]
                # remove duplicates from list of all attribute values
                attr_value_list = list(set(attr_value_list))

                if len(attr_value_list) > 100:
                    attr_value_list = sorted(attr_value_list)
                    total = len(attr_value_list)
                    ten_percentile = int(total / 10)
                    new_list = []
                    for x in range(1, 10):
                        new_list.append(attr_value_list[x * ten_percentile])
                    attr_value_list = new_list

                for val in attr_value_list:
                    # calculate the gain if we split on this value
                    # if gain is greater than local_max_gain, save this gain and this value
                    current_gain = dataset.calculate_information_gain(attr_index, val, entropy)

                    if current_gain > local_max_gain:
                        local_max_gain = current_gain
                        local_split_val = val

                if local_max_gain > maximum_info_gain:
                    maximum_info_gain = local_max_gain
                    split_val = local_split_val
                    splitting_attribute = attr_index

        if maximum_info_gain <= minimum_info_gain or node.height > 20:
            node.is_leaf_node = True
            node.classification = self.classify_leaf(dataset)
            return node

        node.attribute_split_index = splitting_attribute
        node.attribute_split = dataset.attributes[splitting_attribute]
        node.attribute_split_value = split_val

        left_dataset = DataSet(dataset.classifier)
        right_dataset = DataSet(dataset.classifier)

        left_dataset.attributes = dataset.attributes
        right_dataset.attributes = dataset.attributes

        left_dataset.attribute_types = dataset.attribute_types
        right_dataset.attribute_types = dataset.attribute_types

        for row in dataset.rows:
            if splitting_attribute is not None and row[splitting_attribute] >= split_val:
                left_dataset.rows.append(row)
            elif splitting_attribute is not None:
                right_dataset.rows.append(row)

        node.left_child = self.compute_decision_tree(left_dataset, node)
        node.right_child = self.compute_decision_tree(right_dataset, node)

        return node

    # Classify dataset
    @staticmethod
    def classify_leaf(dataset):
        ones = dataset.count_positives()
        total = len(dataset.rows)
        zeroes = total - ones
        if ones >= zeroes:
            return 1
        else:
            return 0

    # Validate row (for finding best score before pruning)
    def validate_row(self, row):
        if self.is_leaf_node:
            projected = self.classification
            actual = int(row[-1])
            if projected == actual:
                return 1
            else:
                return 0
        value = row[self.attribute_split_index]
        if value >= self.attribute_split_value:
            return self.left_child.validate_row(row)
        else:
            return self.right_child.validate_row(row)

    def validate_tree(self, dataset):
        total = len(dataset.rows)
        correct = 0
        for row in dataset.rows:
            # validate example
            correct += self.validate_row(row)
        return correct / total

    # Prune tree
    def prune_tree(self, root, validate_set, best_score):
        # if node is a leaf
        if self.is_leaf_node:
            self.parent.is_leaf_node = True
            self.parent.classification = self.classification
            if self.height < 20:
                new_score = root.validate_tree(validate_set)
            else:
                new_score = 0

            if new_score >= best_score:
                return new_score
            else:
                self.parent.is_leaf_node = False
                self.parent.classification = None
                return best_score
        # if its not a leaf
        else:
            new_score = self.left_child.prune_tree(root, validate_set, best_score)
            if self.is_leaf_node:
                return new_score
            new_score = self.right_child.prune_tree(root, validate_set, new_score)
            if self.is_leaf_node:
                return new_score

            return new_score

    # Final evaluation of the data
    def get_classification(self, example):
        if self.is_leaf_node:
            return self.classification
        else:
            if example[self.attribute_split_index] >= self.attribute_split_value:
                return self.left_child.get_classification(example)
            else:
                return self.right_child.get_classification(example)
