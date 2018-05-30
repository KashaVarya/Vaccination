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
