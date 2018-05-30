import csv
from DataSet import DataSet
from DecisionTreeNode import DecisionTreeNode


def requests():
    data_set = DataSet("")
    test_set = DataSet("")
    name_set = []

    filename_data = "data/Predictors_ver.csv"  # file with general data
    filename_test = "data/Test.csv"  # file with test data
    n = 83  # number of attributes

    # Load data set
    with open(filename_data) as f1:
        data_set.rows = [tuple(line) for line in csv.reader(f1, delimiter=",")]
    with open(filename_test) as f2:
        test_set.rows = [line for line in csv.reader(f2, delimiter=",")]

    for row in test_set.rows:
        name_set.append(row.pop(0))

    data_set.attributes = data_set.rows.pop(0)
    test_set.attributes = test_set.rows.pop(0)

    # this is used to generalize the code for other datasets.
    # true indicates numeric data. false in nominal data
    # example: data_set.attribute_types = ['false', 'true', 'false', 'false', 'true', 'true', 'false']
    data_set.attribute_types = ['false' for _ in range(n)]
    test_set.attribute_types = ['false' for _ in range(n - 1)]
    data_set.classifier = data_set.attributes[-1]

    # find index of classifier
    data_set.class_col_index = data_set.attributes.index(data_set.classifier)

    # preprocessing the data_set
    data_set.preprocessing()
    test_set.preprocessing()

    root = DecisionTreeNode(None)
    root = root.compute_decision_tree(data_set, None)

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set.rows:
        results.append(root.get_classification(instance))

    del root

    for i in range(1, len(name_set)):
        if results[i - 1] == 1:
            print("{} - эпизодически болеющий ребенок, в проведении инвазивных исследований нет необходимости.".format(
                name_set[i]))
        else:
            print("{} - часто болеющий ребенок, проведение инвазивных исследований является необходимым.".format(
                name_set[i]))


if __name__ == "__main__":
    requests()
