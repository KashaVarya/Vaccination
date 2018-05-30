from DecisionTreeNode import DecisionTreeNode
from prepare_data import prepare_data


def requests():
    data_set = prepare_data("data/Predictors_ver.csv", 83)  # prepare general data
    test_set = prepare_data("data/Test.csv", 82)  # prepare test data
    name_set = []

    for row in test_set.rows:
        name_set.append(row.pop(0))

    root = DecisionTreeNode(None)
    root = root.compute_decision_tree(data_set, None)

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set.rows:
        results.append(root.get_classification(instance))

    del root

    for i in range(len(name_set)):
        if results[i] == 1:
            print("{} - эпизодически болеющий ребенок, в проведении инвазивных исследований нет необходимости.".format(
                name_set[i]))
        else:
            print("{} - часто болеющий ребенок, проведение инвазивных исследований является необходимым.".format(
                name_set[i]))


if __name__ == "__main__":
    requests()
