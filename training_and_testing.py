import copy
import time
import math
import csv
from DecisionTreeClassifier import DataSet, DecisionTreeNode


def training_and_testing():
    dataset = DataSet("")

    # Load data set
    filename = "data/Predictors_ver.csv"
    with open(filename) as f:
        dataset.rows = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print("Number of records: %d" % len(dataset.rows))

    dataset.attributes = dataset.rows.pop(0)

    # this is used to generalize the code for other datasets.
    # true indicates numeric data. false in nominal data
    # example: dataset.attribute_types = ['false', 'true', 'false', 'false', 'true', 'true', 'false']
    n = 83
    dataset.attribute_types = ['false' for _ in range(n)]
    dataset.classifier = dataset.attributes[-1]

    # find index of classifier
    dataset.class_col_index = dataset.attributes.index(dataset.classifier)

    # preprocessing the dataset
    dataset.preprocessing()

    training_set = copy.deepcopy(dataset)
    training_set.rows = []
    test_set = copy.deepcopy(dataset)
    test_set.rows = []
    validate_set = copy.deepcopy(dataset)
    validate_set.rows = []
    # Split training/test sets
    # You need to modify the following code for cross validation.

    runs = 10
    # Stores accuracy of the 10 runs
    accuracy = []
    start = time.clock()
    for k in range(runs):
        print("\nDoing fold ", k + 1)
        training_set.rows = [x for i, x in enumerate(dataset.rows) if i % runs != k]
        test_set.rows = [x for i, x in enumerate(dataset.rows) if i % runs == k]

        print("Number of training records: %d" % len(training_set.rows))
        print("Number of test records: %d" % len(test_set.rows))
        root = DecisionTreeNode(None)
        root = root.compute_decision_tree(training_set, None)

        # Classify the test set using the tree we just constructed
        results = []
        for instance in test_set.rows:
            result = root.get_classification(instance)
            results.append(str(result) == str(instance[-1]))

        # Accuracy
        acc = float(results.count(True)) / float(len(results))
        print("Accuracy: %.4f" % acc)

        # pruning code currently disabled
        # best_score = validate_tree(root, validate_set)
        # post_prune_accuracy = 100*prune_tree(root, root, validate_set, best_score)
        # print "Post-pruning score on validation set: " + str(post_prune_accuracy) + "%"
        accuracy.append(acc)
        del root

    mean_accuracy = math.fsum(accuracy) / 10
    print("\nTotal accuracy: {:.2%}".format(mean_accuracy))
    print("\nTook %f secs" % (time.clock() - start))

    # Writing results to a file (DO NOT CHANGE)
    f = open("result.txt", "w")
    f.write("Accuracy: %.4f" % mean_accuracy)
    f.close()


if __name__ == "__main__":
    training_and_testing()
