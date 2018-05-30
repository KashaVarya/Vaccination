import copy
import time
import math
from DecisionTreeNode import DecisionTreeNode
from prepare_data import prepare_data


def training_and_testing():
    dataset = prepare_data("data/Predictors_ver.csv", 83)
    print("Number of records: %d" % len(dataset.rows))

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
