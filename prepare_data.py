import csv
from DataSet import DataSet


def prepare_data(filename, n_attr):
    data_set = DataSet("")

    # Load data set
    with open(filename) as f:
        data_set.rows = [line for line in csv.reader(f, delimiter=",")]

    data_set.attributes = data_set.rows.pop(0)

    # this is used to generalize the code for other datasets.
    # true indicates numeric data. false in nominal data
    # example: data_set.attribute_types = ['false', 'true', 'false', 'false', 'true', 'true', 'false']
    data_set.attribute_types = ['false' for _ in range(n_attr)]
    data_set.classifier = data_set.attributes[-1]

    # find index of classifier
    data_set.class_col_index = data_set.attributes.index(data_set.classifier)

    # preprocessing the data_set
    data_set.preprocessing()

    return data_set