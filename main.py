import csv
import math
from c45 import C45


def read_csv(filename):
    file = open(filename)
    reader = csv.reader(file, delimiter=",", quotechar="\"")

    data = [
        [
            cell
            for cell in row
        ][1:]
        for row in reader
    ][1:]

    data = [
       [
           int(cell)
           for cell in row
       ]
       for row in data
    ]

    file.close()
    print(data)
    return data


def entropy(data):
    entr = []
    n = len(data)
    m = len(data[0])
    for row in range(n):
        pk0l0 = 0
        pk1l0 = 0
        pk0 = data[row].count(0)
        pk1 = m - pk0

        for col in range(m):
            if data[row][col] == 0 and data[n - 1][col] == 0:
                pk0l0 += 1
            if data[row][col] == 1 and data[n - 1][col] == 0:
                pk1l0 += 1

        pk0l0 /= pk0
        pk0l1 = 1 - pk0l0

        pk1l0 /= pk1
        pk1l1 = 1 - pk1l0

        pk0 /= m
        pk1 = 1 - pk0

        if (pk0l0 == 0 or pk0l1 == 0) and (pk1l0 == 0 or pk1l1 == 0):
            ent = 0
        elif pk0l0 == 0 or pk0l1 == 0:
            ent = -(pk1 * (pk1l0 * math.log2(pk1l0) + pk1l1 * math.log2(pk1l1)))
        elif pk1l0 == 0 or pk1l1 == 0:
            ent = -(pk0 * (pk0l0 * math.log2(pk0l0) + pk0l1 * math.log2(pk0l1)))
        else:
            ent = -(pk0 * (pk0l0 * math.log2(pk0l0) + pk0l1 * math.log2(pk0l1)) +
                    pk1 * (pk1l0 * math.log2(pk1l0) + pk1l1 * math.log2(pk1l1)))

        entr.append(ent)

    return entr


def main():
    data = read_csv("data/test10.csv")
    entr = entropy(data)

    arr = {
        "x1": [0, 1, 0],
        "x2": [1, 1, 1],
        "x3": [0, 1, 1],
        "x4": [1, 1, 0],
        "y": [1, 0, 0, 1],
    }

    c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
    c1.fetchData()
    c1.preprocessData()
    c1.generateTree()
    c1.printTree()


if __name__ == "__main__":
    main()
