from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import math
from node import Node
from myC45 import Tree as tree_myC45
from myID3 import Tree as tree_myID3

def main():
    #read data tennis
    print('Data Tennis')
    data_tennis = pd.read_csv("play_tennis.csv")
    print(data_tennis.head())

    #make tennis tree
    data_tennis = data_tennis.drop('day', axis=1)
    tree_tennis = tree_myID3(data_tennis, 'play', use_info_gain=True)
    root_tennis = tree_tennis.make_tree()
    tree_tennis.print_tree(root_tennis, 0, 2)

    #predict with tennis tree
    print()
    print(tree_tennis.predict(data_tennis.tail(4)))

    #read data iris
    print()
    print('Data Iris')
    load, target = load_iris(return_X_y=True)
    iris_data = pd.DataFrame(load, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_data['label'] = pd.Series(target)
    print(iris_data)

    #make iris tree
    print()
    tree_iris = tree_myC45(iris_data, 'label')
    root_iris = tree_iris.make_tree()
    tree_iris.print_tree(root_iris, 0, 2)

    #predict with iris tree
    print()
    test_data = iris_data.sort_values(by='sepal_width').tail(10)
    print(tree_iris.predict(test_data))

main()