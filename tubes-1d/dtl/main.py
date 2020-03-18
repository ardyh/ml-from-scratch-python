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

    #prune tennis tree
    #initialize train and test data
    training_data1=data_tennis.iloc[:10]
    training_data2=data_tennis.iloc[11]
    training_data=training_data1.append(training_data2)
    validate_data=data_tennis.iloc[[10,12,13]]
    #print initial tennis tree with training data
    tree = tree_myC45(training_data, 'play')
    root = tree.make_tree()

    print('')
    print('-------initial tree-------')
    print('Post prunning')
    tree.print_tree(root, 0, 2)
    #prune and print pruned tree
    tree.post_pruning(validate_data)
    print('-------pruned tree-------')
    tree.print_tree(root, 0, 2)

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

    #prune iris tree
    #initialize train and test data
    randomized_iris_data = iris_data.sample(frac=1).reset_index().drop('index', axis=1)
    iris_train_data = randomized_iris_data.iloc[0:120]
    iris_test_data = randomized_iris_data.iloc[120:]
    #make pruned tree with rule_post_pruned
    prune_tree = tree_myC45(iris_train_data, 'label')
    root_prune_tree = prune_tree.make_tree()
    print('-------initial tree-------')
    prune_tree.print_tree(root_prune_tree, 0, 2)
    pruned_rules = prune_tree.rule_post_pruning(iris_test_data)
    #printing pruned tree
    print('-------rule pruned tree-------')
    for pruned_rule in pruned_rules:
        print(pruned_rule[0])

main()