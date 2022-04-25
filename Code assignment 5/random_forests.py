from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def main():
    label_encoder = LabelEncoder()
    label_encoder.fit(['p', 'e'])
    df = pd.read_csv('MushroomDataset/secondary_data.csv', sep=';')
    train, test = train_test_split(df, test_size=0.2)
    y_train = label_encoder.transform(train['class'])
    y_test = label_encoder.transform(test['class'])

    x_train = train.drop('class', axis=1)
    x_test = test.drop('class', axis=1)
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)

    train_models(x_train, y_train, x_test, y_test)

    '''
    tree = DecisionTreeClassifier(max_depth=10, min_samples_split=500, ccp_alpha=0.001)
    forest = RandomForestClassifier(max_depth=10, min_samples_split=500, ccp_alpha=0.001)
    tree.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    print('tree')
    tree_predictions = tree.predict(x_test)
    print_metrics(y_test, tree_predictions)

    print('forest')
    forest_predictions = forest.predict(x_test)
    print_metrics(y_test, forest_predictions)
    '''


def train_models(x_train, y_train, x_test, y_test):
    trees = []
    forests = []
    parameters = ['max depth', 'n trees', 'ccp alpha']
    xticks = []
    depths = [5, 10, 15, 20, 25, 30]
    xticks.append(depths)
    trees.append([DecisionTreeClassifier(max_depth=depth) for depth in depths])
    forests.append([RandomForestClassifier(max_depth=depth, n_jobs=-1) for depth in depths])
    n_estimators = [5, 10, 20, 40, 80, 160]
    xticks.append(n_estimators)
    trees.append([DecisionTreeClassifier(max_depth=10) for _ in n_estimators])
    forests.append([RandomForestClassifier(n_estimators=n, max_depth=10, n_jobs=-1) for n in n_estimators])
    ccp_alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    xticks.append(ccp_alphas)
    trees.append([DecisionTreeClassifier(ccp_alpha=alpha) for alpha in ccp_alphas])
    forests.append([RandomForestClassifier(ccp_alpha=alpha, n_jobs=-1) for alpha in ccp_alphas])

    for x, parameter in enumerate(parameters):
        tree_accuracies = []
        tree_precisions = []
        tree_recalls = []
        forest_accuracies = []
        forest_precisions = []
        forest_recalls = []

        for y, value in enumerate(xticks[x]):
            tree = trees[x][y]
            forest = forests[x][y]
            tree.fit(x_train, y_train)
            forest.fit(x_train, y_train)
            tree_predictions = tree.predict(x_test)
            forest_predictions = forest.predict(x_test)
            tree_accuracies.append(accuracy_score(y_test, tree_predictions))
            tree_precisions.append(precision_score(y_test, tree_predictions))
            tree_recalls.append(recall_score(y_test, tree_predictions))
            forest_accuracies.append(accuracy_score(y_test, forest_predictions))
            forest_precisions.append(precision_score(y_test, forest_predictions))
            forest_recalls.append(recall_score(y_test, forest_predictions))

        plt.title(f'Performance metrics per {parameter}')
        plt.xlabel(parameter)
        plt.xticks(list(range(len(xticks[x]))), xticks[x])
        plt.ylabel('metric score')
        plt.plot(tree_accuracies, '--', label='tree accuracy', color='red')
        plt.plot(tree_precisions, '--', label='tree precision', color='blue')
        plt.plot(tree_recalls, '--', label='tree recall', color='orange')
        plt.plot(forest_accuracies, label='forest accuracy', color='red')
        plt.plot(forest_precisions, label='forest precision', color='blue')
        plt.plot(forest_recalls, label='forest recall', color='orange')
        plt.legend()
        plt.show()


def print_metrics(y_test, predictions):
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'Precision: {precision_score(y_test, predictions)}')
    print(f'Recall: {recall_score(y_test, predictions)}')


if __name__ == '__main__':
    main()
