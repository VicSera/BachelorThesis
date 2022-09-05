import os

from core.config import Config
import matplotlib.pyplot as plt
import numpy as np


def extract_file_results(file_obj, include_columns):
    header = file_obj.readline()
    columns = header[:-1].split(',')
    column_indices = {column: columns.index(column) for column in include_columns}
    values = {column: [] for column in include_columns}

    line = file_obj.readline()
    while line:
        split_values = line[:-1].split(',')
        for column in include_columns:
            value = float(split_values[column_indices[column]])
            values[column].append(value)
        line = file_obj.readline()

    return values


def create_graph_for_file(result_file_path, include_columns, name=None):
    with open(result_file_path, 'r') as file:
        values = extract_file_results(file, include_columns)

    print(values)
    for column in include_columns:
        line, = plt.plot(values[column])
        line.set_label(column)
    plt.legend()

    if name is not None:
        plt.savefig(name)
        plt.clf()


def create_all_graphs(root_path):
    graphs = {
        'losses': ['loss', 'validation_loss'],
        'metrics': ['accuracy', 'precision', 'recall']
    }

    sessions = os.listdir(root_path)
    for session in sessions:
        result_file = f'{root_path}/{session}/results.csv'

        for postfix, columns in graphs.items():
            create_graph_for_file(result_file, columns, name=f'{session}_{postfix}.png')


if __name__ == '__main__':
    # create_all_graphs(Config.SAVED_WEIGHTS_ROOT)

    bar_width = 0.3

    accuracies = [0.1476, 0.1473, 0.1561, 0.1898, 0.2040, 0.2054]
    precisions = [0.0893, 0.0873, 0.0969, 0.1194, 0.1361, 0.1374]
    recalls = [0.0756, 0.0745, 0.0879, 0.1038, 0.1187, 0.1198]

    br1 = np.arange(len(accuracies))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]

    # plt.bar(br1, validation_losses, color='r', label='Validation loss', width=bar_width)

    plt.bar(br1, accuracies, color='g', label='Accuracy', width=bar_width)
    plt.bar(br2, precisions, color='b', label='Precision', width=bar_width)
    plt.bar(br3, recalls, color='m', label='Recall', width=bar_width)

    plt.xlabel('Session')
    plt.ylabel('Value')
    plt.xticks([r + bar_width for r in range(len(accuracies))],
               [f'Session {i + 1}' for i in range(6)])

    plt.legend()
    plt.savefig('session-comparison')
    plt.show()
