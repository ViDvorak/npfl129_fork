#!/usr/bin/env python3
# aa94a035-e929-11e9-9ce9-00505601122b 342b7533-e5d8-11e9-9ce9-00505601122b 5cbfb4bf-8474-4193-b7c1-61fec099ce08

import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    dataset.data = np.hstack([dataset.data, np.ones((dataset.data.shape[0], 1))])

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    train_data_transpose = np.transpose(train_data)
    weights = np.linalg.inv(train_data_transpose @ train_data) @ train_data_transpose @ train_target

    # TODO: Predict target values on the test set.
    test_predictions = test_data @ weights

    # TODO: Manually compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.mean(np.power(test_predictions - test_target, 2)))

    return rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(main_args)
    print("{:.2f}".format(rmse))
