#!/usr/bin/env python3
# aa94a035-e929-11e9-9ce9-00505601122b 342b7533-e5d8-11e9-9ce9-00505601122b 5cbfb4bf-8474-4193-b7c1-61fec099ce08

import argparse

import numpy as np
import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate binary classification data with labels {-1, 1}.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=args.seed)
    target = 2 * target - 1

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    data = np.hstack((data, np.ones( (data.shape[0], 1) ) ))

    print("data size:", data.shape)
    print("target size:", target.shape)

    # Generate initial perceptron weights.
    weights = np.zeros(data.shape[1])

    done = False
    while not done:
        permutation = generator.permutation(data.shape[0])

        # TODO: Implement the perceptron algorithm, notably one iteration
        # over the training data in the order of `permutation`. During the
        # training data iteration, perform the required updates to the `weights`
        # for incorrectly classified examples. If all training instances are
        # correctly classified, set `done=True`, otherwise set `done=False`.
        
        has_changed = False
        for input, expected_result in zip (data[permutation], target[permutation]):
            print("input:", input)
            # compute prediction
            potential = sum( weights * input )

            if potential >= 0:
                prediction = 1
            else:
                prediction = -1
            
            # adjust the wights
            
            if expected_result < prediction:
                weights = weights - input
                has_changed = True
            elif expected_result > prediction:
                weights = weights + input
                has_changed = True

        done = not has_changed

        if args.plot and not done:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.scatter(data[:, 0], data[:, 1], c=target)
            xs = np.linspace(*plt.gca().get_xbound() + (50,))
            ys = np.linspace(*plt.gca().get_ybound() + (50,))
            plt.contour(xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0])
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
