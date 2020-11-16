""" Test Convolution Module """

import sys
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main

from bananas.sampledata.local import load_boston, load_titanic
from bananas.sampledata.synthetic import new_labels, new_line, new_3x3, new_poly, new_trig
from bananas.hyperparameters.bruteforce import BruteForce

from coconuts.learners.convolution import CNNClassifier, CNNRegressor
from coconuts.learners.linear import LogisticRegression, LinearRegressor
from coconuts.learners.multilayer import MLPClassifier, MLPRegressor

# Show traceback for all warnings
from bananas.utils.misc import warn_with_traceback

warnings.showwarning = warn_with_traceback


# pylint: disable=missing-docstring
class TestUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiler = cProfile.Profile()
        cls.profiler.enable()

    @classmethod
    def tearDownClass(cls):
        stats = Stats(cls.profiler)
        stats.strip_dirs()
        stats.sort_stats("cumtime")
        stats.print_stats(20)

    def test_learner_synthetic(self):
        opts = dict(random_seed=0)
        learners_classifiers = [LogisticRegression, MLPClassifier, CNNClassifier]
        learners_regressors = [LinearRegressor, MLPRegressor, CNNRegressor]
        test_data = [
            (learners_regressors, new_line(**opts), 0.95),  # Approximate a line
            (learners_regressors, new_trig(**opts), .75),  # Approximate a sine curve
            (learners_regressors, new_poly(**opts), 0.85),  # Approximate a 4th deg. poly
            (learners_classifiers, new_labels(**opts), 0.80),  # Correctly guess labels
            (learners_regressors, new_3x3(**opts), 0.90),  # 3x3 fuzzy matrix
        ]
        for learners, dataset, target_score in test_data:
            pipeline = BruteForce(dataset, learners, n_jobs=4)
            history = pipeline.train(dataset.input_fn, max_score=target_score, progress=True)
            self.assertGreaterEqual(max(history.scores), target_score, dataset.name)

    def test_learner_datasets(self):
        opts = dict(random_seed=0)
        learners_classifiers = [LogisticRegression, MLPClassifier, CNNClassifier]
        learners_regressors = [LinearRegressor, MLPRegressor, CNNRegressor]
        test_data = [
            (learners_regressors, load_boston(**opts), 0.85),  # Boston housing dataset
            (learners_classifiers, load_titanic(**opts), 0.75),  # Titanic dataset
        ]

        for learners, train_test_datasets, target_score in test_data:
            dataset, test_ds = train_test_datasets
            pipeline = BruteForce(dataset, learners, n_jobs=4)
            history = pipeline.train(dataset.input_fn, max_score=target_score, progress=True)
            test_score = pipeline.score(*test_ds[:])
            self.assertGreaterEqual(max(history.scores), target_score, dataset.name)
            print("%s\t%.3f\t%.3f" % (dataset.name, max(history.scores), test_score))


if __name__ == "__main__":
    sys.exit(main())
