""" Test Utils Module """

import sys
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main

from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.sampledata.local import load_boston, load_titanic
from bananas.sampledata.synthetic import new_labels, new_line, new_3x3, new_poly
from bananas.preprocessing.standard import StandardPreprocessor
from bananas.testing.learners import test_learner

from coconuts.learners.linear import LinearRegressor, LogisticRegression

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

    def test_learner_builtin(self):
        learner_args = []
        learner_kwargs = {}
        for learner in (LinearRegressor, LogisticRegression):
            self.assertTrue(test_learner(learner, *learner_args, **learner_kwargs))

    def test_learner_synthetic(self):
        opts = dict(random_seed=0)
        test_data = [
            (LinearRegressor, new_line(**opts), 0.95),  # Approximate a line
            (LogisticRegression, new_labels(**opts), 0.55),  # Correctly guess labels
            (LinearRegressor, new_poly(**opts), 0.85),  # Approximate a 4th deg. poly
            (LinearRegressor, new_3x3(**opts), 0.95),  # 3x3 fuzzy matrix
        ]
        for learner, dataset, target_score in test_data:
            pipeline = learner(verbose=False, **opts)
            history = pipeline.train(dataset.input_fn, max_score=target_score, progress=True)
            self.assertGreaterEqual(max(history.scores), target_score, dataset.name)

    def test_learner_datasets(self):
        opts = dict(random_seed=0)
        test_data = [
            (LinearRegressor, load_boston(**opts), 0.45),  # Boston housing dataset
            (LogisticRegression, load_titanic(**opts), 0.75),  # Titanic dataset
        ]

        for learner, train_test_datasets, target_score in test_data:
            dataset, test_ds = train_test_datasets
            pipeline = Pipeline(
                [
                    PipelineStep(
                        name="preprocessor",
                        learner=StandardPreprocessor,
                        kwargs={
                            "continuous": dataset.continuous,
                            "categorical": dataset.categorical,
                        },
                    ),
                    PipelineStep(name="estimator", learner=learner, kwargs=opts),
                ]
            )

            history = pipeline.train(dataset.input_fn, max_score=target_score, progress=True)
            test_score = pipeline.score(*test_ds[:])
            self.assertGreaterEqual(max(history.scores), target_score, dataset.name)
            print("%s\t%.3f\t%.3f" % (dataset.name, max(history.scores), test_score))


if __name__ == "__main__":
    sys.exit(main())
