''' Test Convolution Module '''

import sys
import time
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main

from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.sampledata.local import load_boston, load_titanic
from bananas.sampledata.synthetic import new_labels, new_line, new_mat9, new_poly, new_trig
from bananas.testing.learners import test_learner
from bananas.preprocessing.standard import StandardPreprocessor

from coconuts.learners.convolution import CNNClassifier, CNNRegressor

# Show traceback for all warninngs
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
        stats.sort_stats('cumtime')
        stats.print_stats(20)

    def test_learner_builtin(self):
        learner_args = []
        learner_kwargs = {'kernel_size': 1, 'padding': 0, 'maxpool_size': 1}
        for learner in (CNNClassifier, CNNRegressor):
            self.assertTrue(test_learner(learner, *learner_args, **learner_kwargs))

    def test_learner_synthetic(self):
        opts = {'random_seed': 0}
        learner_kwargs = {'kernel_size': 1, 'padding': 0, 'maxpool_size': 1, 'random_seed': 0}
        test_data = [
            (CNNRegressor, new_line(**opts), .95),  # Approximate a line
            (CNNRegressor, new_trig(**opts), .30),  # Approximate a sine curve
            (CNNRegressor, new_poly(**opts), .85),  # Approximate a 4th deg. poly
            (CNNClassifier, new_labels(**opts), .80),  # Correctly guess labels
            (CNNClassifier, new_mat9(**opts), 1)]  # FIXME: Correctly guess matrix
        for learner, dataset, target_score in test_data:
            pipeline = learner(verbose=True, **learner_kwargs)
            pipeline.train(dataset.input_fn, max_score=target_score)
            self.assertGreaterEqual(pipeline.best_score_, target_score, dataset.name)

    def test_learner_datasets(self):
        opts = {'random_seed': 0}
        test_data = [
            (CNNRegressor, load_boston(**opts), .75),  # Boston housing dataset
            (CNNClassifier, load_titanic(**opts), .75)]  # Titanic dataset

        opts['kernel_size'] = 3
        for learner, train_test_datasets, target_score in test_data:
            train_ds, test_ds = train_test_datasets
            pipeline = Pipeline([
                PipelineStep(name='preprocessor', learner=StandardPreprocessor, kwargs={
                    'continuous': train_ds.continuous, 'categorical': train_ds.categorical}),
                PipelineStep(name='estimator', learner=learner, kwargs=opts)], verbose=True)

            pipeline.train(train_ds.input_fn, max_score=target_score)
            test_score = pipeline.score(*test_ds[:])
            self.assertGreaterEqual(pipeline.best_score_, target_score, train_ds.name)
            print('%s\t%.3f\t%.3f' % (train_ds.name, pipeline.best_score_, test_score))

if __name__ == '__main__':
    sys.exit(main())
