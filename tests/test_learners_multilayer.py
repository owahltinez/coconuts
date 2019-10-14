''' Test Utils Module '''

import sys
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main

from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.sampledata.local import load_boston, load_titanic
from bananas.sampledata.synthetic import new_labels, new_line, new_mat9, new_poly, new_trig
from bananas.testing.learners import test_learner
from bananas.preprocessing.standard import StandardPreprocessor

from coconuts.learners.multilayer import MLPClassifier, MLPRegressor

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
        learner_kwargs = {}
        for learner in (MLPClassifier, MLPRegressor):
            self.assertTrue(test_learner(learner, *learner_args, **learner_kwargs))

    def test_learner_synthetic(self):
        opts = {'random_seed': 0}
        test_data = [
            (MLPRegressor, new_line(**opts), .95),  # Approximate a line
            (MLPRegressor, new_trig(**opts), .50),  # Approximate a sine curve
            (MLPRegressor, new_poly(**opts), .75),  # Approximate a 4th deg. poly
            (MLPClassifier, new_labels(**opts), .75),  # Correctly guess labels
            (MLPClassifier, new_mat9(**opts), 1)]  # Correctly guess labels
        for learner, dataset, target_score in test_data:
            pipeline = learner(verbose=True, **opts)
            pipeline.train(dataset.input_fn, max_score=target_score)
            self.assertGreaterEqual(pipeline.best_score_, target_score, dataset.name)

    def test_learner_datasets(self):
        opts = {'random_seed': 0}
        test_data = [
            (MLPRegressor, load_boston(**opts), .85),  # Boston housing dataset
            (MLPClassifier, load_titanic(**opts), .75)]  # Titanic dataset

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
