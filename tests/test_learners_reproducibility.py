''' Test Utils Module '''

import sys
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main

from bananas.sampledata.local import load_boston, load_titanic
from bananas.sampledata.synthetic import new_labels, new_line, new_mat9, new_poly

from coconuts.learners.linear import LinearRegressor, LogisticRegression

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

    def test_learner_reproducibility_synthetic(self):
        trials = 10
        opts = {'random_seed': 0}

        test_data = [
            (LinearRegressor, new_line),  # Approximate a line
            (LinearRegressor, new_poly),  # Approximate a 4th deg. poly
            (LogisticRegression, new_labels)]  # Correctly guess labels

        for learner, dataset in test_data:
            tmp = []
            X_list, y_list, scores_list = [], [], []

            for _ in range(trials):

                scores, X_tmp, y_tmp = [], [], []
                def step_callback(step):
                    for x in step.X_test: X_tmp.append(x)
                    for y in step.y_test: y_tmp.append(y)
                    scores.append(step.score)

                dataset_ = dataset(**opts)
                tmp.append(dataset_.input_fn()[0].tolist())
                pipeline = learner(verbose=False, **opts)
                pipeline.train(dataset_.input_fn, max_steps=100, callback=step_callback)

                X_list.append(X_tmp)
                y_list.append(y_tmp)
                scores_list.append(scores + [pipeline.best_score_])

            for i in range(1, trials):
                self.assertListEqual(tmp[0], tmp[i])
                self.assertListEqual(X_list[0], X_list[i])
                self.assertListEqual(y_list[0], y_list[i])
                self.assertListEqual(scores_list[0], scores_list[i])

if __name__ == '__main__':
    sys.exit(main())
