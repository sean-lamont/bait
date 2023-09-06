"""Tests for predictions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import parameterized
import numpy as np
# import tensorflow as tf
from experiments.holist.tests import mock_predictions_lib
from experiments.holist.agent import predictions

TEST_ARRAY = np.reshape(np.arange(100), (10, 10)).astype(float)
MOCK_PREDICTOR = mock_predictions_lib.MockPredictionsLib


def double(x):
  if x is None:
    return x
  else:
    return 2 * x


class PredictionsTest(parameterized.TestCase):

  def test_batch_array_with_none(self):
    result = predictions.batch_array(TEST_ARRAY, None)
    self.assertEqual(len(result), 1)
    self.assertListEqual(TEST_ARRAY.tolist(), result[0].tolist())

  def test_batch_array_with_batch_size_1(self):
    result = predictions.batch_array(TEST_ARRAY, 1)
    self.assertEqual(len(result), 10)

    for i in range(10):
      self.assertListEqual(np.expand_dims(TEST_ARRAY[i, :], 0).tolist(), result[i].tolist())

  def test_batch_array_with_batch_size_3(self):
    result = predictions.batch_array(TEST_ARRAY, 3)
    expected = [
        TEST_ARRAY[:3, :], TEST_ARRAY[3:6, :], TEST_ARRAY[6:9, :],
        TEST_ARRAY[9:, :]
    ]
    self.assertEqual(len(result), len(expected))
    for i in range(len(expected)):
      self.assertListEqual(expected[i].tolist(), result[i].tolist())

  def test_batch_array_with_batch_size_10(self):
    result = predictions.batch_array(TEST_ARRAY, 10)
    self.assertEqual(len(result), 1)
    self.assertListEqual(TEST_ARRAY.tolist(), result[0].tolist())

  def test_batch_array_with_batch_size_15(self):
    result = predictions.batch_array(TEST_ARRAY, 15)
    self.assertEqual(len(result), 1)
    self.assertListEqual(TEST_ARRAY.tolist(), result[0].tolist())

  def test_batch_array_strlist_with_batch_size_3(self):
    strlist = [str(i) for i in range(10)]
    result = predictions.batch_array(strlist, 3)
    expected = [strlist[:3], strlist[3:6], strlist[6:9], [strlist[9]]]
    print('result:', result)
    self.assertEqual(len(expected), len(result))
    for i in range(len(expected)):
      self.assertListEqual(expected[i], result[i])

  def test_batch_array_strlist_with_batch_size_none(self):
    strlist = [str(i) for i in range(10)]
    result = predictions.batch_array(strlist, None)
    self.assertEqual(len(result), 1)
    self.assertListEqual(result[0], strlist)

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_identity(self, max_batch_size):
    result = predictions.batched_run([TEST_ARRAY], lambda x: x, max_batch_size)
    self.assertListEqual(result.tolist(), TEST_ARRAY.tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_add(self, max_batch_size):
    result = predictions.batched_run(
        [TEST_ARRAY, TEST_ARRAY], lambda x, y: x + y, max_batch_size)
    self.assertListEqual(result.tolist(), (2.0 * TEST_ARRAY).tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_str_to_int_and_back(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    result = predictions.batched_run(
        [strlist], lambda l: np.array([[float(x)] for x in l]), max_batch_size)
    self.assertListEqual(result.tolist(), [[float(i)] for i in range(10)])

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_goal_embedding(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertListEqual(
        predictor.goal_embedding('goal').tolist(),
        predictor.batch_goal_embedding(['goal'])[0].tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_thm_embedding(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertListEqual(
        predictor.thm_embedding('thm').tolist(),
        predictor.batch_thm_embedding(['thm'])[0].tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_goal_embedding(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertListEqual(
        predictor.batch_goal_embedding(strlist).tolist(),
        predictor._batch_goal_embedding(strlist).tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_thm_embedding(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertListEqual(
        predictor.batch_thm_embedding(strlist).tolist(),
        predictor._batch_thm_embedding(strlist).tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_tactic_scores(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertListEqual(
        predictor.batch_tactic_scores(TEST_ARRAY).tolist(),
        predictor._batch_tactic_scores(TEST_ARRAY).tolist())

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_thm_scores(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    state = np.arange(10)
    dup_state = np.tile(np.arange(10), [10, 1])
    self.assertListEqual(
        predictor.batch_thm_scores(state, TEST_ARRAY).tolist(),
        predictor._batch_thm_scores(dup_state, TEST_ARRAY).tolist())
    self.assertListEqual(
        predictor.batch_thm_scores(state, TEST_ARRAY, tactic_id=4).tolist(),
        predictor._batch_thm_scores(dup_state, TEST_ARRAY, tactic_id=4).tolist())


if __name__ == '__main__':
  unittest.main()