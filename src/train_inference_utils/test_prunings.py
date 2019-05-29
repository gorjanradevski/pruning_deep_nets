import pytest
import tensorflow as tf
import numpy as np

from train_inference_utils.prunings import unit_pruning, weight_pruning


@pytest.fixture
def simple_weight_matrix():
    return tf.Variable(tf.constant([[6, 3, 9], [1, 5, 7], [4, 8, 2]], dtype=tf.float32))


@pytest.fixture
def unit_pruned_simple_weight_matrix():
    return np.array([[6, 0, 9], [0, 5, 7], [4, 8, 0]], dtype=np.float32)


@pytest.fixture
def weight_pruned_simple_weight_matrix():
    return np.array([[0, 3, 9], [0, 5, 7], [0, 8, 2]], dtype=np.float32)


@pytest.fixture
def complex_weight_matrix():
    return tf.Variable(tf.random.uniform([10, 10], dtype=tf.float32))


@pytest.fixture
def k_unit_prune():
    return 0.34


@pytest.fixture
def k_weight_prune():
    return 0.3


def test_basic_unit_prune(
    simple_weight_matrix, k_unit_prune, unit_pruned_simple_weight_matrix
):

    pruned_w = unit_pruning(simple_weight_matrix, k_unit_prune)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_equal(sess.run(pruned_w), unit_pruned_simple_weight_matrix)


def test_basic_weight_prune(
    simple_weight_matrix, k_weight_prune, weight_pruned_simple_weight_matrix
):
    pruned_w = weight_pruning(simple_weight_matrix, k_weight_prune)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_equal(sess.run(pruned_w), weight_pruned_simple_weight_matrix)


def test_complex_weight_prune(complex_weight_matrix, k_weight_prune):
    pruned_w = weight_pruning(complex_weight_matrix, k_weight_prune)
    summed = tf.reduce_sum(pruned_w, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert np.count_nonzero(sess.run(summed)) == 7


def test_complex_unit_prune(complex_weight_matrix, k_unit_prune):
    pruned_w = unit_pruning(complex_weight_matrix, k_unit_prune)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert np.count_nonzero(sess.run(pruned_w)) == 66
