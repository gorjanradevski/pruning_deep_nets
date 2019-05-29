import tensorflow as tf


def unit_pruning(w: tf.Variable, k: float) -> tf.Variable:
    """Performs pruning on a weight matrix w in the following way:

    - The absolute value of all elements in the weight matrix are computed.
    - The indices of the smallest k% elements based on their absolute values are
    selected.
    - All elements with the matching indices are set to 0.

    Args:
        w: The weight matrix.
        k: The percentage of values (units) that should be pruned from the matrix.

    Returns:
        The unit pruned weight matrix.

    """
    k = tf.cast(
        tf.round(tf.size(w, out_type=tf.float32) * tf.constant(k)), dtype=tf.int32
    )
    w_reshaped = tf.reshape(w, [-1])
    _, indices = tf.nn.top_k(tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None)
    mask = tf.scatter_nd_update(
        tf.Variable(
            tf.ones_like(w_reshaped, dtype=tf.float32), name="mask", trainable=False
        ),
        tf.reshape(indices, [-1, 1]),
        tf.zeros([k], tf.float32),
    )

    return w.assign(tf.reshape(w_reshaped * mask, tf.shape(w)))


def weight_pruning(w: tf.Variable, k: float) -> tf.Variable:
    """Performs pruning on a weight matrix w in the following way:

    - The euclidean norm of each column is computed.
    - The indices of smallest k% columns based on their euclidean norms are
    selected.
    - All elements in the columns that have the matching indices are set to 0.

    Args:
        w: The weight matrix.
        k: The percentage of columns that should be pruned from the matrix.

    Returns:
        The weight pruned weight matrix.

    """
    k = tf.cast(
        tf.round(tf.cast(tf.shape(w)[1], tf.float32) * tf.constant(k)), dtype=tf.int32
    )
    norm = tf.norm(w, axis=0)
    row_indices = tf.tile(tf.range(tf.shape(w)[0]), [k])
    _, col_indices = tf.nn.top_k(tf.negative(norm), k, sorted=True, name=None)
    col_indices = tf.reshape(
        tf.tile(tf.reshape(col_indices, [-1, 1]), [1, tf.shape(w)[0]]), [-1]
    )
    indices = tf.stack([row_indices, col_indices], axis=1)

    return w.assign(
        tf.scatter_nd_update(w, indices, tf.zeros(tf.shape(w)[0] * k, tf.float32))
    )


def pruning_factory(pruning_type: str, w: tf.Variable, k: float) -> tf.Variable:
    if pruning_type is None:
        return w
    elif pruning_type == "weight_pruning":
        return weight_pruning(w, k)
    elif pruning_type == "unit_pruning":
        return unit_pruning(w, k)
    else:
        raise ValueError(f"Pruning type {pruning_type} unrecognized!")
