import tensorflow as tf

from basic_utils.constants import num_neurons, num_labels, learning_rate


class BasicModel:
    def __init__(self, features: tf.Tensor, labels: tf.Tensor, training: bool = True):
        self.features = features
        self.labels = labels

        w0 = tf.get_variable(
            shape=[self.features.get_shape()[1], num_neurons[0]],
            initializer=tf.glorot_uniform_initializer(),
            name="w0",
        )
        out0 = tf.nn.relu(tf.matmul(features, w0))
        w1 = tf.get_variable(
            shape=[num_neurons[0], num_neurons[1]],
            initializer=tf.glorot_uniform_initializer(),
            name="w1",
        )
        out1 = tf.nn.relu(tf.matmul(out0, w1))
        w2 = tf.get_variable(
            shape=[num_neurons[1], num_neurons[2]],
            initializer=tf.glorot_uniform_initializer(),
            name="w2",
        )
        out2 = tf.nn.relu(tf.matmul(out1, w2))
        w3 = tf.get_variable(
            shape=[num_neurons[2], num_neurons[3]],
            initializer=tf.glorot_uniform_initializer(),
            name="w3",
        )
        out3 = tf.nn.relu(tf.matmul(out2, w3))
        w_project = tf.get_variable(
            shape=[num_neurons[3], num_labels],
            initializer=tf.glorot_uniform_initializer(),
            name="w_project",
        )
        logits = tf.matmul(out3, w_project)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                self.labels, logits, axis=None, name=None, dim=None
            )
        )
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.predictions = tf.nn.softmax(logits)
        self.saver_loader = tf.train.Saver()

    @staticmethod
    def weight_pruning(w: tf.Variable, k: float) -> tf.Tensor:
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
            tf.round(tf.cast(tf.shape(w)[1], tf.float32) * tf.constant(k)),
            dtype=tf.int32,
        )
        norm = tf.norm(w, axis=0)
        row_indices = tf.tile(tf.range(tf.shape(w)[0]), [k])
        _, col_indices = tf.nn.top_k(tf.negative(norm), k, sorted=True, name=None)
        col_indices = tf.reshape(
            tf.tile(tf.reshape(col_indices, [-1, 1]), [1, tf.shape(w)[0]]), [-1]
        )
        indices = tf.stack([row_indices, col_indices], axis=1)

        return tf.scatter_nd_update(
            w, indices, tf.zeros(tf.shape(w)[0] * k, tf.float32)
        )

    @staticmethod
    def unit_pruning(w: tf.Tensor, k: float) -> tf.Tensor:
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
        _, indices = tf.nn.top_k(
            tf.negative(tf.abs(w_reshaped)), k, sorted=True, name=None
        )
        mask = tf.scatter_nd_update(
            tf.Variable(tf.ones_like(w_reshaped, dtype=tf.float32)),
            tf.reshape(indices, [-1, 1]),
            tf.zeros([k], tf.float32),
        )

        return tf.reshape(w_reshaped * mask, tf.shape(w))

    def init(self, sess: tf.Session, checkpoint_path: str = None) -> None:
        """Initializes all variables in the graph. Additionally, if a checkpoint is
        provided it will initialize all variables in the graph from the checkpoint.

        Args:
            sess: The active session.
            checkpoint_path: A path to a valid model checkpoint.

        Returns:
            None

        """
        sess.run(tf.global_variables_initializer())
        if checkpoint_path:
            self.saver_loader.restore(sess, checkpoint_path)

    def save_model(self, sess: tf.Session, save_path: str) -> None:
        """Dumps the model definition.

        Args:
            sess: The active session.
            save_path: Where to save the model.

        Returns:
            None
        """
        self.saver_loader.save(sess, save_path)
