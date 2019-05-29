import tensorflow as tf

from basic_utils.constants import num_neurons, num_labels, learning_rate
from train_inference_utils.prunings import pruning_factory


class BasicModel:
    def __init__(
        self,
        features: tf.Tensor,
        labels: tf.Tensor,
        pruning_type: str = None,
        k: float = 0,
    ):
        self.features = features
        self.labels = labels

        w0 = pruning_factory(
            pruning_type,
            tf.get_variable(
                shape=[self.features.get_shape()[1], num_neurons[0]],
                initializer=tf.glorot_uniform_initializer(),
                name="w0",
            ),
            k,
        )
        out0 = tf.nn.relu(tf.matmul(features, w0))
        w1 = pruning_factory(
            pruning_type,
            tf.get_variable(
                shape=[num_neurons[0], num_neurons[1]],
                initializer=tf.glorot_uniform_initializer(),
                name="w1",
            ),
            k,
        )
        out1 = tf.nn.relu(tf.matmul(out0, w1))
        w2 = pruning_factory(
            pruning_type,
            tf.get_variable(
                shape=[num_neurons[1], num_neurons[2]],
                initializer=tf.glorot_uniform_initializer(),
                name="w2",
            ),
            k,
        )
        out2 = tf.nn.relu(tf.matmul(out1, w2))
        w3 = pruning_factory(
            pruning_type,
            tf.get_variable(
                shape=[num_neurons[2], num_neurons[3]],
                initializer=tf.glorot_uniform_initializer(),
                name="w3",
            ),
            k,
        )
        out3 = tf.nn.relu(tf.matmul(out2, w3))
        w_project = tf.get_variable(
            shape=[num_neurons[3], num_labels],
            initializer=tf.glorot_uniform_initializer(),
            name="w_project",
        )
        logits = tf.matmul(out3, w_project)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(self.labels, logits)
        )
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.predictions = tf.nn.softmax(logits)
        self.saver_loader = tf.train.Saver()

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
