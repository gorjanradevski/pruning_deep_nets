import tensorflow as tf
from abc import ABC

from basic_utils.constants import width, height, num_labels, num_prefetch


class BaseLoader(ABC):
    def __init__(self, batch_size: int):
        """A based loader from which all loaders should inherit."""
        self.batch_size = batch_size

    @staticmethod
    def parse_data(image_matrix: tf.Tensor, label: tf.Tensor):
        """Parses the the data.

        - Unrolls the image matrix to be [28 * 28] instead of [28, 28].
        - Converts the image label to one hot.

        Args:
            image_matrix: The matrix representing the image.
            label: The label of the image.

        Returns:
            The unrolled image and the one-hotted label.

        """
        return (
            tf.cast(tf.reshape(image_matrix, [width * height]), tf.float32),
            tf.one_hot(label, depth=num_labels),
        )


class TrainValLoader(BaseLoader):
    """A data loader used for training."""

    def __init__(self, x_train, y_train, x_val, y_val, batch_size: int):
        super().__init__(batch_size)
        # Create the training dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # Shuffle after each epoch
        self.train_dataset = self.train_dataset.shuffle(buffer_size=x_train.shape[0])
        # Prepare the data in the right format
        self.train_dataset = self.train_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Batch the data
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        # Prefetch the batches when training on GPU
        self.train_dataset = self.train_dataset.prefetch(num_prefetch)

        # Create the validation dataset
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        # Prepare the data in the right format

        self.val_dataset = self.val_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # Batch the data
        self.val_dataset = self.val_dataset.batch(batch_size)
        # Prefetch the batches when training on GPU
        self.val_dataset = self.val_dataset.prefetch(num_prefetch)

        # Create the iterator
        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes
        )
        # Create the training and validation initializers
        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init = self.iterator.make_initializer(self.val_dataset)

    def get_next(self):
        """Utility wrapper around the get next method for readability.

        Returns:
            The images and labels.

        """
        images, labels = self.iterator.get_next()

        return images, labels


class TestLoader(BaseLoader):
    """A data loader used for inference performed on the test set."""

    def __init__(self, x_test, y_test, batch_size):
        super().__init__(batch_size)
        # Create the validation dataset
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # Prepare the data in the right format

        self.test_dataset = self.test_dataset.map(
            self.parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # Batch the data
        self.test_dataset = self.test_dataset.batch(self.batch_size)
        # Prefetch the batches when training on GPU
        self.test_dataset = self.test_dataset.prefetch(num_prefetch)
        self.iterator = self.test_dataset.make_one_shot_iterator()

    def get_next(self):
        """Utility wrapper around the get next method for readability.

        Returns:
            The images and labels.

        """
        images, labels = self.iterator.get_next()

        return images, labels
