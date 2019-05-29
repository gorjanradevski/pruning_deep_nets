import tensorflow as tf
import argparse
from tqdm import tqdm
import logging
import os

from basic_utils.constants import train_size
from train_inference_utils.loaders import TrainValLoader
from train_inference_utils.models import BasicModel
from train_inference_utils.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def train(
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    save_model_path: str,
) -> None:
    """Performs training of the model. Additionally, it saves the model only when the
    accuracy on the validation set is the highest so far.

    Args:
        batch_size: The size of the batch to use.
        epochs: The number of epochs to train the model.
        save_model_path: Where to save the model.
        weight_decay: The amount of weight decay to use when training.
        learning_rate: The learning rate to use when training the model.

    Returns:
        None

    """
    # Prepare the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    x_val, y_val = x_train[train_size:], y_train[train_size:]
    x_train, y_train = x_train[:train_size], y_train[:train_size]
    logger.info("Data prepared...")
    # Reset the default graph and set the random seed
    tf.reset_default_graph()
    # Fixing the random seed
    tf.set_random_seed(42)

    # Create the data loader
    loader = TrainValLoader(x_train, y_train, x_val, y_val, batch_size)
    features, labels = loader.get_next()
    logger.info("Loader created...")

    # Create the model
    model = BasicModel(features, labels)
    logger.info("Model created...")

    # Create evaluator
    val_evaluator = Evaluator()

    with tf.Session() as sess:
        # Intialize the graph and set the best loss
        model.init(sess)

        for e in range(epochs):
            val_evaluator.reset_all()
            # Initialize iterator with training data
            sess.run(loader.train_init)
            try:
                with tqdm(total=len(y_train)) as pbar:
                    while True:
                        _, loss, labels = sess.run(
                            [model.opt, model.loss, model.labels],
                            feed_dict={
                                model.weight_decay: weight_decay,
                                model.learning_rate: learning_rate,
                            },
                        )
                        pbar.update(len(labels))
                        pbar.set_postfix({"Batch loss": loss})
            except tf.errors.OutOfRangeError:
                pass

            # Initialize iterator with validation data
            sess.run(loader.val_init)
            try:
                with tqdm(total=len(y_val)) as pbar:
                    while True:
                        loss, labels, predictions = sess.run(
                            [model.loss, model.labels, model.predictions]
                        )
                        pbar.update(len(labels))
                        val_evaluator.update_metrics(loss, labels, predictions)
            except tf.errors.OutOfRangeError:
                pass

            if val_evaluator.is_best_accuracy():
                val_evaluator.update_best_accuracy()
                logger.info("=============================")
                logger.info(
                    f"Found new best on epoch {e + 1} with accuracy:"
                    f" {val_evaluator.best_accuracy} | Saving model..."
                )
                logger.info("=============================")
                model.save_model(sess, save_model_path)


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.weight_decay,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.save_model_path,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a model on the MNIST dataset.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="For how many epochs to train the model.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/best_model",
        help="Where to save the best model.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The amount of weight decay"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="The amount of weight decay",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
