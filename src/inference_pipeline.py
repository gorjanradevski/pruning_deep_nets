import tensorflow as tf
import logging
import os
from tqdm import tqdm
import argparse


from train_inference_utils.loaders import TestLoader
from train_inference_utils.evaluator import Evaluator
from train_inference_utils.models import BasicModel
from basic_utils.constants import prune_k, prune_types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def inference(batch_size: int, load_model_path: str):
    """Performs inference on the test set of the MNIST dataset while going through both
    pruning schemes with different pruning coefficient.

    Args:
        batch_size: The size of the batch to use.
        load_model_path: From where to load the model

    Returns:

    """
    # Prepare the data
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    logger.info("Data prepared...")

    for prune_type in prune_types:
        for k in prune_k:
            # Reset the default graph and set the random seed
            tf.reset_default_graph()
            # Fixing the random seed
            tf.set_random_seed(42)

            # Create the data loader
            loader = TestLoader(x_test, y_test, batch_size)
            features, labels = loader.get_next()
            logger.info("Loader created...")

            # Create the model
            model = BasicModel(features, labels, prune_type, k)
            logger.info("Model created...")

            # Create evaluator
            test_evaluator = Evaluator()

            with tf.Session() as sess:
                # Intialize the graph and set the best loss
                model.init(sess, load_model_path)

                try:
                    with tqdm(total=len(y_test)) as pbar:
                        while True:
                            loss, labels, predictions = sess.run(
                                [model.loss, model.labels, model.predictions]
                            )
                            pbar.update(len(labels))
                            test_evaluator.update_metrics(loss, labels, predictions)
                except tf.errors.OutOfRangeError:
                    pass

                test_evaluator.update_best_accuracy()

                logger.info(
                    f"The accuracy for {prune_type} with {k} removal on the MNIST "
                    f"dataset is: {test_evaluator.best_accuracy}"
                )

            if prune_type is None:
                break


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(args.batch_size, args.load_model_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference on the MNIST test" "set."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="models/best_model",
        help="From where to load the model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
