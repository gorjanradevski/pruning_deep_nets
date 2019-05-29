import numpy as np


class Evaluator:
    def __init__(self):
        self.loss = 0.0
        self.accuracies = []
        self.num_batches = 0
        self.best_accuracy = -1.0

    def update_metrics(
        self, loss: float, labels: np.ndarray, predictions: np.ndarray
    ) -> None:
        """Updates the current metrics in the following way:

        1. Sums the loss with the total loss for the current epoch.
        2. Compute the accuracy for the current batch.
        3. Increases the counter for the number of batches that have passed so far.

        Args:
            loss: The loss for the current batch.
            labels: The labels for the current batch.
            predictions: The predictions for the current batch.

        Returns:
            None

        """
        self.loss += loss
        accuracy = np.sum(
            np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)
        ) / len(labels)
        self.accuracies.append(accuracy)
        self.num_batches += 1

    def final_accuracy(self) -> float:
        """Return the total accuracy of the data passed through the evaluator.

        Returns:
            The accuracy.

        """
        return sum(self.accuracies) / self.num_batches

    def reset_all(self) -> None:
        """Resets all attributes. Usually it is done after each training epoch.

        Returns:
            None

        """
        self.loss = 0.0
        self.accuracies = []
        self.num_batches = 0

    def is_best_accuracy(self) -> bool:
        """Because every evaluator object stores the best accuracy computed so far, this
        method checks whether the current computed accuracy is higher that the best
        accuracy. Usually it is done at the end of a training epoch.

        Returns:
            True if the current accuracy is the best so far, else otherwise.

        """
        if self.final_accuracy() > self.best_accuracy:
            return True
        return False

    def update_best_accuracy(self) -> None:
        """Updates the best accuracy attribute.

        Returns:
            None

        """
        self.best_accuracy = self.final_accuracy()
