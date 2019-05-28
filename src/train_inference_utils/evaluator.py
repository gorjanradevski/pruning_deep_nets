import numpy as np


class Evaluator:
    def __init__(self):
        self.loss = 0.0
        self.accuracies = []
        self.num_batches = 0
        self.best_accuracy = -1.0

    def update_metrics(self, loss, labels, predictions):
        self.loss += loss
        accuracy = np.sum(
            np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)
        ) / len(labels)
        self.accuracies.append(accuracy)
        self.num_batches += 1

    def final_accuracy(self):
        return sum(self.accuracies) / self.num_batches

    def reset_all(self):
        self.loss = 0.0
        self.accuracies = []
        self.num_batches = 0

    def is_best_accuracy(self):
        if self.final_accuracy() > self.best_accuracy:
            return True
        return False

    def update_best_accuracy(self):
        self.best_accuracy = self.final_accuracy()
