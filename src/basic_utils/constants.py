# Number of images used for training (rest goes for validation)
train_size = 55000

# Image width and height of mnist
width = 28
height = 28

# The total number of labels
num_labels = 10

# The number of batches to prefetch when training on GPU
num_prefetch = 5

# Number of neurons the weight matrices as specified in the document about the challenge
num_neurons = [1000, 1000, 500, 200]

# The learning rate used to train the model
learning_rate = 0.0001

# Testing with prunning for
prune_k = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]

# Prune types
prune_types = ["unit_pruning", "weight_pruning"]
