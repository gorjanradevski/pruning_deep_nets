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
