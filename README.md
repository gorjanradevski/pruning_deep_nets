# FOR.ai challenge

### Reproducing the virtualenv:

- In case you have [Poetry](https://github.com/sdispater/poetry) installed,
`poetry install` inside the main project directory is sufficient for getting
all the dependencies straight.
- A `requirements.txt` file is included as well so the other tools can be used
to reproduce the virtualenv as well.


### Running the pipelines

- `training_pipeline.py` trains a model and if invoked from the main project
directory, the model will be saved in `models/bestmodel`.

- `inference_pipeline.py` performs inference given a trained model. It goes through
the two different pruning methods and through all different `k` pruning percentages.

### Notebooks

- In the `notebooks/` directory there is a notebook `MNIST_pruning_results.ipynb`
that does a walk-through of the results.


### Running the tests
- In `src/train_inference_utils/test_prunings.py` there are several tests for the
pruning strategies implemented. To run the tests run: `pytest src/`