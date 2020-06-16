# perturbation-transportability

You can setup a virtual environment with all necessary packages by running
```
bash setup.sh
```

This will also set up the ipython kernel `drug-prediction`, which you should use when running the Jupyter notebook.

The data files are too large to be kept in the repository, but can be downloaded by running `download.sh`.

The files are organized as follows:
* `src` contains algorithms
* `processing` contains managers for the data
* `evaluation` contains classes for evaluating the performance of various algorithms
* `exploration` contains data analysis that is more related to describing the data than describing the results
* `scratch` contains temporary files for one-off tasks, such as checking that new code works
