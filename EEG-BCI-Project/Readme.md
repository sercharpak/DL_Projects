# Mini project 1 - EEG Recordings - Laterality of Finger Movement

BCI competition [website](http://bbci.de/competition/ii/).
The literature is detailed in the report pdf.


## Prerequisites:

The code for this project is written exclusively in Python (3.5.2). It makes use of the standard math and Numpy libraries. PyTorch 0.3.0 is used extensively for deep learning.

## Executing the code:
Once the archive is unzipped, you can use `python3 test.py` to run the training of all the models described in the report. By default, the code is run on the 1000Hz-sample, without cross validation. Please look into the next section to see how to change this.

## Description of the files:

### models.py
Contains all the models mentioned in the report, namely `NHLP`, `SimpleConv`, `EEGNet_2018`, `ShallowConvNet`. There are also a few additional ones, that were developed, but discontinued.

### test.py
The main executable, can be run by entering `python3 test.py` in the terminal. Additionally, two optional boolean arguments are accepted:
* `one_khz` : setting to 1 or (anything else) corresponds to setting the `one_khz` variable to `True` or `False` respectively, what results in loading, and training on the 100 or 100 Hz datasets.
*  `cross_val`: setting to 1 corresponds to doing the cross validation, followed a training

Beware that if only none or only one of the two arguments is provided, the default options are chosen, namely: `one_khz = True`, `cross_val = False`.

If the cross validation option is chosen, the
### cross_validation.py
Contains the routine and it's helpers for cross validation and training of the implemented models:
* `split_data`: splits a dataset in several subsets of equal size
* `build_train_and_validation`: create a train set and a validation set from a list of subsets
* `train_model`: train the model using the dataset and criteria (loss, optimizer, scheduler) specified in the parameters
* `evaluate_error`: evaluates the classification error of a model on a sample. Specifically, return the fraction of missclassified data
* `cross_validate`: performs cross validation. Can also output the evolution of the train and validation error. Beware, the last feature serves only as an indication and cannot be used to evaluate model's performance.

### data_augmentation.py
Contains methods for data augmentation.
* `augment_data`: returns noise ( a time series) to be added to the input to obtain a noisy copy
* `augment_train`: deals with augmenting the input while preserving the labels

### plot_CV_results.py
Plots the results from cross_validation from the output of `test.py` ran with the cross validation option. It is to be run `python3 plot_CV_results.py fname n_time_points`, where `fname` is the name of the output file and `n_time_points` is the number of time-points in the dataset for which the file `fname` was produced. The default parameters allow one to run the script on the output of the unmodified `test.py`.
