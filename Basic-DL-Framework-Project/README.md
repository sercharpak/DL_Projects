# Mini project 2 - Designing a mini deep learning framework using pytorch's tensor operations

All the implementation and the mathematical background is described in the joint report. Please see the code for more details.

## Prerequisites
The implemented framework heavily relies on the `Tensor` type from the PyTorch 0.3.0 package. It was run on Python 3.5.2. See the respective manual online for installation guides. Running `test.py` does not require more.

## Running the code
Unzip the archive. After fulfilling the requirements listed above, the executable required by the project description can be run from the command line `python3 test.py`

## Use of the framework
For sample use, please see `test.py` and the report. The implementation choices described and justified in the report impact the API, which is exemplified in the file mentioned above.

## Description of the files

### test.py
The required script. Generates the toy dataset, initializes and trains the suggested model. Prints the loss and the number of misclassified points after training (in the terminal). Generates an output file, `train_loss.out`, logging the loss.

### module.py
Contains an implementation of the modules - basic elements of a neural network, here implemented as `MLP` and `Sequential`. The implemented elements, subclasses of the abstract class `Module` are:
* `Linear`: a linear layer
* `Tanh`: implementation of the element-wise Tanh activation function
* `LeakyReLU`: the LeakyReLU activation function (see report for references)
* `ReLU`: Rectified Linear Unit, implemented as a special case of `LeakyReLU`

### optim.py
Contains the implementation of the optimization algorithms. It contains the basic gradient descent and two versions of the stochastic gradient descent:
* `GD`: standard gradient descent algorithm, with full batch forward and backward passes
* `SGD`: the standard stochastic gradient descent algorithm. Shuffles the samples and performs the passes on mini batches specified by the user at initialization

### loss.py
Contains the implementation of the loss functions listed below. The first two feature backpropagation, while the third one doesn't (see report or below for justification):
* `MSE`: standard mean square error - see report for the formula.
* `CrossEntropyLoss`: the cross entropy loss (see report and PyTorch documentation for reference) implemented in the case of 2 classes (binary classification)
* `ClassificationLoss`: implements a loss evaluating the number of misclassified samples. More specifically, it gives the fraction of wrongly classified points. Beware, it does not implement the backward pass (raises `InvalidOperation`), since the loss function is not differentiable with respect to the variables.

## utils.py
Provides ways to functions necessary for the creation of the required dataset and conversion to the right label format, namely one-hot-labels.
