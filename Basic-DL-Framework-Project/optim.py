import math
from torch import Tensor
from random import shuffle

class OptimizationAlgo(object):
    """ An abstract class that represents a parameter optimization method. Should contain everything that is useful for this method - all the parameters.
    Implements methods that optimize one parameter (optimize) and that train a model (train_model).
    The optimize method is to be overriden as it is, by definition, specific to the optimization algorithmself.
    The train_model method is provided as a basic algorithm for optimization. However, subclasses may override this method.
    """
    def optimize(self, w, grad):
        raise NotImplementedError

    #@profile
    def train_model(self, model, train, train_target, loss, n_epochs, **kwargs):
        """
        Assertions:
            assertion_input(model.modules[first linear layer],x) == True
            assertion_output(model.modules[last linear layer], train_target) == True
        Arguments:
            model - object of class MLP
            train - torch.Tensor(Float), the training set
            train_target - torch.Tensor(Float), the labels/expected results for each sample of the trainin set
            loss - object of class Loss, with respect to which the optimization algorithm trains the network (calculates costs and gradients)
            n_epochs - number of passes through the whole training set
            batch_size - the number of samples for one-time processing. By default, equal to '0', in which case, the whole input (train) is processed jointly

        Output:
            error - the loss computed at each epoch. Useful for visualising the training progress
        """
        errors = []

        n = train.size()[1]

        for e in range(0,n_epochs):
            errors.append(self.one_epoch(model, train, train_target, loss, **kwargs))
        return errors


class GradientDescent(OptimizationAlgo):
    """ Implements a basic gradient descent as an optimization algorithm.
    This computes the gradient with respect to ALL samples before performing any step.
    This should mean convergence should be more steady but there is also greater risk than in SGD
    to fall in a local minimum, which would result in bad generalization.
    Arguments:
        gamma - the step size, a real, positive number
    """
    def __init__(self, gamma):
        if(gamma <= 0):
            raise ValueError("You entered a gamma parameter = {}. Expected a positive value".format(gamma))
        self.gamma = gamma

    #@profile
    def one_epoch(self, model, train, train_target, loss, **kwargs):
        output = model.forward(train)
        error = loss.loss(output, train_target)
        loss.backward(model)
        model.updateWeights(self)
        return error

    def optimize(self, w, grad):
        """Perform one iteration of gradient descent on the parameter w, with gradient grad
        Arguments:
            w - torch.Tensor(Float)
            grad - torch.Tensor(Float), of dimension matching w.shape .
        """
        return w - self.gamma*grad

class SGD(OptimizationAlgo):
    """ Implements a basic batch stochastic gradient descent as an optimization algorithm.
    This is stochastic gradient descent for a mini_batch_size of 1, and a gradient descent for a mini_batch_size
    equal to the number of samples.
    Arguments:
        mini_batch_size - specify the batch size in the data

    """
    def __init__(self, mini_batch_size, gamma):
        #self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.GD = GradientDescent(gamma)

    def optimize(self, w, grad):
        #Here it should shuffle to get the minibatches
        #Now it sums over all the minibatches
        summinibatch = grad;
        #After having the minibatch

        return w - self.gamma*summinibatch;

    def one_epoch(self, model, train, train_target, loss, **kwargs):
        n = train.size()[1]
        output = model.forward(train)
        error = loss.loss(output, train_target) # calculate the loss on the whole batch at each epoch, to have a good estimate
        indices = list(range(0,n))
        shuffle(indices)
        indices = Tensor(indices).long()
        b_save = 0
        k = n
        for b in range(0,n - self.mini_batch_size,self.mini_batch_size):
            self.GD.one_epoch(model, train[:,indices.narrow(0,b,self.mini_batch_size)], train_target[:,indices.narrow(0,b,self.mini_batch_size)], loss, **kwargs)
            k = n - (b + self.mini_batch_size + 1)
            b_save = b + self.mini_batch_size
        self.GD.one_epoch(model, train[:,indices.narrow(0,b_save,k)], train_target[:,indices.narrow(0,b_save,k)], loss, **kwargs)
        # prevents overshooting the indices in the last mini_batch
        return error

class SGDWithRepetition(OptimizationAlgo):
    """This version of SGD selects samples with replacement after each gradient step.
    This imply that the notion of epochs is not really well defined. This has greater chance of overfitting
    as it might happen that certain samples are used much more than other (if the number of epochs is short)"""
    def __init__(self, mini_batch_size, gamma):
        #self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.GD = GradientDescent(gamma)

    def optimize(self, w, grad):
        return w - self.gamma*grad;

    #@profile
    def one_epoch(self, model, train, train_target, loss, **kwargs):
        n = train.size()[1]
        output = model.forward(train)
        error = loss.loss(output, train_target) # calculate the loss on the whole batch at each epoch
        indices = Tensor(n).random_(0,n).long()
        for b in range(0,n - self.mini_batch_size,self.mini_batch_size):
            self.GD.one_epoch(model, train[:,indices.narrow(0,b,self.mini_batch_size)], train_target[:,indices.narrow(0,b,self.mini_batch_size)], loss, **kwargs)
            k = n - (b + self.mini_batch_size + 1)
            b_save = b + self.mini_batch_size
        self.GD.one_epoch(model, train[:,indices.narrow(0,b_save,k)], train_target[:,indices.narrow(0,b_save,k)], loss, **kwargs)
        # prevents overshooting the indices in the last mini_batch
        return error
