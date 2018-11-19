import math
from torch import Tensor

class Loss(object):
    def loss(self,input,target):
        raise NotImplementedError

    def backward(self,MLP):
        raise NotImplementedError


class MSE(Loss):
    """ Implementing the MSE loss function, along with it's backpropagation
    """
    def loss(self, input, target):
        """ Computes the l2 loss- discreptancy between the input and the target
        """
        self.input = input
        self.target = target
        delta = input - target
        n = delta.size()[1]
        return (delta*delta).sum()/n

    def backward(self, MLP):
        """ Calculates the gradient of the loss w.r.t the output of MLP and calls backpropagation on the MLP
        """
        delta = self.input - self.target
        n = delta.size()[1]
        return MLP.backward(2*delta/n)

class CrossEntropyLoss(Loss):
    """
    """
    def loss(self, input, target):
        self.input = input
        self.target = target
        y = 1/(1+(-self.input).exp())
        return -(target*(y.log()) + (1-target)*((1-y).log())).mean()


    def backward(self, MLP):
        y = 1/(1+(-self.input).exp())
        n = self.input.size()[1]
        grad = ((-self.input).exp()*self.target*y - (1-self.target)*((-self.input).exp()*(y**2))/(1-y))
        return MLP.backward(-grad/n)

class ClassificationLoss(Loss):
    """ Implements the loss which counts the number of misclassified points (labels are assumed to be 0,1, or one-hot-labels !). It does not feature backpropagation!
    """
    @staticmethod
    def loss(input, target):
        if(input.size(0) == 1): #Case non one-hot-labels
            x = (input > 1/2).float()
            return (target-x).nonzero().size()
        elif(input.size(0) == 2): #Case one-hot-labels
            x = input.max(dim=0)[1].float()
            y = target.max(dim=0)[1].float()
            return(y-x).nonzero().size()

    def backward(self, MLP):
        raise InvalidOperation('ClassificationLoss does not implement backpropagation. Only evaluates the accuracy of the model on the test set')
