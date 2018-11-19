import math
from torch import Tensor

class Module(object):
    """Base class for all modules, ie primary components of a neural network
    Each subclass needs to provide the forward, backward, updateWeights methods
    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def backward(self, gradSoFar):
        raise NotImplementedError

    def updateWeights(self, OptimAlgo):
        raise NotImplementedError


class Linear(Module):
    """ A Module representing a Linear node in a MLP
    Arguments:
        input_dim - the number of nodes that are incident to the layer
        output_dim - the number of nodes that the network outputs

    y = w*x+b, where w and b are tensors of dimensions (m1,m2) and (m1,1), while x is (m2,N) (2D TENSOR!!!), where N is the number of samples

    The parameters are initialized at random, using the ???? initialization
    """
    def __init__(self, input_dim, output_dim, bias = True):
        self.w = Tensor(output_dim, input_dim).normal_()*math.sqrt(2/input_dim)
        #print("At initalization weights W : {}".format(self.w))
        self.b = Tensor(output_dim,1).normal_()*math.sqrt(2/input_dim)

        self.gradW = Tensor(output_dim,input_dim).zero_()
        self.gradB = Tensor(output_dim,1).zero_()

        self.isBias = bias
        if(not bias):
            self.b.fill_(0)

    def forward(self,x):
        """ Implements a forward pass for the Linear node module
        Saves both the input and output for the backward pass, under self.input and self.output respectively
        Arguments:
            x - torch.Tensor(Float), of size (inpu_dim,N), where N is the batch size
        Outputs:
            x - torch.Tensor(Float), of size (output_dim,N), where N is the batch size
        """
        self.input = x #save the input
        x = self.w.mm(x)
        x = x + self.b
        self.output = x #save the output
        return x

    def backwardW(self, gradSoFar):
        """ Calcuates the gradient of the loss with respect to the parameter w of the layer
        and stores it in the parameter
        Arguments:
            gradSoFar -
        """
        return gradSoFar.mm(self.input.t()) # Column*Row vectors = matrix
        #return self.input.mm(gradSoFar)

    def backwardB(self, gradSoFar):
        """ Calculates the gradient of the loss with respect to the parameter b of the layer
        """
        #v = Tensor(self.input.size()[1],1).fill_(1)
        return gradSoFar.sum(1).view_as(self.b) # .mm(v)

    def backward(self, gradSoFar): #Row*matrix
        """ A method to compute the backward pass for the whole module.
        Delegates computations of gradients with respect to parameters to the corresponding functions
        """
        if(self.isBias):
            self.gradB = self.backwardB(gradSoFar)

        self.gradW = self.backwardW(gradSoFar)
        return gradSoFar.t().mm(self.w).t()

    def updateWeights(self, algo):
        """ Calls the optimizing algorithm to update the weights, given the gradients currently stored in the fields
        """
        self.w = algo.optimize(self.w, self.gradW)
        if(self.isBias):
            self.b = algo.optimize(self.b, self.gradB)

    def param(self):
        """ Returns a list of pairs (parameter, gradientWithRespectToParameter)
        """
        out = [(self.w,self.gradW)]
        if(self.isBias):
            out.append((self.b,self.gradB))
        return out

class Tanh(Module):
    """ A Module representing the tanh, a non-linear activation function. It has no parameters. Implements forward and backward functions
    """
    def forward(self,x):
        """ Calculates and returns tanh(input)
        """
        self.input = x
        x = (x.exp() - (-x).exp())/(x.exp()+(-x).exp())
        return x

    def backward(self, gradSoFar):
        """ Calculates the gradient of tanh, with respect to the input.
        """
        return gradSoFar*(self.input.clone().fill_(1)-self.forward(self.input)**2)

    def updateWeights(self, OptimAlgo):
        pass
        #do nothing, since no parameters

class LeakyReLU(Module):
    """ Implements the Leaky Rectified Linear Unit - a non-linear activation function.
    """

    def __init__(self,alpha):
        self.alpha = alpha

    #@profile
    def forward(self,x):
        """
        """
        self.greaterThan0 = (x>0).float()
        #self.input = x
        return x*(self.greaterThan0) + x*(1-self.greaterThan0).mul(self.alpha)

    def backward(self,gradSoFar):
        return (self.greaterThan0 + (1-self.greaterThan0)*self.alpha).mul(gradSoFar)

    def updateWeights(self, OptimAlgo):
        pass

class ReLU(LeakyReLU):
    """ Implement the Rectified Linear Unit - a non-linear activation function,
    as a particular case of the LeakyReLU
    """
    def __init__(self):
        super(ReLU,self).__init__(0)


class MLP(object):
    """ Implements a multilayer perceptron, a general model composed of several linear layers
    with non-linear activation function inbetween"""

    def __init__(self, moduleList = []):
        self.modules = moduleList

    #@profile
    def forward(self,x):
        # Just call, in order, the forward pass of each contained module
        self.input = x
        y = self.input
        for module in self.modules:
            y = module.forward(y)
        #y = y.t()
        return y

    def backward(self,grad):
        """ Backpropagation on the modules of the network
        """
        # For the backpropagation, do not forget to traverse the modules from the end to the beginning
        counter = 0
        for module in reversed(self.modules):
            #counter +=1
            #print('counter = {}, size of gradient = {}'.format(counter, grad.size()))
            grad = module.backward(grad)
        return grad

    def updateWeights(self, OptimAlgo):
        """ Uses algo to update the weights
        """
        for module in self.modules:
            module.updateWeights(OptimAlgo)

    def param(self):
        out = []
        for module in self.modules:
            out.extend(module.param())
        return out


class Sequential(MLP):
    """ The same as MLP, but with a different constructor - takes a number of parameters as modules.
    Comment: another class was introduced to have more freedom in that case - perhaps redefining some of the procedures, if we want to omit modules. Still not cear about the architecture
    """
    def __init__(self, *args):
        l = []
        for obj in args:
            if(isinstance(obj,Module)):
                l.append(obj)
            else:
                raise ArgumentError("only modules should be in the constructor's parameter list")
        super(Sequential, self).__init__(l)
