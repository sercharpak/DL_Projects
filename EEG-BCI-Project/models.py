import torch
from torch import nn
import math

class NHLP(nn.Module):
    """NO HIDDEN LAYER PERCEPTRON : Implement a linear perceptron with no hidden layer"""
    def __init__(self, in_features, n_classes=2):
        """Arguments:
            in_features - the number of features of the input
            n_classes - the number of classes in the classification task
        """
        super(NHLP, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.lin1 = nn.Linear(self.in_features, self.n_classes)
    def forward(self, x):
        x = x.view(-1, self.in_features) # reshape to match the 2-dimensional format, for each sample
        x = self.lin1(x)
        return x
    def reset(self):
        self.lin1.reset_parameters()
    def name(self):
        return "Linear baseline"

class Avg_NHLP(nn.Module):
    """NO HIDDEN LAYER PERCEPTRON : Implement a linear perceptron with no hidden layer"""
    def __init__(self, n_channels,n_time_points, n_classes=2):
        super(Avg_NHLP, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size = (n_channels,1))
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.n_classes = n_classes
        self.lin1 = nn.Linear(self.n_time_points, self.n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(-1, self.n_time_points)
        x = self.lin1(x)
        return x

    def reset(self):
        self.lin1.reset_parameters()
    def name(self):
        return "Linear baseline with average"

class ShallowLinear(nn.Module):
    """2-perceptron neural net: Implements a net: linear(in_features, size_hidden) -> ReLU() -> linear(size_hidden, n_classes)"""
    def __init__(self, in_features, size_hidden, n_classes=2):
        """Arguments:
            in_features - the number of features of the input
            size_hidden - number of features in the unique hidden layer
            n_classes - the number of classes in the classification task
        """
        super(ShallowLinear, self).__init__()
        self.in_features = in_features
        self.size_hidden = size_hidden
        self.n_classes = n_classes
        self.lin1 = nn.Linear(self.in_features, self.size_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(size_hidden, self.n_classes)
    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
    def reset(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def name(self):
        return '2-perceptron neural net'

class SimpleConv(nn.Module):
    """SIMPLE CONVOLUTIONAL : 1D time convolution with a single filter, Max pooling then a Linear layer for classification"""
    def __init__(self, n_channels, n_points, kernel_size, pool_stride, n_classes=2):
        """Arguments:
            n_channels - the number of channels in the input signal
            kernel_size - the size of the kernel in the first convolution layer
            pool-stride - the pool-stride for the max pooling layer. The kernel size is taken to be equal to the kernel size
            n_classes - the number of classes in the classification task
        """
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv1d(n_channels,1,kernel_size)
        self.maxpool = nn.MaxPool1d(pool_stride,pool_stride)
        self.lin = nn.Linear((n_points-kernel_size+1)//pool_stride, n_classes) # explicit calculation of the number of features in the input, given the known structure of the network
        self.net = self.build_net()
    def build_net(self):
        return nn.Sequential( #[1, n_channels, n_points]
        self.conv, #[1,1,n_points-kernel_size+1]
        self.maxpool,#[1,1,(n_point-kernel_size+1)/pool_stride]
        self.lin #[1,n_classes]
        )
    def forward(self,x):
        x = self.net.forward(x)
        x = x.squeeze()
        return x
    def reset(self):
        self.conv.reset_parameters()
        self.lin.reset_parameters()
        self.net = self.build_net()
    def name(self):
        return "Convolutional baseline"

class SimpleConv_FullyConv(nn.Module):
    """SIMPLE FULLY CONVOLUTIONAL : 1D time convolution with a single filter, Max pooling then a final 1D convolution to classify"""
    def __init__(self, n_channels, n_points, kernel_size, pool_stride, n_classes=2):
        """Arguments:
            n_channels - the number of channels in the input signal
            kernel_size - the size of the kernel in the first convolution layer
            pool-stride - the pool-stride for the max pooling layer. The kernel size is taken to be equal to the kernel size
            n_classes - the number of classes in the classification task
        """
        super(SimpleConv_FullyConv, self).__init__()
        self.conv = nn.Conv1d(n_channels,1,kernel_size)
        self.maxpool = nn.MaxPool1d(pool_stride,pool_stride)
        self.conv_class = nn.Conv1d(1,n_classes,(n_points-kernel_size+1)//pool_stride)
        self.net = self.build_net()

    def forward(self,x):
        x = self.net.forward(x)
        x = x.squeeze()
        return x
    def reset(self):
        self.conv.reset_parameters()
        self.conv_class.reset_parameters()
        self.net = self.build_net()
    def build_net(self):
        return nn.Sequential( #[1, 28,n_points]
        self.conv, #[1,1,n_points-kernel_size+1]
        self.maxpool,#[1,1,(n_points-kernel_size+1)//pool_stride]
        self.conv_class
        )
    def name(self):
        return "Fully convolutional"


class FullyConv_Embedding(nn.Module):
    """Fully convolutional model for 100Hz sampling, with an embedding for the 1000Hz sampling rate, with 28 channels
    For a 100 Hz sample, it corresponds to SimpleConv_FullyConv(28, 50, 6, 2). It is reimplemented here so as to avoid implementing training_inner() for SimpleConv_FullyConv, what does not make sense
    A 1000 Hz sample is first projected into the space of 100 Hz samples, through a series of initial layers: convBig, relu, convBig2. Then, it is treated as a 100 Hz sample.
    Allows to train using both 100 and 1000 Hz recordings to train the same model.
    Example of a training procedure:
        cv.train_model(model_fully_conv, criterion, optimizer, train_input, train_target, n_epochs = n_epochs, verbose = 2) # Train the "classifier" first
        model_fully_conv.training_inner(False) # Disable training on the classifier part
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fully_conv.parameters()),lr = 0.001) # change the learning rate and the parameters to optimize
        cv.train_model(model_fully_conv, criterion, optimizer, train_input_large, train_target_large, n_epochs = n_epochs, verbose = 2) #
    """

    def __init__(self,n_channels, n_classes=2):
        super(FullyConv_Embedding, self).__init__()
        self.channel = n_channels
        self.convBig = nn.Conv1d(n_channels,2*n_channels, kernel_size = 10, stride = 10, groups = n_channels)
        self.relu = nn.Softsign()
        self.convBig2 = nn.Conv2d(2, 1 , (1,1) , stride = (1,1))
        self.conv = nn.Conv1d(n_channels,1, 6)
        self.maxpool = nn.MaxPool1d(3, 3)
        self.conv2 = nn.Conv1d(1, 1, 4, dilation = 1)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        self.conv_class = nn.Conv1d(1, n_classes, 6)
        self.net = self.build_net()

    def forward(self,x):
        if(x.shape[2]==500):
            self.n_points = x.size()[0] #[1,28,500]
            x = self.convBig(x) #[1,2*28,50]
            x = x.view(self.n_points, 2, self.channel, x.size()[2]) #[1,2,28,50]
            x = self.relu(x) # [1,2,28,50]
            x = self.convBig2(x) #[1,1,28,50]
            x.squeeze_() #[1,28,50]
            x = self.net.forward(x) #[1,2,1] (see build_net())
            x.squeeze_()
        else:
            x = self.net.forward(x)
            x.squeeze_()
        return x

    def training_inner(self, b = False):
        """ Allows switching training of the classification part on & off, when we want to train on 1000 Hz
        """
        for p in self.conv.parameters():
            p.requires_grad = b
        for p in self.conv2.parameters():
            p.requires_grad = b
        for p in self.conv_class.parameters():
            p.requires_grad = b


    def reset(self):
        self.convBig.reset_parameters()
        self.convBig2.reset_parameters()
        self.conv.reset_parameters()
        self.conv2.reset_parameters()
        self.conv_class.reset_parameters()
        self.net = self.build_net()
    def build_net(self):
        return nn.Sequential( #[1, 28, 50]
        self.conv, #[1,1,45]
        self.maxpool, #[1,1,15]
        self.conv2, #[1,1,12]
        self.maxpool2, #[1,1,6]
        self.conv_class #[1,2,1]
        )
    def name(self):
        return "Fully convolutional for 100Hz, with a depth-wise Embedding for 1000 Hz"

class EEGNet_2018(nn.Module):
    """EEGNet : Compact CNN Architecture in its latest March 2018 version.
    Input is: torch.Size([#Samples, #Channels, #TimePoints]), ex: torch.Size([316, 28, 500]) or torch.Size([316, 28, 50])
    Described in :
    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and
    B. J. Lance. EEGNet: A Compact Convolutional Network for EEG-based
    Brain-Computer Interfaces. CoRR, abs/1611.08024, 2016
    https://arxiv.org/abs/1611.08024"""
    def __init__(self, n_channels, n_time_points, n_filters=8, n_classes=2):
        super(EEGNet_2018, self).__init__()
        #For the last layer, the classification layer lenght
        #Depends on the time_points. The dependence is specified to be 4X4XT/16 in the paper but
        #had to twich it to work.
        if (n_time_points <= 64):
            self.length = (((n_time_points//16)*2*n_filters)) #As in the paper
            self.kernelL1Conv_1 = 50
            self.kernelL2Conv_1 = 8
            self.kernelL3Conv_1 = 8
        else:#(n_time_points > 64)
            self.length = (((n_time_points//16)*2*n_filters)) #As in the paper
            self.kernelL1Conv_1 = 400
            self.kernelL2Conv_1 = 20
            self.kernelL3Conv_1 = 20
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        #EEGNet - Layer - 1
        filters_layer_1 = n_filters
        self.l1Conv2d = nn.Conv2d(1,filters_layer_1,(1,self.kernelL1Conv_1),padding = 0)
        self.batch1 = nn.BatchNorm2d(filters_layer_1,False)
        self.padd1 = nn.ZeroPad2d((self.kernelL1Conv_1//2,(self.kernelL1Conv_1//2)-1))
        self.l1DeptConv2d = nn.Conv2d(filters_layer_1,filters_layer_1,(self.n_channels,1),padding = 0, groups=filters_layer_1)#Depthwise
        self.batch1_2 = nn.BatchNorm2d(filters_layer_1,False)
        #EEGNet - Layer - 2
        filters_layer_2 = n_filters
        self.conv2 = nn.Conv2d(filters_layer_1,filters_layer_2,(1,self.kernelL2Conv_1),groups = filters_layer_2)#Depthwise
        self.conv2_2 = nn.Conv2d(filters_layer_2,filters_layer_2,kernel_size=1)#Pointwise
        self.batch2 = nn.BatchNorm2d(filters_layer_2,False)
        self.padd2 = nn.ZeroPad2d((self.kernelL2Conv_1//2,(self.kernelL2Conv_1//2)-1))
        self.pool2 = nn.AvgPool2d(1,4)#(self.kernelL2Conv_1//2))
        #EEGNet - Layer - 3
        filters_layer_3 = 2*n_filters
        self.conv3 = nn.Conv2d(filters_layer_2,filters_layer_3, (1,self.kernelL3Conv_1), groups = filters_layer_2)#Depthwise
        self.conv3_2 = nn.Conv2d(filters_layer_3,filters_layer_3,kernel_size=1)#Pointwise
        self.batch3 = nn.BatchNorm2d(filters_layer_3,False)
        self.padd3 = nn.ZeroPad2d((self.kernelL3Conv_1//2-1,(self.kernelL3Conv_1//2)-1))
        self.pool3 = nn.AvgPool2d(1,4)#(self.kernelL3Conv_1//2))
        #EEGNet - Clasification Layer
        self.lin = nn.Linear(self.length, n_classes)
        self.softMax = nn.Softmax(dim=1)
    def forward(self,x):
        #Reshape for the Convolutions
        x = x.view(-1,1,self.n_channels,self.n_time_points);
        #Layer - 1
        x = self.l1Conv2d(x)

        x = self.batch1(x)
        x = nn.functional.elu(x)
        x = self.padd1(x)

        x = self.l1DeptConv2d(x)

        x = self.batch1_2(x)
        x = nn.functional.elu(x)
        x = nn.functional.dropout2d(x, 0.25)
        #Layer - 2
        x = self.conv2(x)
        x = self.conv2_2(x)
        x = self.batch2(x)
        x = nn.functional.elu(x)
        x = self.padd2(x)

        x = self.pool2(x)
        x = nn.functional.dropout2d(x, 0.25)
        #Layer - 3
        x = self.conv3(x)
        x = self.conv3_2(x)
        x = self.batch3(x)
        x = nn.functional.elu(x)
        x = self.padd3(x)

        x = self.pool3(x)
        x = nn.functional.dropout2d(x, 0.25)
        #Clasification Layer
        x = x.view(-1,self.length)
        x = self.lin.forward(x)
        x = self.softMax.forward(x)
        x = x.squeeze()
        return x
    def reset(self):
        self.l1Conv2d.reset_parameters()
        self.l1DeptConv2d.reset_parameters()
        self.conv2.reset_parameters()
        self.conv2_2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv3_2.reset_parameters()
        self.lin.reset_parameters()
    def name(self):
        return "EEG Net (2018)"

class EEGNet_2016(nn.Module):
    """EEGNet : Compact CNN Architecture in its first version, 2016.
    Input is: torch.Size([#Samples, #Channels, #TimePoints]), ex: torch.Size([316, 28, 500]) or torch.Size([316, 28, 50])
    Described in :
    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and
    B. J. Lance. EEGNet: A Compact Convolutional Network for EEG-based
    Brain-Computer Interfaces. CoRR, abs/1611.08024, 2016
    https://arxiv.org/abs/1611.08024"""
    def __init__(self, n_channels, n_time_points, n_filters=16, n_classes=2):
        super(EEGNet_2016, self).__init__()
        #For the last layer, the classification layer lenght
        #Depends on the time_points. The dependence is specified to be 4X4XT/16 in the paper but
        #had to twich it to work.
        if (n_time_points == 50):
            self.length = (((n_time_points//16)*4*4)//4 ) #Works for 50
        elif(n_time_points == 500):
            self.length = (((n_time_points//16)*4*4)//4 +4) #Works for 500
        else:
            self.length = (((n_time_points//16)*4*4)) #As in the paper eventhough (((T//16)*4*4)//2) is used in the implementation.
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        #EEGNet - Layer - 1
        filters_layer_1 = n_filters
        self.conv1 = nn.Conv2d(1,filters_layer_1,(n_channels,1),padding = 0)
        self.batch1 = nn.BatchNorm2d(filters_layer_1,False)
        #EEGNet - Layer - 2
        self.padd1 = nn.ZeroPad2d((16,17,0,1))
        filters_layer_2 = 4
        self.conv2 = nn.Conv2d(1,filters_layer_2,(2,32))
        self.batch2 = nn.BatchNorm2d(filters_layer_2,False)
        self.pool2 = nn.MaxPool2d(2,4)
        #EEGNet - Layer - 3
        self.padd2 = nn.ZeroPad2d((2, 1, 4, 3))
        filters_layer_3 = 4
        self.conv3 = nn.Conv2d(4,filters_layer_3, (8,4))
        self.batch3 = nn.BatchNorm2d(filters_layer_3,False)
        self.pool3 = nn.MaxPool2d(2,4)
        #EEGNet - Clasification Layer
        self.lin = nn.Linear(self.length, n_classes)
        self.softMax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        #Reshape for the Convolutions
        x = x.view(-1,1,self.n_channels,self.n_time_points);
        #Layer - 1
        x = nn.functional.elu(self.conv1(x))
        x = self.batch1(x)
        x = nn.functional.dropout(x, 0.25)
        x = x.permute(0, 2, 1, 3)
        #Layer - 2
        x = self.padd1(x)
        x = nn.functional.elu(self.conv2(x))
        x = self.batch2(x)
        x = nn.functional.dropout(x, 0.25)
        x = self.pool2(x)
        #Layer - 3
        x = self.padd2(x)
        x = nn.functional.elu(self.conv3(x))
        x = self.batch3(x)
        x = nn.functional.dropout(x, 0.25)
        x = self.pool3(x)
        #Clasification Layer
        x = x.view(-1,self.length)
        x = self.lin.forward(x)
        x = self.softMax.forward(x)
        x = x.squeeze()
        return x
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin.reset_parameters()
    def name(self):
        return "EEG Net (2016)"

class ShallowConvNet(nn.Module):
    """ShallowConvNet:
    Input is: torch.Size([#Samples, #Channels, #TimePoints]), ex: torch.Size([316, 28, 500])
    Note: the network dimension are chosen to be convenient for an input of 500 or 50 time points.
    Other size of inputs are not guaranteed to work as well
    Described in:
    R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter,
    K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball.
    Deep learning with convolutional neural networks for EEG decoding and
    visualization: Convolutional Neural Networks in EEG Analysis. Human Brain
    Mapping, 38(11):5391–5420, Nov. 2017.
    https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730"""
    def __init__(self, n_channels, n_time_points, n_classes=2):
        super(ShallowConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.scaling = n_time_points/500
        self.timeConv = nn.Conv2d(1,40,(1,math.ceil(25*self.scaling)))
        self.spatConv = nn.Conv2d(40,40,(self.n_channels,1),bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.meanPooling = nn.AvgPool2d((1,math.ceil(75*self.scaling)),stride=(1,math.ceil(15*self.scaling)))
        self.dropout = nn.Dropout(p=0.5)
        if(n_time_points == 500):
            self.classifier = nn.Conv2d(40,2,(1,27))
        elif(n_time_points==50):
            self.classifier = nn.Conv2d(40,2,(1,21))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1,1,self.n_channels,self.n_time_points);
        x = self.timeConv(x)
        x = self.spatConv(x)
        x = self.batchnorm(x)
        x = x*x;
        x = self.meanPooling(x)
        x = torch.log(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.softmax(x)
        x = x.squeeze()
        return x

    def reset(self):
        self.timeConv.reset_parameters()
        self.spatConv.reset_parameters()
        self.classifier.reset_parameters()

    def name(self):
        return "ShallowConvNet"

class ShallowConvNet_SR(nn.Module):
    """ShallowConvNet:
    Input is: torch.Size([#Samples, #Channels, #TimePoints]), ex: torch.Size([316, 28, 500])
    Described in:
    R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter,
    K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball.
    Deep learning with convolutional neural networks for EEG decoding and
    visualization: Convolutional Neural Networks in EEG Analysis. Human Brain
    Mapping, 38(11):5391–5420, Nov. 2017.
    https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

    This version tests a rescaling of the original ShallowConvNet with respect to sampling rate
    rather than the number of time points
    """
    def __init__(self, n_channels, n_time_points, sampling_rate, n_classes=2):
        super(ShallowConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.timeConv = nn.Conv2d(1,40,(1,int(sampling_rate/10)+1)) # +1 For having convenient output dim (400 or 40)
        self.spatConv = nn.Conv2d(40,40,(self.n_channels,1),bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.meanPooling = nn.AvgPool2d((1,5*int(sampling_rate/100)),stride=int(sampling_rate/100)) #5 kernels superposition as in ref
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Conv2d(40,2,(1,36))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1,1,self.n_channels,self.n_time_points);
        x = self.timeConv(x)
        x = self.spatConv(x)
        x = self.batchnorm(x)
        x = x*x;
        x = self.meanPooling(x)
        x = torch.log(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.softmax(x)
        x = x.squeeze()
        return x

    def reset(self):
        self.timeConv.reset_parameters()
        self.spatConv.reset_parameters()
        self.classifier.reset_parameters()

    def name(self):
        return "ShallowConvNet"
