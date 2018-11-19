import torch
from torch import Tensor
import math

#lower_bound_frequency = 1
#upper_bound_frequency = 2
#upper_bound_amplitude = 1
#after_normalization = False
# random phase

def augment_data(shape, lower_bound_frequency = 1, upper_bound_frequency = 5, upper_bound_amplitude = 1):
    '''
    Takes the parameters of the signal and the shape of the data to augment.
    Each time series (for each sample and each channel) has it's own frequency, amplitude and phase shift.
    Please look into the report for more details
    Input: shape -> shape of the signal to output. The signal is assumed to be of dimension 3, of the form (n_samples, n_channels, time_points)
           lower_, upper_bound_frequency -> upper and lower bounds for the frequency of the added signals
           upper_bound_amplitude -> bound on the amplitude of the added signal
    Output: x -> a time series of size (shape), with frequencies and amplitude in the right range. See report for details
    '''
    n_observations = shape[0]
    n_samples = shape[2]
    n_channels = shape[1]

    time_discr = torch.linspace(0, 0.5, n_samples).view(1,1,-1) # discretize the time length
    time_discr.expand(n_observations, n_channels,-1)
    frequencies = (upper_bound_frequency-lower_bound_frequency)*(torch.rand(n_observations, n_channels, 1).expand(-1,-1, n_samples) + lower_bound_frequency)# for each channel, for each observation, take a random frequency
    phase_shift = 2*math.pi*torch.rand(n_observations, n_channels, 1).expand(-1,-1, n_samples)
    amplitude = upper_bound_amplitude*torch.rand(n_observations, n_channels, 1).expand(-1,-1, n_samples)

    x = amplitude*torch.cos(frequencies*time_discr + phase_shift) # what we add
    return x

def augment_train(train_input, train_target, n_augmentation = 2, lower_bound_f = 1, upper_bound_f = 5, upper_bound_amplitude = 2,verbose=0):
    """
    Routine that augments the train set: both the input and target.
    Input: train_input, train_target ->
           n_augmentation -> the number of augmentations to perform
           lower_bound_f, upper_bound_f, upper_bound_amplitude -> the parameters for augment_data
    Output: (train_input, train_target) -> augmented input, with the right labels
    """
    if(verbose >=1 and n_augmentation > 0):
        print("Augmenting the train data...")
    shape = train_input.shape
    temp_input = train_input # we will be concatenating the successive augmentations
    temp_target = train_target
    for e in range(0,n_augmentation):
        # The slow part, involving concatenating
        temp_input = torch.cat((temp_input, train_input + augment_data(shape, lower_bound_f, upper_bound_f, upper_bound_amplitude)),0)
        temp_target = torch.cat((temp_target, train_target),0)
    train_target = temp_target
    train_input = temp_input
    return (train_input, train_target)
