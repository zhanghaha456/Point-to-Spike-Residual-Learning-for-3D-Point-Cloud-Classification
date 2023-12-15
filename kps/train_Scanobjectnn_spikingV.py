#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.Scanobjectnn import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer_spikingV import ModelTrainer
from models.architectures_spikingV import KPSNet


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class ScanobjectnnConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'Scanobjectnn'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = 15

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb1',
                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 20

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.032
    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 4
    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.05

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-3
    momentum = 0.98
    lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 10

    # Number of steps per epochs
    epoch_steps = 300

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.1

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

   

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = ScanobjectnnConfig()
 
    # Initialize datasets
    training_dataset = ScanobjectnnDataset(config, train=True)
    test_dataset = ScanobjectnnDataset(config, train=False)
    # # Initialize samplers
    training_sampler = ScanobjectnnSampler(training_dataset, balance_labels=True)
    test_sampler = ScanobjectnnSampler(test_dataset, balance_labels=True)

    # # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=ScanobjectnnCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ScanobjectnnCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    #.calibration(training_loader)
    #test_sampler.calibration(test_loader)

    #debug_timing(test_dataset, test_sampler, test_loader)
    #debug_show_clouds(training_dataset, training_sampler, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPSNet(config)
    print(net)
    # Define a trainer class
    trainer = ModelTrainer(net, config)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training

    trainer.train(net, training_loader, test_loader, config)
   
