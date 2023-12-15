#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#
import torch
import torch.nn as nn
import math
from models.blocks_spikingV import *
import numpy as np
from spikingjelly.activation_based import neuron
from datetime import datetime
def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0
    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPSNet(nn.Module):
    """
    Class defining KPSNet
    """

    def __init__(self, config):
        super(KPSNet, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'strided' in block or 'pool' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        #self.average= GlobalAverageBlock()
        self.head_mlp=nn.Sequential(nn.Linear(out_dim,1024),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    nn.Linear(1024,config.num_classes))
        # self.head_mlp = UnaryBlock1(out_dim, 1024, False, 0)
        # self.head_softmax = UnaryBlock1(1024, config.num_classes, False, 0)
        #self.sn=neuron.IFNode()
        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):
        # x = batch.features.clone().detach()
        # for j in range(len(self.block_ops) - 1):
        #      x = self.block_ops[j](x, batch)
        # x=self.sn(x)
        # x = self.block_ops[-1](x, batch)
        # x = self.head_mlp(x)
        # return x #if-av-fc
        x = batch.features.clone().detach()
        for block_op in self.block_ops:
             x = block_op(x, batch) 
        x=self.head_mlp(x)
        return x
    
    
    

       

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total

