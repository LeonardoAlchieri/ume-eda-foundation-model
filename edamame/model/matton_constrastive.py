# CODE DERIVED FROM:
# Contrastive Learning of Electrodermal Activity Representations for Stress Detection
# Katie Matton (kmatton@mit.edu), CSAIL and MIT Media Lab, Massachusetts Institute of Technology, United States
# Robert Lewis (roblewis@mit.edu), MIT Media Lab, Massachusetts Institute of Technology, United States
# John Guttag (guttag@mit.edu), CSAIL, Massachusetts Institute of Technology, United States
# Rosalind Picard (picard@mit.edu), MIT Media Lab, Massachusetts Institute of Technology, United States


"""
Basic architecture for contrastive learning network.
"""

import torch.nn as nn


class ContrastiveModel(nn.Module):
    """
    Class for basic contrastive learning model.
    """

    def __init__(
        self,
        encoder_net: nn.Module,
        loss_fn: nn.Module,
        transform_head: nn.Module,
    ):

        """
        encoder_name: name of network to use for mapping from raw data to learned representations (h)
        encoder_args: arguments for encoder network
        encoder_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        transform_head_name (optional): name of network for mapping from learned representations (h) to metric embedding
        transform_head_args (optional): arguments for transform head network
        transform_head_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        """
        super().__init__()
        # init encoder and transformer head
        self.encoder_net = encoder_net
        self.transform_head = transform_head

        self.nn_dict = nn.ModuleDict({"encoder_net": self.encoder_net})
        self.output_dim = self.encoder_net.output_dim

        self.loss_fn = loss_fn

        if self.transform_head is not None:
            self.nn_dict["transform_head"] = self.transform_head
            self.output_dim = self.transform_head.output_dim


    def forward(self, x_1, x_2):
        
        # get two views of data
        h_1 = self.encoder_net(x_1)
        h_2 = self.encoder_net(x_2)
        if self.transform_head is not None:
            h_1 = self.transform_head(h_1)
            h_2 = self.transform_head(h_2)
        return h_1, h_2

    def loss(self, output, batch, step='train'):
        h_1, h_2 = output  # Assuming forward returns (h_1, h_2)
        return self.loss_fn(h_1, h_2)
    
    def get_representation(self, x):
        h = self.encoder_net(x)
        if self.transform_head is not None:
            h = self.transform_head(h)
        return h