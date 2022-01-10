import argparse
import configparser
import os
from os.path import join, exists, isfile
from os import makedirs

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from patchnetvlad.tools.datasets import PlaceDataset
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

def load_patchNet():
    config = configparser.ConfigParser()
    config.read(join(PATCHNETVLAD_ROOT_DIR, 'configs/performance.ini'))
    encoder_dim, encoder = get_backend()
    resume_ckpt='/home/dchan/workspace/Access/Depth/Patch-NetVLAD/patchnetvlad/./pretrained_models/mapillary_WPCA4096.pth.tar'
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])
    pool_size = int(config['global_params']['num_pcs'])
    model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model