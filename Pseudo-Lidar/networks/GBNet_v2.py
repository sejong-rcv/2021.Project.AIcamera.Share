# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os 
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from networks.test_hr_encoder import hrnet18
from networks.HR_Depth_Decoder import HRDepthDecoder
from transdssl.transdssl_encoder import TRANSDSSLEncoder
from transdssl.transdssl_decoder import TRANSDSSLDecoder
class GBNet_v2(nn.Module):
    def __init__(self,opt):
        super(GBNet_v2, self).__init__()
        self.opt=opt
        self.scales=self.opt.scales
        num_ch_enc= np.array([ 64, 18, 36, 72, 144 ])
        self.encoder2 = hrnet18(True)
        self.decoder2 = HRDepthDecoder(num_ch_enc, scales=self.opt.scales,opt=self.opt)
        self.encoder3 = hrnet18(True)
        self.decoder3 = HRDepthDecoder(num_ch_enc, scales=self.opt.scales,opt=self.opt)
        self.load_kitti_weight("diffnet_640x192_ms")
    def forward(self, disp1):
#         import pdb;pdb.set_trace()
        self.outputs = {}
        # disp1=self.decoder1(input_features)
        input_2=torch.cat((disp1[("disp", 0)],disp1[("disp", 0)],disp1[("disp", 0)]),dim=1)
        disp2=self.decoder2(self.encoder2(input_2))
        input_3=torch.cat((disp2[("disp", 0)],disp2[("disp", 0)],disp2[("disp", 0)]),dim=1)
        disp3=self.decoder3(self.encoder3(input_3))

        for i in range(4, -1, -1):
            if i in self.scales:
                self.outputs[("disp", i)]=disp1[("disp", i)]*(0.5)+disp2[("disp", i)]*(0.3)+disp3[("disp", i)]*(0.2)
        
        return self.outputs
    def load_weight(self,model,pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
    
    def load_kitti_weight(self,path):
        print("Load Weight")
        encoder_path=os.path.join(path,"encoder.pth")
        decoder_path=os.path.join(path,"depth.pth")
        encoder_statedict=torch.load(encoder_path)
        decoder_statedict=torch.load(decoder_path)
        # self.decoder1=self.load_weight(self.decoder1,decoder_statedict)
        
        self.encoder2=self.load_weight(self.encoder2,encoder_statedict)
        self.decoder2=self.load_weight(self.decoder2,decoder_statedict)
        
        self.encoder3=self.load_weight(self.encoder3,encoder_statedict)
        self.decoder3=self.load_weight(self.decoder3,decoder_statedict)