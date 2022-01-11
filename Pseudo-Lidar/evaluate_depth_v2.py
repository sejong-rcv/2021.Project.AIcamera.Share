from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from  torchvision.utils import save_image
from layers import disp_to_depth
from utils import readlines, sec_to_hm_str
from options import MonodepthOptions
import datasets
import networks
from tqdm import tqdm
from datetime import datetime
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import torchvision
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from IPython import embed
import wandb
import random
import tarfile
from evaluate_depth import evaluate_with_train
from networks.discriminator import FCDiscriminator
from torch.autograd import Variable
from models.hrt import  HighResolutionTransformer
from config import get_config
from Load_patchNet import load_patchNet
from transdssl.transdssl_encoder import TRANSDSSLEncoder
from transdssl.transdssl_decoder import TRANSDSSLDecoder
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from evaluate_depth import compute_errors

import matplotlib.pyplot as plt
import matplotlib as mpl

##########################################################################################

def set_random_seed(seed):
    if seed >= 0:
        print("Set random seed@@@@@@@@@@@@@@@@@@@@")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
class Evaluater:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        for arg in vars(self.opt):
            print("%30s :"%arg, getattr(self.opt, arg))
        
        self.models = {}
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        if self.opt.model=="DIFF":
            self.models["encoder"] = networks.test_hr_encoder.hrnet18(True)
            self.models["encoder"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
            self.models["depth"] = networks.HRDepthDecoder(self.models["encoder"].num_ch_enc, scales=self.opt.scales,opt=self.opt)
            
            if self.opt.distill:
                self.models["encoder_t"] = networks.test_hr_encoder.hrnet18(True)
                self.models["encoder_t"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
                self.models["depth_t"] = networks.HRDepthDecoder(self.models["encoder"].num_ch_enc, scales=self.opt.scales,opt=self.opt)
            
            para_sum = sum(p.numel() for p in self.models['encoder'].parameters())
            print('params in encoder',para_sum)

        elif self.opt.model=="resnet":
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            if self.opt.distill:
                self.models["encoder_t"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained")
                self.models["depth_t"] = networks.DepthDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales)
                
        elif self.opt.model=="GBNet":
            self.models["encoder"] =networks.test_hr_encoder.hrnet18(True)
            self.models["depth"] = networks.GBNet(self.opt)
            if self.opt.distill:
                self.models["encoder_t"] = networks.test_hr_encoder.hrnet18(True)
                self.models["encoder_t"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
                self.models["depth_t"] =  networks.GBNet(self.opt)
        elif self.opt.model=="GBNet_v2":
            self.models["encoder"] =networks.GBNetEncoder(self.opt)
            self.models["depth"] = networks.GBNet_v2(self.opt)
            if self.opt.distill:
                self.models["encoder_t"] = networks.GBNetEncoder(self.opt)
                self.models["encoder_t"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
                self.models["depth_t"] = self.models["depth"] # networks.GBNet(self.opt)
            
        elif self.opt.model=="transdssl":
            self.models["encoder"] =TRANSDSSLEncoder(backbone="S",infer=False)
            self.models["depth"] = TRANSDSSLDecoder(backbone="S",infer=False)
            
        self.models["encoder"].to(self.device)
        if self.opt.distill:
            self.models["encoder_t"].to(self.device)
        
        self.models["depth"].to(self.device)
        if self.opt.distill:
            self.models["depth_t"].to(self.device)
            
        para_sum = sum(p.numel() for p in self.models['depth'].parameters())
        
        print('params in depth decoder',para_sum)
 
        self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        cmap = plt.get_cmap('magma');
        cval = [[cmap(x)[0],cmap(x)[1],cmap(x)[2]] for x in range(0,255)];
        cval.append( cval[-1] );
        self.cval = np.array( cval );
        
    def tensor2cv(self,img,idx):
        img=img[idx].cpu().detach().numpy().transpose((1,2,0))
        
        
        return img
    def evaluate_with_depth(self,encoder,depth_decoder,num_workers,data_path,eval_split,height,width,opt):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 50
        splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        filenames = readlines(os.path.join(splits_dir, eval_split, "test_files.txt"))
        dataset = datasets.KAISTRAWDataset(data_path, filenames,height,width,[0], 4, is_train=False,thermal=opt.thermal)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                                pin_memory=True, drop_last=False)
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        gt_depths = []
        print("-> Computing predictions with size {}x{}".format(
            width,height))
        if not os.path.exists(self.opt.load_weights_folder+"/INPUT"):
            os.makedirs(self.opt.load_weights_folder+"/INPUT")
            os.makedirs(self.opt.load_weights_folder+"/DISP")
        if not os.path.exists(self.opt.load_weights_folder+"/GT"):
            os.makedirs(self.opt.load_weights_folder+"/GT")
        name_idx=0     
        with torch.no_grad():
            for idx,data in enumerate(tqdm(dataloader)):

                if opt.thermal or opt.distill:
                    input_color = data[("thermal", 0, 0)].cuda()
                else:    
                    input_color = data[("color", 0, 0)].cuda()
                output = depth_decoder(encoder(input_color))

                pred_disp=output[("disp", 0)]

                if opt.scale_depth:
                    pred_disp,_=disp_to_depth(pred_disp, opt.min_depth, opt.max_depth)
                for i in range(input_color.shape[0]):
                    color_numpy=self.tensor2cv(input_color,i)
                    pred_disp_numpy=self.tensor2cv(pred_disp,i)
                    depth_GT_numpy=self.tensor2cv(data["depth_gt"],i)
                    # name_idx=input_color.shape[0]*idx+i
                    # print(name_idx)
                    # import pdb;pdb.set_trace()
                    plt.imsave(self.opt.load_weights_folder+"/INPUT/%05d.png"%name_idx,color_numpy)
                    plt.imsave(self.opt.load_weights_folder+"/DISP/%05d.png"%name_idx,pred_disp_numpy[:,:,0],cmap="magma")
                    plt.imsave(self.opt.load_weights_folder+"/GT/%05d.png"%name_idx,1/depth_GT_numpy[:,:,0],cmap="magma")
                    name_idx+=1
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                gt_depths.append(data["depth_gt"].squeeze().cpu().numpy())

                pred_disps.append(pred_disp)

        errors = []
        ratios = []

        for ii in tqdm(range(len(pred_disps))):
            for i in range(len(pred_disps[ii])):
                gt_depth = gt_depths[ii][i]
                gt_height, gt_width = gt_depth.shape[:2]
                pred_disp = pred_disps[ii][i]
                pred_disp     = pred_disp#*10 #*1280
                if opt.softplus:
                    pred_depth=np.log(np.exp(pred_disp)+1)
                else:
                    pred_depth = 1 / pred_disp

                mask = gt_depth > 0
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                errors.append(compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
        results_error={"abs_rel":mean_errors[0],"sq_rel":mean_errors[1],"rmse":mean_errors[2],\
                      "rmse_log":mean_errors[3],"a1":mean_errors[4],"a2":mean_errors[5],\
                      "a3":mean_errors[6]}
        return results_error
    def colormap(self,depth):
        depth = depth * (255.);
        depth = self.cval[(depth).astype(np.uint8)];
        depth = (depth*255.).astype(np.uint8);
        depth = depth[:,:,[2,1,0]];
        return depth

    def eval_(self):
        """Run the entire training pipeline
        """
        if self.opt.distill:
            model_encoder=self.models["encoder_t"]
            model_decoder=self.models["depth_t"]
        elif self.opt.model =="GBNet_v2": 
            model_encoder=self.models["encoder_t"]
            model_decoder=self.models["depth_t"]
        else:
            model_encoder=self.models["encoder"]
            model_decoder=self.models["depth"]
            
        evalresult=self.evaluate_with_depth(model_encoder,\
                            model_decoder,self.opt.num_workers,\
                            self.opt.data_path,self.opt.split\
                            ,self.opt.height,self.opt.width,self.opt)
        # import pdb;pdb.set_trace()


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            print(path)
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            if self.opt.distill:
                n_t=n+"_t"
                print("Loading {} weights...".format(n_t))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n_t))
                model_dict = self.models[n_t].state_dict()
                pretrained_dict = torch.load(path)
                print(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n_t].load_state_dict(model_dict)      
                
        # # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
if __name__ == "__main__":
    
    from options import MonodepthOptions

    options = MonodepthOptions()
    opts = options.parse()
    evaler = Evaluater(opts)
    evaler.eval_()
