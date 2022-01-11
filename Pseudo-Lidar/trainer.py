from __future__ import absolute_import, division, print_function

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
def set_random_seed(seed):
    if seed >= 0:
        print("Set random seed@@@@@@@@@@@@@@@@@@@@")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
class Trainer:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        #######################
        if not self.opt.debug:
            wandb.init(project="Access", entity="bigchan")
            wandb.run.name =self.opt.model_name
        #######################
        #######################
        set_random_seed(42)
        if os.path.isdir(self.log_path) is False:
            os.makedirs(self.log_path)
        if not self.opt.debug:
            tar = tarfile.open( os.path.join(self.log_path, 'sources.tar'), 'w' )
            ####################### 
            tar.add( 'networks' )
            tar.add( 'trainer.py' )
            tar.add( 'train.py' )
            tar.add( 'start2train.sh' )
            tar.add( 'utils.py' )
            tar.add( 'datasets' )
            tar.add( 'layers.py' )
            tar.add( 'options.py' )
            tar.add( 'hr_layers.py' )
            tar.close()
            #######################

        for arg in vars(self.opt):
            print("%30s :"%arg, getattr(self.opt, arg))
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        #able if not using use_stereo or frame_ids !=0
        #use_stereo defualt disable
        #frame_ids defualt =[0,-1,1]

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
        self.parameters_to_train += list(self.models["encoder"].parameters())
        if self.opt.distill:
            self.models["encoder_t"].to(self.device)
            self.parameters_to_train += list(self.models["encoder_t"].parameters())
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        if self.opt.distill:
            self.models["depth_t"].to(self.device)
            self.parameters_to_train += list(self.models["depth_t"].parameters())
            
        para_sum = sum(p.numel() for p in self.models['depth'].parameters())
        
        print('params in depth decoder',para_sum)
        ####################################################
        if self.opt.discriminator:
            self.model_D=[]
            self.model_optimizer_D=[]
            for i in range(len(self.opt.scales)):
                self.model_D.append(FCDiscriminator(1).to(self.device))
                self.model_optimizer_D.append(optim.Adam(list(self.model_D[i].parameters()),1e-4))#learning_rate=1e-4
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if self.opt.transloss:
            
            config = get_config("configs/hrt/hrt_tiny.yaml")
            self.trans_model=HighResolutionTransformer(config["MODEL"]["HRT"]).cuda()
            self.trans_model.init_weights("hrt_tiny.pth")

            for param in self.trans_model.parameters():
                param.requires_grad = False
            self.trans_model.eval()
        if self.opt.vggloss:
            self.vggmodel=torchvision.models.vgg19(pretrained=True).features.cuda()
            for param in self.vggmodel.parameters():
                param.requires_grad = False
            self.vggmodel.eval()
        if self.opt.patchvlad:
            self.patchmodel=load_patchNet().cuda()
            self.patchmodel.eval()
        ####################################################
        
        self.model_optimizer = optim.AdamW(self.parameters_to_train, 0.5 * self.opt.learning_rate)#learning_rate=1e-4
        
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'
        
        if self.opt.load_weights_folder is not None and self.opt.model !="GBNet_v2" :
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "vk2":datasets.VK2Dataset,
                         "kaist":datasets.KAISTRAWDataset
                         }
        
        self.dataset = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        #change trainset
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        #dataloader for kitti
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,thermal=self.opt.thermal)
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch_k = train_dataset.__len__() // self.opt.batch_size

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items\n".format(
            len(train_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.init_time = time.time()
#         if isinstance(self.opt.load_weights_folder,str):
#             self.epoch_start = int(self.opt.load_weights_folder[-1]) + 1
#         else:
        self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
                if self.opt.dataset=="kaist":
                    if self.opt.distill:
                        model_encoder=self.models["encoder_t"]
                        model_decoder=self.models["depth_t"]
                    else:
                        model_encoder=self.models["encoder"]
                        model_decoder=self.models["depth"]
                    evalresult=evaluate_with_train(model_encoder,\
                                        model_decoder,self.opt.num_workers,\
                                        self.opt.data_path,self.opt.split\
                                        ,self.opt.height,self.opt.width,self.opt)
                    if not self.opt.debug:
                        self.log_eval(evalresult)
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Threads: " + str(torch.get_num_threads()))
        print("Training")
        self.set_train()
        self.every_epoch_start_time = time.time()
        source_label=0
        target_label=1
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
                    
            outputs, losses,outputs_t = self.process_batch(inputs)
            total_loss= losses["loss"]
            ####################################################
            self.model_optimizer.zero_grad()
            ####################################################
            
            if self.opt.discriminator:
                for i in self.opt.scales:
                    self.model_optimizer_D[i].zero_grad()
            if self.opt.discriminator:
                ## train_G
                for i in self.opt.scales:
                    for param in self.model_D[i].parameters():
                        param.requires_grad = False
                ### thermal_0
                for i in self.opt.scales:
                    D_t_0 = self.model_D[i](F.softmax(outputs_t[('disp', i)]))
                    D_t_0_loss = \
                    self.bce_loss(D_t_0,Variable(torch.FloatTensor(D_t_0.data.size()).fill_(source_label)).to(self.device))
                    losses["loss/D_t_0_{}".format(i)] = D_t_0_loss
                    total_loss+=D_t_0_loss
            ####################################################
            total_loss.backward()
            ####################################################
            if self.opt.discriminator:
                ## train_D
                for i in self.opt.scales:
                    for param in self.model_D[i].parameters():
                        param.requires_grad = True

                for i in self.opt.scales:
                    ### RGB_0
                    D_R_0 = self.model_D[i](F.softmax(outputs[('disp', i)].detach()))
                    loss_rgb_0 =self.bce_loss(D_R_0,Variable(torch.FloatTensor(D_R_0.data.size()).fill_(source_label)).to(self.device))
                    losses["loss/D_R_0_{}".format(i)] = loss_rgb_0
                    loss_rgb_0.backward()
                    ### thermal_1
                    D_t_1 = self.model_D[i](F.softmax(outputs_t[('disp', i)].detach()))
                    loss_thermal_1 =self.bce_loss(D_t_1,Variable(torch.FloatTensor(D_t_1.data.size()).fill_(target_label)).to(self.device))
                    losses["loss/D_t_1_{}".format(i)] = loss_thermal_1
                    loss_thermal_1.backward()
            ####################################################
            self.model_optimizer.step()
            ####################################################
            if self.opt.discriminator:
                for i in self.opt.scales:
                    self.model_optimizer_D[i].step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000#log_fre 's defualt = 250
            late_phase = self.step % 1000 == 0
            
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                if not self.opt.debug:
                    if self.opt.distill:
                        self.log("train", inputs, outputs, losses,outputs_t)
                    else:
                        self.log("train", inputs, outputs, losses,outputs)
            self.step += 1
        
        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))
   
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?
            
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            if self.opt.distill:
                self.features_color = self.models["encoder"](inputs["color_aug", 0, 0])
                self.features_thermal = self.models["encoder_t"](inputs["thermal", 0, 0])
                # import pdb;pdb.set_trace()
                outputs_c = self.models["depth"](self.features_color)
                outputs_t = self.models["depth_t"](self.features_thermal)
                # import pdb;pdb.set_trace()
            else:
                if self.opt.thermal:
                    features = self.models["encoder"](inputs["thermal", 0, 0])
                else:
                    features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs_c = self.models["depth"](features)

        self.generate_images_pred(inputs, outputs_c)
        
        if self.opt.distill:
            losses = self.compute_losses(inputs, outputs_c,outputs_t)
            return outputs_c, losses,outputs_t
        else:
            losses = self.compute_losses(inputs, outputs_c,outputs_c)
            return outputs_c, losses,outputs_c
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #pose_feats is a dict:
            #key:
            """"keys
                0
                -1
                1
            """
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    #axisangle and translation are two 2*1*3 matrix
                    #f_i=-1,1
                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            if self.opt.dataset=="kaist":
                if self.opt.scale_depth:
                    _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                elif self.opt.softplus:
                    depth=torch.log(torch.exp(disp)+1)
                else:
                    depth=1/disp
            else:
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) #disp_to_depth function is in layers.py
            
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    #doing this
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    def compute_trans_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        
        B,C,H,W=pred.shape
        if C==1:
            pred=torch.cat((pred,pred,pred),dim=1)
            target=torch.cat((target,target,target),dim=1)
#         import pdb;pdb.set_trace()
        pred_features=self.trans_model(pred)
        target_features=self.trans_model(target)
        len_=len(pred_features)
        loss_trans=0
#         import pdb;pdb.set_trace()
        for i in range(len_):
            loss_trans+=torch.abs(pred_features[i]-target_features[i]).mean()
        return loss_trans/len_
    def foward_vlad(self,input_data):
        pool_size=4096
        image_encoding = self.patchmodel.encoder(input_data)
        vlad_local, vlad_global = self.patchmodel.pool(image_encoding)
        vlad_global_pca = get_pca_encoding(self.patchmodel, vlad_global)
        vlad_local_pca=[]
        for this_iter, this_local in enumerate(vlad_local):
            this_local_pca = get_pca_encoding(self.patchmodel, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))).\
                    reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            vlad_local_pca.append(this_local_pca)
        return vlad_global_pca,vlad_local_pca
    def compute_patch_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        
        B,C,H,W=pred.shape
        if C==1:
            pred=torch.cat((pred,pred,pred),dim=1)
            target=torch.cat((target,target,target),dim=1)
        
        vlad_global_pca_pred,vlad_local_pca_pred=self.foward_vlad(pred)
        vlad_global_pca_target,vlad_local_pca_target=self.foward_vlad(target)
        loss=torch.abs(vlad_global_pca_pred-vlad_global_pca_target).mean()
        for i in range(len(vlad_local_pca_pred)):
            loss+=torch.abs(vlad_local_pca_pred[i]-vlad_local_pca_target[i]).mean()
        
        return loss
    def compute_vgg_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        
        B,C,H,W=pred.shape
        if C==1:
            pred=torch.cat((pred,pred,pred),dim=1)
            target=torch.cat((target,target,target),dim=1)
        pred_features=self.vggmodel(pred)
        target_features=self.vggmodel(target)
#         import pdb;pdb.set_trace()
        loss_vgg=torch.abs(pred_features-target_features).mean()
        return loss_vgg
    
    def compute_distill_loss(self, pred, target):
        g = torch.log(pred) - torch.log(target)
        
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
    
    def compute_losses(self, inputs, outputs, outputs_t):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            
            disp = outputs[("disp", scale)]
            disp_t = outputs_t[("disp", scale)]
            
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            if self.opt.model=="GBNet_v2":
                disp_feature = self.features_color[("disp", scale)]
                disp_t_feature = self.features_thermal[("disp", scale)]
            
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                    if self.opt.vggloss:
#                         import pdb;pdb.set_trace()
                        loss_vgg=self.compute_vgg_loss(pred, target)
                        loss +=loss_vgg
                    if self.opt.transloss:
                        loss_trans=self.compute_trans_loss(pred, target)*0.1
                        loss +=loss_trans
                    if self.opt.patchvlad:
                        loss_patch=self.compute_patch_loss(pred, target)
                        loss +=loss_patch
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                #doing_this
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                #doing_this
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                #doing this
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                
                #outputs["identity_selection/{}".format(scale)] = (
                outputs["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
                automask=(idxs > identity_reprojection_loss.shape[1] - 1).float().unsqueeze(1)
                
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)#defualt=1e-3 something with get_smooth_loss function

            if self.opt.distill and self.opt.compute:
#                 try:
                    disp=F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)            
                    disp_t=F.interpolate( disp_t, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    loss_tcdistill=self.compute_reprojection_loss(disp_t, disp.detach())*automask  

                    loss += loss_tcdistill.mean()#*10

                    if self.opt.model=="GBNet_v2":
                        # import pdb;pdb.set_trace()
                        disp_feature=F.interpolate(disp_feature, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)            
                        disp_t_feature=F.interpolate( disp_t_feature, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    
                        loss_tcdistillv2=self.compute_reprojection_loss(disp_t_feature, disp_feature.detach())*automask  

                        loss += loss_tcdistillv2.mean()#*10

#                     if self.opt.SIlogloss:
#                         loss_SIlogs=self.compute_distill_loss(disp_t, disp.detach())*0.01
#                         loss += loss_SIlogs
                        
#                     if self.opt.transloss:
#                         loss_trans=self.compute_trans_loss(disp_t, disp.detach())*0.1
#                         loss +=loss_trans

#                     if self.opt.vggloss:
#                         loss_vgg=self.compute_vgg_loss(disp_t, disp.detach())

#                         loss +=loss_vggs
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        
        total_loss /= self.num_scales
        losses["loss"] = total_loss 
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so i#s only used to give an indication of validation performance


        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log_eval(self, evalresult):
        for l, v in evalresult.items():
            wandb.log({"{}".format(l): v})
            
    def log(self, mode, inputs, outputs, losses,outputs_t):
        """Write an event to the tensorboard events file
        """
#         writer = self.writers[mode]
        for l, v in losses.items():
            wandb.log({"{}".format(l): v})
#             writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    wandb.log({"color_{}_{}/{}".format(frame_id, s, j): wandb.Image(inputs[("color", frame_id, s)][j].data)})
                    if frame_id!="s" and self.opt.thermal:
                        wandb.log({"thermal_{}_{}/{}".format(frame_id, s, j): wandb.Image(inputs[("thermal", frame_id, s)][j].data)})
                    if s == 0 and frame_id != 0:
                        wandb.log({"color_pred_{}_{}/{}".format(frame_id, s, j): wandb.Image(outputs[("color", frame_id, s)][j].data)})
                
                wandb.log({"disp_{}/{}".format(s, j): wandb.Image(normalize_image(outputs[("disp", s)][j]))})
                wandb.log({"disp_t_{}/{}".format(s, j): wandb.Image(normalize_image(outputs_t[("disp", s)][j]))})

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    wandb.log({"automask_{}/{}".format(s, j): wandb.Image(outputs["identity_selection/{}".format(s)][j][None, ...])})
                              
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    def tensor2cv(self,img):
        disp=pred_disp[0].cpu().detach().numpy().transpose((1,2,0))
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

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
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            if self.opt.distill:
                n_t=n+"_t"
                print("Loading {} weights...".format(n_t))
                path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
                model_dict = self.models[n_t].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n_t].load_state_dict(model_dict)      
                
        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
