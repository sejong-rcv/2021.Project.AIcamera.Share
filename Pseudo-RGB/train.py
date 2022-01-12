import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import wandb
from tqdm import tqdm
from image_util import *
import _init_paths
import tarfile
# from config import config
# from config import update_config                                               

opt = TrainOptions().parse()

def set_random_seed(seed):
    if seed >= 0:
        import random
        print("Set random seed@@@@@@@@@@@@@@@@@@@@")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)
# update_config(opt.config, opt)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0    

opt.print_freq = lcm(opt.print_freq, opt.batch_size)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
else:
    source_dir = os.path.join("checkpoint",opt.name, "source")
    if os.path.isdir(source_dir) is False:
        os.makedirs(source_dir)
    tar = tarfile.open( os.path.join(source_dir, 'sources.tar'), 'w' )

    tar.add( 'models' )
    tar.add( 'configs' )
    tar.add( 'train.py' )
    tar.add( 'scripts' )
    tar.close()

    wandb.init(project=opt.project_name)
    wandb.run.name = opt.name

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
model = nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()

optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter


display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in tqdm(range(start_epoch, opt.niter + opt.niter_decay + 1)):
    epoch_start_time = time.time()
    if not opt.debug:
        wandb.log({"epoch":epoch})

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        
        # whether to collect output images
        if opt.debug:
            save_fake=True
        else:
            save_fake = total_steps % opt.display_freq == display_delta

        ############## Image Processing ##################
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
        ############## Forward Pass ######################
        
        losses, generated = model(data['label'], data['image'], infer=save_fake)
        
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))
        
        # calculate final loss scalar
        ############### Backward Pass ####################
        # update generator weights
        if not opt.no_gan_loss:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_L1',0)

            optimizer_G.zero_grad()
            loss_G.backward()          
            optimizer_G.step()      

            # update discriminator weights
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            if not opt.debug:
                for name in errors.keys():
                    wandb.log({name:errors[name]})
        ### display output images
        if save_fake : 
            visuals = OrderedDict([('real_image', util.tensor2im(data['image'], \
                                                                normalize=opt.normalize)),
                                ('synthesized_image', util.tensor2im(generated, \
                                                                    normalize=opt.normalize)),                             
                                ('input_label', util.tensor2label(data['label'], opt.label_nc))
                                ])

            if not opt.debug:
                for label, image in visuals.items():
                    wandb.log({label:wandb.Image(image)})

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
      
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
