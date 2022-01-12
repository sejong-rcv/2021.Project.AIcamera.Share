import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from image_util import *
from tqdm import tqdm
import math
import numpy as np
import cv2
from IQA_pytorch import SSIM, MS_SSIM,utils, LPIPSvgg

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
lpips_score = 0

def PSNR(predict, gt):
    predict = np.asarray(predict)
    gt = np.asarray(gt)
    
    mse = np.mean((predict-gt)**2)
    if mse ==0:
        return 100
    PIXEL_MAX = 255.0
     
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

model = create_model(opt)
model.eval()
ssim_score = 0
psnr_score = 0
model1 = SSIM(channels=3)
model2 = LPIPSvgg(channels=3).cuda()

# if opt.verbose:
#     print(model)

predict_list = []
gt_list = []

for i, data in tqdm(enumerate(dataset)):

    ############## Image Processing ##################
    data['label'] = data['label'].cuda()

    with torch.no_grad():   
        generated = model.inference(data['label'])
        # _, generated = model(data['label'], None, infer=True)

    ssim_score += model1(generated, data['image'].cuda(), as_loss=False)
    lpips_score += model2(generated, data['image'].cuda(), as_loss=False)
    
    visuals = OrderedDict([('real_RGB', util.tensor2im(data['image'], \
                                                        normalize=opt.normalize)),
                        ('fake_RGB', util.tensor2im(generated, \
                                                    normalize=opt.normalize)),        
                        ('input_label', util.tensor2label(data['label'], opt.label_nc))
                        ])
    # visuals = OrderedDict([
    #                     ('fake_RGB', util.tensor2im(generated, \
    #                                                 normalize=opt.normalize))
    #                     ])

    # predict_list.append(visuals['fake_RGB'])
    # gt_list.append(visuals['real_RGB'])

    psnr_score += PSNR(visuals['fake_RGB'], visuals['real_RGB'])
    # psnr_score += PSNR(visuals['fake_RGB'], util.tensor2im(data['image']))

    img_path = data['path']
    # print('process image... %s' % img_path)
    
    if opt.save_result is True:
        visualizer.save_images(webpage, visuals, img_path)

# predict_list = np.stack(predict_list)
# gt_list = np.stack(gt_list)
# import pdb;pdb.set_trace()
# np.save(os.path.join(web_dir,"predict"), predict_list)
# np.save(os.path.join(web_dir,"gt"), gt_list)

# webpage.save()

print('avg_ssim_score: %.4f' % (ssim_score/len(dataset)).item())
print('avg_psnr_score: %.4f' %(psnr_score/len(dataset)))
print('avg_lpips_score: %.4f'%(lpips_score/len(dataset)).item())