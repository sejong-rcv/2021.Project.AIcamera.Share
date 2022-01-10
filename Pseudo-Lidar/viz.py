import torch 
import networks
import os
import torch
import cv2
import scipy.io
from matplotlib import cm
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
load_weights_folder="tmp/Kaist_stereo_resnet_scale_thermal/models/weights_14"
encoder_path = os.path.join(load_weights_folder, "encoder.pth")
decoder_path = os.path.join(load_weights_folder, "depth.pth")

encoder_dict = torch.load(encoder_path)
model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
depth_decoder.load_state_dict(torch.load(decoder_path))
encoder.cuda()
encoder.eval()
depth_decoder.cuda()
depth_decoder.eval()
txt=open("splits/kaist/test_files.txt","r")
img_list=[]
gt_list=[]
import os 
for i in txt:
    sp=i.split()
    img_list.append(os.path.join("MTN_data/",sp[0],"THERMAL/THER_%09d.jpg"%int(sp[1])))
txt.close()
data_path="MTN_data"
depthtxt=open(data_path+"/txt/test_depth.txt","r")
depthdata=[]
for line in depthtxt:
    depthdata.append(os.path.join(data_path,line[:-1]))
depthtxt.close()
from PIL import Image
from torchvision.transforms import ToTensor,Resize,ToPILImage
tp=ToPILImage()
tt=ToTensor()
rs=Resize((448,512))
def viz_pre(feature):
    k = feature.shape[-1]  

    feature_image = feature.sum(1).squeeze()#.transpose(0,1)
    return feature_image.numpy()
import matplotlib.pyplot as plt

root="THERMAL_predict"
if not os.path.exists(root):
    os.makedirs(root)
def disp2depth( disp, max_disp ):
    max_depth = 50;
    min_depth = 1;
    min_disp  = 1;
    disp[ disp < min_disp ] = min_disp;
    depth = (3233.93339530 * 0.245) / disp;
    depth[depth < min_depth] = min_depth;
    depth[depth > max_depth] = max_depth;

    return depth                      
import numpy as np
from tqdm import tqdm
import cv2
cmap = plt.get_cmap('plasma');
cval = [[cmap(x)[0],cmap(x)[1],cmap(x)[2]] for x in range(0,255)];
cval.append( cval[-1] );
cval = np.array( cval );
def colormap(depth):
    depth = depth * (255./80);
    depth = cval[(depth).astype(np.uint8)];
    depth = (depth*255.).astype(np.uint8);
    depth = depth[:,:,[2,1,0]];
    return depth
for idx,i in enumerate(tqdm(img_list)):
#     print(i)
    img=tt(rs(Image.open(i).convert("RGB"))).unsqueeze(0).cuda()
    with torch.no_grad():
        inv_depth=depth_decoder(encoder(img))[("disp",0)]
#     inv_depth     = inv_depth *1280
#     inv_depth,_    =disp_to_depth(disp,0.01, 50)
    folder=depthdata[idx]
    Depth=scipy.io.loadmat(folder)["depth"]
    inv_depth=1/(inv_depth)
    img_=viz_pre(inv_depth.cpu()) 
    thermal_=viz_pre(img.cpu())*255
#     Depth_=viz_pre(Depth)
#     import pdb;pdb.set_trace()
    img_=np.hstack((Depth,img_))
    concat=colormap(img_)
    img_=np.hstack((img[0].permute((1,2,0)).cpu().numpy() * 255.,concat))
    cv2.imwrite(os.path.join(root,"%09d.png"%idx),img_)
#     import pdb;pdb.set_trace()
#     plt.imsave(os.path.join(root,"%09d.png"%idx),img_,cmap="plasma")
    
save_folder=root
pathIn=root
pathOut = save_folder+'.mp4'
fps = 10
frame_array = []

for idx , path in enumerate(os.listdir(pathIn)) :
    if path[0]==".":
        continue
    if "npy" in path:
        continue
    img = cv2.imread(os.path.join(pathIn,path))
    try:
        height, width, layers = img.shape
    except:
        import pdb;pdb.set_trace()
    size = (width,height)
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
print("Done")
print('Finished Testing')