import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import timm

import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from eval_helper import log_metrics, performances


mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(size):
    datatrans =  transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                            std=std_train),])
    return datatrans

def gt_transforms(size):
    gttrans =  transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),])
    return gttrans


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_map_one_to_all(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    
    
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        test_size = fs.shape
        pred_mask = torch.zeros(1, test_size[2], test_size[3], dtype=torch.float32)

        for i in range(test_size[2]):
            for j in range(test_size[3]):
                a_map_p = F.cosine_similarity(fs[:,:,i,j].unsqueeze(-1), ft[0, :, :,:].reshape(1, test_size[1], -1), dim=1)
                pred_mask[0, i, j] = 1 - torch.max(a_map_p)
        
        a_map = torch.unsqueeze(pred_mask, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def get_dino_attentions(model, img, patch_size=8, img_size=256, threshold=None):
    w_featmap = img_size // patch_size
    h_featmap = img_size // patch_size
    attentions = model.get_last_selfattention(img.cuda())

    nh = attentions.shape[1] # number of head
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

    return [attentions.unsqueeze(0)]


def run_amap_extraction(input_img, output_img, im_name, root, mid_path, dino_version='v1s8',use_dino=True, save_visuals=True, fg_mask=None):
    # feature extraction for input image
    if use_dino:
        input_img_dino = input_img.clone()
        if 'r50' in dino_version:
            # DINO R50 
            input_features = pretrained_model.forward(input_img_dino.cuda())
            input_features = [v for k, v in input_features.items()]
            input_features = input_features[1:4]
        elif 'v1' in dino_version:
            # patch features
            input_features = [pretrained_model.get_intermediate_layers(input_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, 1:, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
        elif 'v2' in dino_version:
            # Dinov2
            input_features = [pretrained_model.get_intermediate_layers(input_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, :, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
            # attention maps, not working very well
            # input_features = get_dino_attentions(pretrained_model, input_img_dino)
    else:
        # RESNET-50
        input_features = pretrained_model(input_img.cuda())

    # feature extraction for reconstructed image
    if use_dino:
        output_img_dino = output_img.clone()
        if 'r50' in dino_version:
            # DINO R50 
            output_features = pretrained_model.forward(output_img_dino.cuda())
            output_features = [v for k, v in output_features.items()]
            output_features = output_features[1:4]
        elif 'v1' in dino_version:
            # patch features
            output_features = [pretrained_model.get_intermediate_layers(output_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, 1:, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
        elif 'v2' in dino_version:
            # Dinov2 patch
            output_features = [pretrained_model.get_intermediate_layers(output_img_dino.resize_(1, 3, img_size, img_size).cuda())[0][:, :, :].reshape(1, int(img_size/patch_size), int(img_size/patch_size), -1).permute(0, 3, 1, 2)]
            # attention maps, not working well
            # output_features = get_dino_attentions(pretrained_model, output_img_dino)
    else:
        output_features = pretrained_model(output_img.cuda())

    if not use_dino:
        input_features = input_features[1:4]
        output_features = output_features[1:4]

    # Calculate the anomaly score
    sigma = 5
    # anomaly_map, _ = cal_anomaly_map_one_to_all(input_features, output_features, input_img.shape[-1], amap_mode='a')
    anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a')
    if fg_mask is not None:
        anomaly_map *= fg_mask
    anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)
    anomaly_map = torch.from_numpy(anomaly_map)
    anomaly_map_prediction = anomaly_map.unsqueeze(dim=0).unsqueeze(dim=1)

    filename_feature = "{}-features.jpg".format(im_name)
    path_feature = os.path.join(root, mid_path, filename_feature)
    os.makedirs(os.path.dirname(path_feature), exist_ok=True)
    pred_feature = anomaly_map_prediction.squeeze().detach().cpu().numpy()
    pred_feature = (pred_feature * 255).astype("uint8")
    pred_feature = Image.fromarray(pred_feature, mode='L')
    if save_visuals:
        pred_feature.save(path_feature)

    #Heatmap
    anomaly_map_new = np.round(255 * (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()))
    anomaly_map_new = anomaly_map_new.cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_new, colormap=cv2.COLORMAP_JET)
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    pixel_mean = torch.tensor(pixel_mean).unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
    pixel_std = torch.tensor(pixel_std).unsqueeze(1).unsqueeze(1)
    image = (input_img.squeeze() * pixel_std + pixel_mean) * 255
    image = image.permute(1, 2, 0).to('cpu').numpy().astype('uint8')
    image_copy = image.copy()
    out_heat_map = cv2.addWeighted(heatmap, 0.5, image_copy, 0.5, 0, image_copy)
    heatmap_name = "{}-heatmap.png".format(im_name)
    if save_visuals:
        cv2.imwrite(os.path.join(root, mid_path, heatmap_name), out_heat_map)

    return anomaly_map_prediction

parser = argparse.ArgumentParser(description="DIVAD Evaluation")
parser.add_argument("--data_set", default='visa', help="choices are btad|mtdd|mvtec|visa")
parser.add_argument("--data_path", default='/mnt/isilon/shicsonmez/ad/data/visa_dataset_processed')
parser.add_argument('--use_dino', action='store_false', help='use DINO as segmenter or not!')
parser.add_argument("--dino_version", default='v1s8', help="choices are v1s8|v1s16|v1b8|v1b16|v2s14|v1r50")
parser.add_argument('--save_visuals', action='store_false', help='save predictions visuals such as heatmaps, anomaly maps etc.')
parser.add_argument("--exp_name", default='results_test/{}_ddim_inversion_results_sd{}_ss{}_is{}_nis{}_dinov1_vits8_img_size256_sigma5_no_fg_mask', 
                    help='npz files will be saved inside this folder. make it unique please!')
parser.add_argument("--input_dir", default='/mnt/isilon/shicsonmez/ad/repos/Semantic-SAM/{}_ddim_inversion_results_sd{}_ss{}_is{}_nis{}',
                     help='read the ddim inversion results here')
parser.add_argument("--object_name", default='', help='which object to test')
parser.add_argument("--mask_dir", default='/mnt/isilon/shicsonmez/ad/repos/CutLER/{}_cutler_conf_010', 
                    help='read the detected foreground masks here.')

parser.add_argument("--nis", default=50, type=int, help='num_inference_steps')
parser.add_argument("--inf_step", default=100, type=int, help='inference_steps')
parser.add_argument("--ss", default=40, type=int, help='start_step')
parser.add_argument("--sd_version", default="21", type=str, help='stable diff version, 15 or 21')

args = parser.parse_args()

# DDIM params
num_inference_steps = args.nis
inference_steps = args.inf_step
# The reason we want to be able to specify start step
start_step = args.ss
sd_version = args.sd_version

# Configs
data_set = args.data_set
exp_name = args.exp_name
data_path = args.data_path
object_name = args.object_name
save_visuals = args.save_visuals
input_dir = args.input_dir
fg_mask_dir = args.mask_dir

input_dir = input_dir.format(data_set, sd_version, start_step, inference_steps, num_inference_steps)
exp_name = exp_name.format(data_set, sd_version, start_step, inference_steps, num_inference_steps)
fg_mask_dir = fg_mask_dir.format(data_set)
fg_mask_dir = ""

use_dino = args.use_dino
run_count = 1  # for batch run put a bigger number like 20
dino_version = args.dino_version

print(args)

if use_dino:
    if dino_version == 'v2s14':
        # DINOv2
        # pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        patch_size = 14
        img_size = 392 # 224
    # DINOv1
    elif dino_version == 'v1s8':
        pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')  # best performant!!
        patch_size = 8
        img_size = 256
    elif dino_version == 'v1s16':
        pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')  # best performant!!
        patch_size = 16
        img_size = 256
    elif dino_version == 'v1b8':
        pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        patch_size = 8
        img_size = 256
    elif dino_version == 'v1b16':
        pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')  # best performant!!
        patch_size = 16
        img_size = 256
    elif dino_version == 'v1r50':
        # DINO R50!
        from torchvision.models.feature_extraction import create_feature_extractor
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        pretrained_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        pretrained_model = create_feature_extractor(pretrained_model, return_nodes=return_nodes)
        patch_size = 8
        img_size = 256

    pretrained_model = pretrained_model.cuda()
    pretrained_model.eval()
else:  # imagenet R50 model
    img_size = 224
    pretrained_model = timm.create_model("resnet50", pretrained=True, features_only=True)
    pretrained_model = pretrained_model.cuda()
    pretrained_model.eval()


image_size = (img_size, img_size)
transform = data_transforms(image_size)
target_transform = gt_transforms(image_size)

fileinfos, preds, masks = [], [], []

all_objects = os.listdir(data_path)
for ind, object_name in enumerate(all_objects):
    if not os.path.isdir(os.path.join(data_path, object_name)):
        continue
    subfolders = os.listdir(os.path.join(data_path, object_name, 'test'))

    for sf in subfolders:
        normal_set = False
        if sf in ['good', 'ok']:
            # normal, non-anomaly images 
            normal_set = True

        test_ims = os.listdir(os.path.join(data_path, object_name, 'test', sf))
        
        inter_path = os.path.join(data_path, object_name)

        # run the test for subset images
        for i, im_name in tqdm(enumerate(test_ims)):

            input_img = Image.open(os.path.join(inter_path, 'test', sf, im_name)).convert('RGB')
            output_img = Image.open(os.path.join(input_dir, object_name, sf, im_name + "_sampled.png")).convert('RGB')
            if normal_set:
                mask = np.zeros(image_size).astype(np.uint8)
                mask = Image.fromarray(mask, "L")
            else:
                if data_set == 'mvtec':
                    mask = Image.open(os.path.join(inter_path, 'ground_truth', sf, im_name[:3]+'_mask.png')).convert('L')
                elif data_set == 'visa':
                    mask = Image.open(os.path.join(inter_path, 'ground_truth', sf, im_name[:3]+'.png')).convert('L')
                elif data_set == 'btad':
                    try:
                        mask = Image.open(os.path.join(inter_path, 'ground_truth', sf, '0'+im_name[:3]+'.png')).convert('L')
                    except:
                        mask = Image.open(os.path.join(inter_path, 'ground_truth', sf, '0'+im_name[:3]+'.bmp')).convert('L')
                elif data_set == 'mpdd':
                    mask = Image.open(os.path.join(inter_path, 'ground_truth', sf, im_name[:3]+'_mask.png')).convert('L')

            if fg_mask_dir:
                # fg_mask_temp = np.zeros(input_img.size, dtype=bool)
                fg_mask = np.load(os.path.join(fg_mask_dir, object_name, sf, im_name + "_cutler.png.npz"))['arr_0']
                if len(fg_mask.shape) == 3:
                    fg_mask = np.sum(fg_mask, axis=0, dtype=bool)
                fg_mask = np.asanyarray(Image.fromarray(fg_mask).resize(image_size))
            else:
                fg_mask = None
            
            if data_set == 'mvtec' and object_name in ['carpet', 'leather', 'wood', 'tile', 'zipper']:
                fg_mask = None
            
            input_img = transform(input_img)
            mask = target_transform(mask)
            output_img = transform(output_img)

            pred = run_amap_extraction(input_img, output_img, sf+'_'+im_name, exp_name, object_name, dino_version, use_dino, save_visuals, fg_mask)
            pred_2d = pred.squeeze_(0).squeeze_(0)
            preds.append(pred_2d)
            masks.append(mask.squeeze_(0).squeeze_(0))
            fileinfos.append({"filename": sf+'_'+im_name, "clsname": object_name})
        
# calculate the metrics in the end for all classes
evl_metrics = {'auc': [ {'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'}, {'name': 'f1px'}, {'name': 'f1sp'}]}
print("Gathering final results ...")
ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
log_metrics(ret_metrics, evl_metrics)
