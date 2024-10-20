"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from linear import LinearClassifierSingle
from datalist import DataListDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path to root dataset')
parser.add_argument('--test_set', help='test datalist')
parser.add_argument('--threshold', type=float, default=0.5, help='checkpoint ID')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--blocks', type=int, default=4, help='number of blocks')

parser.add_argument('--dim', type=int, default=1536, help='output dim')
# parser.add_argument('--dim', type=int, default=3072, help='output dim')

parser.add_argument('--model', default='dino_vits8', help='model name')
parser.add_argument('--outf', default='checkpoints/cls_concat_4/dinov1_s8', help='folder to output images and model checkpoints')
parser.add_argument('--id', type=int, help='checkpoint ID')

# parser.add_argument('--model', default='dino_vits16', help='model name')
# parser.add_argument('--outf', default='checkpoints/cls_concat_4/dinov1_s16', help='folder to output images and model checkpoints')
# parser.add_argument('--id', type=int, help='checkpoint ID')

# parser.add_argument('--model', default='dino_vitb8', help='model name')
# parser.add_argument('--outf', default='checkpoints/cls_concat_4/dinov1_b8', help='folder to output images and model checkpoints')
# parser.add_argument('--id', type=int, help='checkpoint ID')

# parser.add_argument('--model', default='dino_vitb16', help='model name')
# parser.add_argument('--outf', default='checkpoints/cls_concat_4/dinov1_b16', help='folder to output images and model checkpoints')
# parser.add_argument('--id', type=int, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_test = DataListDataset(data_dir=opt.dataset, data_list=opt.test_set, transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    dino = torch.hub.load('facebookresearch/dino:main', opt.model)
    model = LinearClassifierSingle(dim=opt.dim)
    model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.id) + '.pt'), map_location='cpu'))
    
    dino.eval()
    model.eval()

    if opt.gpu_id >= 0:
        dino.cuda(opt.gpu_id)
        model.cuda(opt.gpu_id)

    tol_label = np.array([], dtype=np.float64)
    tol_pred = np.array([], dtype=np.float64)
    tol_pred_prob = np.array([], dtype=np.float64)

    for img_file, img_data, labels_data in tqdm(dataloader_test):

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float64)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)

        with torch.no_grad():
            output = dino.get_intermediate_layers(img_data, opt.blocks)
            concat = output[0][:,0]
            for i in range(len(output) - 1):
                concat = torch.cat((concat, output[i+1][:,0]), dim=-1)
            classes = model(concat)

        pred_prob = torch.softmax(classes, dim=1).cpu()

        output_pred = np.zeros((pred_prob.shape[0]), dtype=np.float64)

        for i in range(pred_prob.shape[0]):
            if pred_prob[i,1] >= opt.threshold:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))
    
    acc_test = metrics.accuracy_score(tol_label, tol_pred)

    fprs, tprs, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
    
    n_pos = sum(tol_label[tol_label == 1])
    n_neg = sum(1 - tol_label[tol_label == 0])

    n_fp = sum(tol_pred[tol_label==0])
    n_fn = sum(1 - tol_pred[tol_label==1])

    fpr = n_fp * 1.0 / n_neg
    fnr = n_fn * 1.0 / n_pos

    hter = (fpr + fnr) / 2

    print('[Epoch %d]   acc: %.2f   eer: %.2f   hter: %.2f' % (opt.id, acc_test*100, eer*100, hter*100))
    text_writer.write('%d,%.2f,%.2f,%.2f\n' % (opt.id, acc_test*100, eer*100, hter*100))

    thres_eer = interp1d(fprs, thresholds)(eer) 
    print(thres_eer)

    text_writer.flush()
    text_writer.close()
