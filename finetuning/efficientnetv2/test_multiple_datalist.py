"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen

This script is used to test the model on multiple datalists.
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from model import EfficientNetv2
from datalist import DataListDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path to root dataset')
parser.add_argument('--data_list', help='path to datalists folder')
parser.add_argument('--threshold', type=float, default=0.5, help='checkpoint ID')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=384, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

parser.add_argument('--model', default='tf_efficientnetv2_l.in21k_ft_in1k', help='model name')
parser.add_argument('--dim', type=int, default=1280, help='number of blocks')
parser.add_argument('--num_unfrozen_blocks', type=int, default=1, help='number of last blocks to train')
parser.add_argument('--outf', help='path to checkpoints folder')
parser.add_argument('--id', type=int, help='checkpoint ID')


opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test_detail.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    model = EfficientNetv2(model_name=opt.model, num_unfrozen_blocks=opt.num_unfrozen_blocks, hidden_dim=opt.dim)
    model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.id) + '.pt'), map_location='cpu'))
    model.eval()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if opt.gpu_id >= 0:
        model.cuda(opt.gpu_id)

    test_list = ['data_list_1.txt', 'data_list_2.txt', 'data_list_3.txt']

    glo_label = np.array([], dtype=np.float64)
    glo_label_pos = []
    glo_label_neg = []

    glo_pred = np.array([], dtype=np.float64)
    glo_pred_pos = []
    glo_pred_neg = []
    glo_pred_prob = np.array([], dtype=np.float64)

    res_stack = []

    print('Checkpoint ID: ', opt.id)
    print('Threshold: ', opt.threshold)
    text_writer.write('Checkpoint ID: %d\n' % opt.id)
    text_writer.write('Threshold: %.4f\n' % opt.threshold)

    for dt_set in test_list:
        print('Dataset: ', dt_set)
        text_writer.write('Dataset: %s\n' % dt_set)

        dataset_test = DataListDataset(data_dir=opt.dataset, data_list=os.path.join(opt.data_list, dt_set), transform=transform_fwd)
        assert dataset_test
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

        tol_label = np.array([], dtype=np.float64)
        tol_pred = np.array([], dtype=np.float64)
        tol_pred_prob = np.array([], dtype=np.float64)

        for img_file, img_data, labels_data in tqdm(dataloader_test):
   
            img_label = labels_data.numpy().astype(np.float64)

            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            with torch.no_grad():
                classes = model(img_data)

            pred_prob = torch.softmax(classes, dim=1).cpu()

            output_pred = np.zeros((pred_prob.shape[0]), dtype=np.float64)

            for i in range(pred_prob.shape[0]):
                if pred_prob[i,1] >= opt.threshold:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

                if labels_data[i] == 1:
                    glo_label_pos.append(img_label[i])
                    glo_pred_pos.append(output_pred[i])
                else:
                    glo_label_neg.append(img_label[i])
                    glo_pred_neg.append(output_pred[i])

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))
            tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

            glo_label = np.concatenate((glo_label, img_label))
            glo_pred = np.concatenate((glo_pred, output_pred))
            glo_pred_prob = np.concatenate((glo_pred_prob, pred_prob[:,1].data.numpy()))
        
        acc_test = metrics.accuracy_score(tol_label, tol_pred)


        print('Accuracy: ', (acc_test*100))
        text_writer.write('Accuracy: %.2f\n' % (acc_test*100))

        res_stack.append(acc_test*100)
    
    text_writer.write('Summary:\n')


    acc_test = metrics.accuracy_score(glo_label, glo_pred)
    acc_pos = metrics.accuracy_score(np.array(glo_label_pos), np.array(glo_pred_pos))
    acc_neg = metrics.accuracy_score(np.array(glo_label_neg), np.array(glo_pred_neg))

    fprs, tprs, thresholds = roc_curve(glo_label, glo_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
    
    n_pos = sum(glo_label[glo_label == 1])
    n_neg = sum(1 - glo_label[glo_label == 0])

    n_fp = sum(glo_pred[glo_label==0])
    n_fn = sum(1 - glo_pred[glo_label==1])

    fpr = n_fp * 1.0 / n_neg
    fnr = n_fn * 1.0 / n_pos

    hter = (fpr + fnr) / 2

    print('Acc: %.2f   Acc_pos: %.2f   Acc_neg: %.2f   EER: %.2f   HTER: %.2f' % (acc_test*100, acc_pos*100, acc_neg*100, eer*100, hter*100))
    text_writer.write('Acc: %.2f   Acc_pos: %.2f   Acc_neg: %.2f   EER: %.2f   HTER: %.2f\n' % (acc_test*100, acc_pos*100, acc_neg*100, eer*100, hter*100))

    thres_eer = interp1d(fprs, thresholds)(eer) 
    print('Thres. based on current EER: %.4f' % thres_eer)
    text_writer.write('Thres. based on current EER: %.4f\n' % (thres_eer))

    res_stack.append(acc_test*100)
    res_stack.append(acc_neg*100)
    res_stack.append(acc_pos*100)
    res_stack.append(eer*100)
    res_stack.append(hter*100)

    for item in res_stack:
        text_writer.write('%.2f\t' % item)
