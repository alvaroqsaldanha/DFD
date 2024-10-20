"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import os
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from linear import LinearClassifierWSAll
from datalist import DataListDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path to root dataset')
parser.add_argument('--train_set', help='train datalist')
parser.add_argument('--val_set', help='validation datalist')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--blocks', type=int, default=4, help='number of blocks')

parser.add_argument('--dim', type=int, default=301440, help='output dim')
# parser.add_argument('--dim', type=int, default=75648, help='output dim')
# parser.add_argument('--dim', type=int, default=602880, help='output dim')
# parser.add_argument('--dim', type=int, default=151296, help='output dim')

parser.add_argument('--model', default='dino_vits8', help='model name')
parser.add_argument('--outf', default='checkpoints/all_ws_4/dinov1_s8', help='folder to output model checkpoints')

# parser.add_argument('--model', default='dino_vits16', help='model name')
# parser.add_argument('--outf', default='checkpoints/all_ws_4/dinov1_s16', help='folder to output model checkpoints')

# parser.add_argument('--model', default='dino_vitb8', help='model name')
# parser.add_argument('--outf', default='checkpoints/all_ws_4/dinov1_b8', help='folder to output model checkpoints')

# parser.add_argument('--model', default='dino_vitb16', help='model name')
# parser.add_argument('--outf', default='checkpoints/all_ws_4/dinov1_b16', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)


if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')


    dino = torch.hub.load('facebookresearch/dino:main', opt.model)
    model = LinearClassifierWSAll(dim=opt.dim, num_blocks=opt.blocks)
    network_loss = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=opt.lr)

    if opt.resume > 0:
        model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.resume) + '.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    model.train(mode=True)

    if opt.gpu_id >= 0:
        dino.cuda(opt.gpu_id)
        model.cuda(opt.gpu_id)
        network_loss.cuda(opt.gpu_id)

    transforms = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_train = DataListDataset(data_dir=opt.dataset, data_list=opt.train_set, transform=transforms)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = DataListDataset(data_dir=opt.dataset, data_list=opt.val_set, transform=transforms)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=int(opt.batchSize/8), shuffle=False, num_workers=int(opt.workers))

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float64)
        tol_pred = np.array([], dtype=np.float64)

        for img_file, img_data, labels_data in tqdm(dataloader_train):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float64)

            optimizer.zero_grad()
            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            with torch.no_grad():
                output = dino.get_intermediate_layers(img_data, opt.blocks)
            classes = model(output)

            loss_dis = network_loss(classes, labels_data.data)
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float64)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(model.state_dict(), os.path.join(opt.outf, 'model_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        model.eval()

        tol_label = np.array([], dtype=np.float64)
        tol_pred = np.array([], dtype=np.float64)
        tol_pred_prob = np.array([], dtype=np.float64)

        count = 0

        for img_file, img_data, labels_data in tqdm(dataloader_val):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float64)

            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            with torch.no_grad():
                output = dino.get_intermediate_layers(img_data, opt.blocks)
                classes = model(output)

            loss_dis = network_loss(classes, labels_data.data)
            loss_dis_data = loss_dis.item()
            
            output_dis = classes.data.cpu()
            pred_prob = torch.softmax(output_dis, dim=1)

            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float64)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))
            tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].numpy()))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        fprs, tprs, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)

        n_pos = sum(tol_label[tol_label == 1])
        n_neg = sum(1 - tol_label[tol_label == 0])

        n_fp = sum(tol_pred[tol_label==0])
        n_fn = sum(1 - tol_pred[tol_label==1])

        fpr = n_fp * 1.0 / n_neg
        fnr = n_fn * 1.0 / n_pos

        hter = (fpr + fnr) / 2

        print('[Epoch %d] Train loss: %.4f  acc: %.2f | Test loss: %.4f  acc: %.2f  eer: %.2f  hter: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100, eer*100, hter*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f,%.2f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100, eer*100, hter*100))

        text_writer.flush()

        model.train(mode=True)

    text_writer.close()
