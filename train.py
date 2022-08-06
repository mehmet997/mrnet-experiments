import shutil
import os
import time
from datetime import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import MRDataset
from models.mrnet import MRNet

from sklearn import metrics
import csv
import utils as ut


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, device, log_every=100):
    """
    Procedure to train a model on the training set
    """
    model.train()

    model = model.to(device)

    y_preds = np.array([])
    y_trues = np.array([])
    losses = []

    for i, (images, label, weight) in enumerate(train_loader):

        images = [image.to(device).float() for image in images]
        # label = label[0]
        label = label.to(device)
        weight = weight.to(device)

        prediction = model(images)

        loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        if y_preds.size == 0:
            y_trues = np.asarray([[int(x) for x in label[0]]])
            y_preds = np.asarray([[x.item() for x in probas[0]]])
        else:
            y_trues = np.vstack((y_trues, [int(x) for x in label[0]]))
            y_preds = np.vstack((y_preds, [x.item() for x in probas[0]]))

        multiclass_auc, acl_auc, men_auc, abn_auc = calculata_auc_values(y_preds, y_trues)

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', multiclass_auc, epoch * len(train_loader) + i)
        writer.add_scalar('Train/ACL_AUC', acl_auc, epoch * len(train_loader) + i)
        writer.add_scalar('Train/MENISCUS_AUC', men_auc, epoch * len(train_loader) + i)
        writer.add_scalar('Train/ABNORMAL_AUC', abn_auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print(
                '''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc_multiclass : {5} | lr : {6}'''.
                    format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(train_loader),
                    np.round(np.mean(losses), 4),
                    np.round(multiclass_auc, 4),
                    current_lr
                )
            )

    writer.add_scalar('Train/AUC_epoch', multiclass_auc, epoch)
    writer.add_scalar('Train/ACL_AUC_epoch', acl_auc, epoch)
    writer.add_scalar('Train/MENISCUS_AUC_epoch', men_auc, epoch)
    writer.add_scalar('Train/ABNORMAL_AUC_epoch', abn_auc, epoch)
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(multiclass_auc, 4)

    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, device, log_every=20):
    """
    Procedure to evaluate a model on the validation set
    """
    model.eval()

    y_preds = np.array([])
    y_trues = np.array([])
    y_class_preds = []
    losses = []

    for i, (images, label, weight) in enumerate(val_loader):

        images = [image.to(device).float() for image in images]
        label = label.to(device)
        weight = weight.to(device)

        prediction = model.forward(images)

        loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weight)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        if y_preds.size == 0:
            y_trues = np.asarray([[int(x) for x in label[0]]])
            y_preds = np.asarray([[x.item() for x in probas[0]]])
            y_class_preds = (probas.cpu()[0] > 0.5).float()
        else:
            y_trues = np.vstack((y_trues, [int(x) for x in label[0]]))
            y_preds = np.vstack((y_preds, [x.item() for x in probas[0]]))
            y_class_preds = np.vstack((y_class_preds, (probas.cpu()[0] > 0.5).float()))

        multiclass_auc, acl_auc, men_auc, abn_auc = calculata_auc_values(y_preds, y_trues)

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', multiclass_auc, epoch * len(val_loader) + i)
        writer.add_scalar('Val/ACL_AUC', acl_auc, epoch * len(val_loader) + i)
        writer.add_scalar('Val/MENISCUS_AUC', men_auc, epoch * len(val_loader) + i)
        writer.add_scalar('Val/ABNORMAL_AUC', abn_auc, epoch * len(val_loader) + i)
        if (i % log_every == 0) & (i > 0):
            print(
                '''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc_multiclass : {5} | lr : {6}'''.
                    format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(val_loader),
                    np.round(np.mean(losses), 4),
                    np.round(multiclass_auc, 4),
                    current_lr
                )
            )

    writer.add_scalar('Val/AUC_epoch', multiclass_auc, epoch)
    writer.add_scalar('Val/ACL_AUC_epoch', acl_auc, epoch)
    writer.add_scalar('Val/MENISCUS_AUC_epoch', men_auc, epoch)
    writer.add_scalar('Val/ABNORMAL_AUC_epoch', abn_auc, epoch)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(multiclass_auc, 4)

    val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_trues, y_class_preds)
    val_accuracy = np.round(val_accuracy, 4)
    val_sensitivity = np.round(val_sensitivity, 4)
    val_specificity = np.round(val_specificity, 4)

    return val_loss_epoch, val_auc_epoch, val_accuracy, val_sensitivity, val_specificity


def calculata_auc_values(y_preds, y_trues):
    try:
        multiclass_auc = metrics.roc_auc_score(y_trues, y_preds, multi_class='ovr', average="macro")
    except:
        multiclass_auc = 0.5
    # acl AUC
    try:

        acl_auc = metrics.roc_auc_score(y_trues[:, 0], y_preds[:, 0])
    except:
        acl_auc = 0.5
    # meniscus AUC
    try:
        men_auc = metrics.roc_auc_score(y_trues[:, 1], y_preds[:, 1])
    except:
        men_auc = 0.5
    # abnormal AUC
    try:
        abn_auc = metrics.roc_auc_score(y_trues[:, 2], y_preds[:, 2])
    except:
        abn_auc = 0.5
    return multiclass_auc, acl_auc, men_auc, abn_auc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create dirs to store experiment checkpoints, logs, and results
    exp_dir_name = args.experiment
    exp_dir = os.path.join('/content/drive/MyDrive/CV_Project/experiments', exp_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, 'models'))
        os.makedirs(os.path.join(exp_dir, 'logs'))
        os.makedirs(os.path.join(exp_dir, 'results'))

    log_root_folder = exp_dir + "/logs/"
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    # create training and validation set
    train_dataset = MRDataset(args.data_path, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                               drop_last=False)

    validation_dataset = MRDataset(args.data_path, train=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=-True, num_workers=2,
                                                    drop_last=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # create the model
    mrnet = MRNet()
    mrnet = mrnet.to(device)

    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.01)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_auc = float(0)
    best_val_accuracy = float(0)
    best_val_sensitivity = float(0)
    best_val_specificity = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    # train and test loop
    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()

        # train
        train_loss, train_auc = train_model(mrnet, train_loader, epoch, num_epochs, optimizer, writer, current_lr,
                                            device, log_every)

        # evaluate
        val_loss, val_auc, val_accuracy, val_sensitivity, val_specificity = evaluate_model(mrnet, validation_loader,
                                                                                           epoch, num_epochs, writer,
                                                                                           current_lr, device)

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_accuracy = val_accuracy
            best_val_sensitivity = val_sensitivity
            best_val_specificity = val_specificity
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}.pth'
                for f in os.listdir(exp_dir + '/models/'):
                    if (args.prefix_name in f):
                        os.remove(exp_dir + f'/models/{f}')
                torch.save(mrnet, exp_dir + f'/models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    # save results to csv file
    with open(os.path.join(exp_dir, 'results', f'model_{args.prefix_name}-results.csv'), 'w') as res_file:
        fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['LOSS', 'AUC-best', 'Accuracy-best', 'Sensitivity-best', 'Specifity-best'])
        fw.writerow([best_val_loss, best_val_auc, best_val_accuracy, best_val_sensitivity, best_val_specificity])
        res_file.close()

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
