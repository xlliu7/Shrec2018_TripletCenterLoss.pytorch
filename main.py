#coding=utf-8
## code on 1032
from __future__ import print_function, absolute_import
import torch
# import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
import torch.optim as optim

import os, shutil 
import argparse
import sys
import time
import numpy as np
import scipy.io as scio

# defined by zczhou
# from models.sketch2shape import sketch_net, shape_net
import misc.custom_loss as custom_loss 

import dataset.sk_dataset as sk_dataset 
import dataset.sh_views_dataset as sh_views_dataset 

import misc.transforms as T
# from misc.utils import Logger
import models

from evaluation import map_and_auc, compute_distance, compute_map

import misc.utils as utils
# from sampler import RandomIdentitySampler

from IPython.core.debugger import Tracer 
debug_here = Tracer() 


def get_data(train_shape_views_folder, test_shape_views_folder, train_shape_flist, test_shape_flist, 
            train_sketch_folder, test_sketch_folder, train_sketch_flist, test_sketch_flist, 
             height, width, batch_size, workers, pk_flag=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        # T.RandomSizedRectCrop(height, width),
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # define sketch dataset 
    extname = '.png' if 'image' not in train_sketch_flist else '.JPEG'
    sketch_train_data = sk_dataset.Sk_Dataset(train_sketch_folder, train_sketch_flist, transform=train_transformer, ext=extname)
    sketch_test_data = sk_dataset.Sk_Dataset(test_sketch_folder, test_sketch_flist, transform=test_transformer, ext=extname)
    
    # define shape views dataset 
    shape_train_data = sh_views_dataset.Sh_Views_Dataset(train_shape_views_folder, train_shape_flist, transform=train_transformer)
    shape_test_data = sh_views_dataset.Sh_Views_Dataset(test_shape_views_folder, test_shape_flist, transform=test_transformer)
   
    # num_classes = sketch_train_data.num_classes

    if pk_flag:
        train_sketch_loader = DataLoader(
            sketch_train_data,
            batch_size=batch_size, num_workers=workers,
            # sampler=RandomIdentitySampler(sketch_train_data.imgs, num_instances),
            pin_memory=True, drop_last=True)

        train_shape_loader = DataLoader(
            shape_train_data,
            batch_size=batch_size, num_workers=workers,
            # sampler=RandomIdentitySampler(shape_train_data.imgs, num_instances),
            pin_memory=True, drop_last=True)
    else:
        train_sketch_loader = DataLoader(
            sketch_train_data,
            batch_size=batch_size*2, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

        train_shape_loader = DataLoader(
            shape_train_data,
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)


    test_sketch_loader = DataLoader(
        sketch_test_data,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_shape_loader = DataLoader(
        shape_test_data,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    # sketch_weight = utils.get_weight(sketch_train_data.imgs)
    #　shape_weight = utils.get_weight(shape_train_data.imgs)
    # cls_weight = sketch_weight / (train_sketch_loader.batch_size*1.0 / train_shape_loader.batch_size) + shape_weight
    # cls_weight = cls_weight / cls_weight.sum() * cls_weight.size
    # cls_weight = torch.Tensor(cls_weight)

    return train_sketch_loader, train_shape_loader, test_sketch_loader, test_shape_loader # , cls_weight


def main(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    opt.checkpoint_folder += '_'+opt.backbone
    if opt.sketch_finetune:
        opt.checkpoint_folder += '_finetune'
    if not os.path.exists(opt.checkpoint_folder):
        os.makedirs(opt.checkpoint_folder)

    print(opt)
    # Redirect print to both console and log file
    # if not opt.evaluate:
    #    sys.stdout = Logger(os.path.join(opt.logs_dir, opt.log_name))

    # Create data loaders
    if opt.height is None or opt.width is None:
        opt.height, opt.width = (224, 224)

    train_sketch_loader, train_shape_loader, test_sketch_loader, test_shape_loader  =  get_data(opt.train_shape_views_folder, 
                            opt.test_shape_views_folder, opt.train_shape_flist, opt.test_shape_flist, 
                            opt.train_sketch_folder, opt.test_sketch_folder, opt.train_sketch_flist, opt.test_sketch_flist, 
                            opt.height, opt.width, opt.batch_size, opt.workers, pk_flag=False)

    # Create model
    #if opt.pool_idx is None:
    #    opt.pool_idx = set_default_pool
    kwargs = {'pool_idx': opt.pool_idx} if opt.pool_idx is not None else {} 
    backbone = eval('models.'+opt.backbone)
    net_bp = backbone.Net_Prev_Pool(**kwargs)
    net_vp = backbone.View_And_Pool()
    net_ap = backbone.Net_After_Pool(**kwargs)
    if opt.sketch_finetune:
        net_whole = backbone.Net_Whole(nclasses = 10, use_finetuned=True)
    else:
        net_whole = backbone.Net_Whole(nclasses = 10)
    # for alexnet or vgg, feat_dim = 4096
    # for resnet, feat_dim = 2048
    net_cls = backbone.Net_Classifier(nclasses = 10)
    # Criterion
    # criterion = nn.CrossEntropyLoss().cuda()
    # if opt.balance: # current no balancing
    #    crt_cls = nn.CrossEntropyLoss().cuda()
    # else:
    # classification loss 
    crt_cls = nn.CrossEntropyLoss().cuda()
    # triplet center loss 
    crt_tlc = custom_loss.TripletCenterLoss(margin=opt.margin).cuda()
    if opt.wn:
        crt_tlc = torch.nn.utils.weight_norm(crt_tlc, name='centers')
    criterion = [crt_cls, crt_tlc, opt.w1, opt.w2]

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if opt.resume:
        checkpoint = torch.load(opt.resume)
        net_bp.load_state_dict(checkpoint['net_bp'])
        net_ap.load_state_dict(checkpoint['net_ap'])
        net_whole.load_state_dict(checkpoint['net_whole'])
        net_cls.load_state_dict(checkpoint['net_cls'])
        crt_tlc.load_state_dict(checkpoint['centers'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_prec']
        # start_epoch = checkpoint['epoch']
        # best_top1 = checkpoint['best_top1']
        # print("=> Start epoch {}  best top1 {:.1%}"
        #      .format(start_epoch, best_top1))
    
    # model = nn.DataParallel(model).cuda()
    net_bp = nn.DataParallel(net_bp).cuda()
    net_vp = net_vp.cuda()
    net_ap = nn.DataParallel(net_ap).cuda()
    net_whole = nn.DataParallel(net_whole).cuda()
    net_cls = nn.DataParallel(net_cls).cuda()
    # wrap multiple models in optimizer 
    optim_shape = optim.SGD([{'params': net_ap.parameters()},
                            {'params': net_bp.parameters(), 'lr':1e-3},
                            {'params': net_cls.parameters()}],
                          lr=0.001, momentum=0.9, weight_decay=opt.weight_decay)

    base_param_ids = set(map(id, net_whole.module.features.parameters()))
    new_params = [p for p in net_whole.parameters() if id(p) not in base_param_ids]
    param_groups = [
    {'params': net_whole.module.features.parameters(), 'lr_mult':0.1},
    {'params':new_params, 'lr_mult':1.0}]

    # optim_sketch = optim.SGD(net_whole.module.parameters(), lr=0.01)
    optim_sketch = optim.SGD(param_groups, lr=0.001, momentum=0.9, weight_decay=opt.weight_decay)
    optim_centers = optim.SGD(crt_tlc.parameters(), lr=0.1)

    optimizer = (optim_sketch, optim_shape, optim_centers)
    model = (net_whole, net_bp, net_vp, net_ap, net_cls)

    # Schedule learning rate
    def adjust_lr(epoch, optimizer):
        step_size = 800 if opt.pk_flag else 80 # 40
        lr = opt.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    top1 = 0.0
    if opt.evaluate:
        # validate and compute mAP
        _, top1 = validate(test_sketch_loader, test_shape_loader, model, criterion, 0, opt)
        exit()
    best_epoch = -1
    best_metric = None
    # total_epochs = opt.max_epochs*10 if opt.pk_flag else opt.max_epochs
    for epoch in range(start_epoch, opt.max_epochs):
        # adjust_lr(epoch, optim_sketch)
        # adjust_lr(epoch, optim_shape)
        # adjust_lr(epoch, optim_centers)
        # cls acc top1
        train_top1 = train(train_sketch_loader, train_shape_loader, model, criterion, optimizer, epoch, opt)
        if epoch < opt.start_save and (epoch % opt.interval == 0):
            continue

        if train_top1 > 0.1:
            print("Test:")
            cur_metric = validate(test_sketch_loader, test_shape_loader, model, criterion, epoch, opt)
            top1 = cur_metric[-1]

        is_best = top1 > best_top1
        if is_best:
            best_epoch = epoch + 1
            best_metric = cur_metric
        best_top1 = max(top1, best_top1)


        
        checkpoint = {} 
        checkpoint['epoch'] = epoch + 1
        checkpoint['current_prec'] = top1
        checkpoint['best_prec'] = best_top1
        checkpoint['net_bp'] = net_bp.module.state_dict() 
        checkpoint['net_ap'] = net_ap.module.state_dict() 
        checkpoint['net_whole'] = net_whole.module.state_dict() 
        checkpoint['net_cls'] = net_cls.module.state_dict() 
        checkpoint['centers'] = crt_tlc.state_dict()
        
        path_checkpoint = '{0}/model_latest.pth'.format(opt.checkpoint_folder)
        utils.save_checkpoint(checkpoint, path_checkpoint)
        
        if is_best: # save checkpoint 
            path_checkpoint = '{0}/model_best.pth'.format(opt.checkpoint_folder)
            utils.save_checkpoint(checkpoint, path_checkpoint)
            if opt.sf:
              shutil.copyfile(opt.checkpoint_folder+'/test_feat_temp.mat', opt.checkpoint_folder+'/test_feat_best.mat')

        print('\n * Finished epoch {:3d}  top1: {:5.3%}  best: {:5.3%}{} @epoch {}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else '', best_epoch))

        print('Best metric', best_metric)

def train(sketch_dataloader, shape_dataloader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    tpl_losses = utils.AverageMeter()

    # training mode
    net_whole, net_bp, net_vp, net_ap, net_cls = model
    optim_sketch, optim_shape, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion

    net_whole.train()
    net_bp.train()
    net_vp.train()
    net_ap.train()
    net_cls.train()

    end = time.time()
    # debug_here() 
    for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(sketch_dataloader, shape_dataloader)):

        shapes = shapes.view(shapes.size(0)*shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))

        # expanding: (bz * 12) x 3 x 224 x 224
        shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())


        o_bp = net_bp(shapes_v)
        o_vp = net_vp(o_bp)
        shape_feat = net_ap(o_vp)
        sketch_feat = net_whole(sketches_v)
        feat = torch.cat([shape_feat, sketch_feat])
        target = torch.cat([p_labels_v, k_labels_v])
        score = net_cls(feat) 
        
        cls_loss = crt_cls(score, target)
        tpl_loss, _ = crt_tlc(score, target)
        # tpl_loss, _ = crt_tlc(feat, target)

        loss = w1 * cls_loss + w2 * tpl_loss

        ## measure accuracy
        prec1 = utils.accuracy(score.data, target.data, topk=(1,))[0]
        losses.update(cls_loss.data[0], score.size(0)) # batchsize
        tpl_losses.update(tpl_loss.data[0], score.size(0))
        top1.update(prec1[0], score.size(0))

        ## backward
        optim_sketch.zero_grad()
        optim_shape.zero_grad()
        optim_centers.zero_grad()

        loss.backward()
        utils.clip_gradient(optim_sketch, opt.gradient_clip)
        utils.clip_gradient(optim_shape, opt.gradient_clip)
        utils.clip_gradient(optim_centers, opt.gradient_clip)
        
        optim_sketch.step()
        optim_shape.step()
        optim_centers.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Trploss {triplet.val:.4f}({triplet.avg:.3f})'.format(
                epoch, i, len(sketch_dataloader), batch_time=batch_time,
                loss=losses, top1=top1, triplet=tpl_losses))
            # print('triplet loss: ', tpl_center_loss.data[0])
    print(' * Train Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def validate(sketch_dataloader, shape_dataloader, model, criterion, epoch, opt):

    """
    test for one epoch on the testing set
    """
    sketch_losses = utils.AverageMeter()
    sketch_top1 = utils.AverageMeter()

    shape_losses = utils.AverageMeter()
    shape_top1 = utils.AverageMeter()

    net_whole, net_bp, net_vp, net_ap, net_cls = model
    # optim_sketch, optim_shape, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion

    net_whole.eval()
    net_bp.eval()
    net_vp.eval()
    net_ap.eval()
    net_cls.eval()

    sketch_features = []
    sketch_scores = []
    sketch_labels = []

    shape_features = []
    shape_scores = []
    shape_labels = []

    batch_time = utils.AverageMeter()
    end = time.time()

    for i, (sketches, k_labels) in enumerate(sketch_dataloader):
        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())
        sketch_feat = net_whole(sketches_v)
        sketch_score = net_cls(sketch_feat)

        loss = crt_cls(sketch_score, k_labels_v)

        prec1 = utils.accuracy(sketch_score.data, k_labels_v.data, topk=(1,))[0]
        sketch_losses.update(loss.data[0], sketch_score.size(0)) # batchsize
        sketch_top1.update(prec1[0], sketch_score.size(0))
        sketch_features.append(sketch_feat.data.cpu())
        sketch_labels.append(k_labels)
        sketch_scores.append(sketch_score.data.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(sketch_dataloader), batch_time=batch_time, loss=sketch_losses,
                      top1=sketch_top1))
    print(' *Sketch Prec@1 {top1.avg:.3f}'.format(top1=sketch_top1))

    batch_time = utils.AverageMeter()
    end = time.time()
    for i, (shapes, p_labels) in enumerate(shape_dataloader):
        shapes = shapes.view(shapes.size(0)*shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
        # expanding: (bz * 12) x 3 x 224 x 224
        shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        o_bp = net_bp(shapes_v)
        o_vp = net_vp(o_bp)
        shape_feat = net_ap(o_vp)
        shape_score = net_cls(shape_feat)

        loss = crt_cls(shape_score, p_labels_v)

        prec1 = utils.accuracy(shape_score.data, p_labels_v.data, topk=(1,))[0]
        shape_losses.update(loss.data[0], shape_score.size(0)) # batchsize
        shape_top1.update(prec1[0], shape_score.size(0))
        shape_features.append(shape_feat.data.cpu())
        shape_labels.append(p_labels)
        shape_scores.append(shape_score.data.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(shape_dataloader), batch_time=batch_time, loss=shape_losses,
                      top1=shape_top1))
    print(' *Shape Prec@1 {top1.avg:.3f}'.format(top1=shape_top1))

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()

    shape_scores = torch.cat(shape_scores, 0).numpy()
    sketch_scores = torch.cat(sketch_scores, 0).numpy()

    shape_labels = torch.cat(shape_labels, 0).numpy()
    sketch_labels = torch.cat(sketch_labels, 0).numpy()

    # d = compute_distance(sketch_features.copy(), shape_features.copy(), l2=False)
    # scio.savemat('test/example.mat',{'d':d, 'feat':dataset_features, 'labels':dataset_labels})
    # AUC, mAP = map_and_auc(sketch_labels.copy(), shape_labels.copy(), d)
    # print(' * Feature AUC {0:.5}   mAP {0:.5}'.format(AUC, mAP))

    d_feat = compute_distance(sketch_features.copy(), shape_features.copy(), l2=False)
    d_feat_norm = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    mAP_feat = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat)
    mAP_feat_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat_norm)
    print(' * Feature mAP {0:.5%}\tNorm Feature mAP {1:.5%}'.format(mAP_feat, mAP_feat_norm))


    d_score = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=False)
    mAP_score = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score)
    d_score_norm = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=True)
    mAP_score_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score_norm)
    if opt.sf:
        shape_paths = [img[0] for img in shape_dataloader.dataset.shape_target_path_list]
        sketch_paths = [img[0] for img in sketch_dataloader.dataset.sketch_target_path_list]
        scio.savemat('{}/test_feat_temp.mat'.format(opt.checkpoint_folder), {'score_dist':d_score, 'score_dist_norm': d_score_norm, 'feat_dist': d_feat, 'feat_dist_norm': d_feat_norm,'sketch_features':sketch_features, 'sketch_labels':sketch_labels, 'sketch_scores': sketch_scores,
        'shape_features':shape_features, 'shape_labels':shape_labels, 'sketch_paths':sketch_paths, 'shape_paths':shape_paths})
    print(' * Score mAP {0:.5%}\tNorm Score mAP {1:.5%}'.format(mAP_score, mAP_score_norm))
    return [sketch_top1.avg, shape_top1.avg, mAP_feat, mAP_feat_norm, mAP_score, mAP_score_norm]


if __name__ == '__main__':
    from options import get_arguments

    opt = get_arguments()
    main(opt)
