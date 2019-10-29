import torch 
import torch.nn as nn 
import numpy as np
import os
import time
from torch.autograd import Variable
from IPython.core.debugger import Tracer
debug_here = Tracer() 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n # val*n: how many samples predicted correctly among the n samples 
        self.count += n     # totoal samples has been through 
        self.avg = self.sum / self.count 

#################################################
## confusion matrix 
#################################################
class ConfusionMatrix(object):
    def __init__(self, K): # K is number of classes 
        self.reset(K) 
    def reset(self, K):
        self.num_classes = K 
        # declare a table matrix and zero it 
        self.cm = torch.zeros(K, K) # one row for each class, column is predicted class 
        # self.valids
        self.valids = torch.zeros(K) 
        # mean average precision, i.e., mean class accuracy  
        self.mean_class_acc = 0 

    def batchAdd(self, outputs, targets): 
        """
        output is predicetd probability 
        """
        _, preds = outputs.topk(1, 1, True, True)
        # convert cudalong tensor to long tensor 
        # preds:  bz x 1 
        for m in range(preds.size(0)):
            self.cm[targets[m]][preds[m][0]] = self.cm[targets[m]][preds[m][0]] + 1 


    def updateValids(self):
        # total = 0 
        for t in range(self.num_classes):
            if self.cm.select(0, t).sum() != 0: # column  
                # sum of t-th row is the number of samples coresponding to this class (groundtruth)
                self.valids[t] = self.cm[t][t] / self.cm.select(0, t).sum()
            else:
                self.valids[t] = 0 

        self.mean_class_acc = self.valids.mean() 


#################################################
## compute accuracy 
#################################################
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # top k 
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(model, output_path):    

    ## if not os.path.exists(output_dir):
    ##    os.makedirs("model/")        
    torch.save(model, output_path)
        
    print("Checkpoint saved to {}".format(output_path))


# do gradient clip 
def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)

def preprocess(inputs_12v, mean, std, data_augment):
    """
    inputs_12v: (bz * 12) x 3 x 224 x 224 
    """
    # to tensor 
    if isinstance(inputs_12v, torch.ByteTensor):
        inputs_12v = inputs_12v.float() 

    inputs_12v.sub_(mean).div_(std)

    if data_augment: 
        print('currently not support data augmentation')

    return inputs_12v


# centers: 40(or 55) x 3 x 4096
# features: bz * 4096 -> bz * 3 * 4096 
# compute distance features between each features and centers 
def get_center_loss(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)
    num_centers = centers.size(1)
    # bz x 3 x 4096
    features_view = features.unsqueeze(1).expand(batch_size, num_centers, features_dim)

    target_expand = target.view(batch_size,1, 1).expand(batch_size,num_centers, features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features_view,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features_view # bz x 3 x 4096 
    
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1,1).expand(batch_size,num_centers, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())
        
    return center_loss, centers

def get_centers_loss_margin(centers, features, target, alpha, num_classes, margin=1.0):
    batch_size = target.size(0)
    features_dim = features.size(1)
    # bz x 3 x 4096
    features_view = features.unsqueeze(1).expand(batch_size, 3, features_dim)

    target_expand = target.view(batch_size,1, 1).expand(batch_size,3, features_dim)
    centers_var = Variable(centers, requires_grad=False)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()

    center_loss = criterion(features_view,  centers_batch)
    centers_var.requires_grad=True

    # compute gradient w.r.t. center
    diff = centers_batch - features_view # bz x 3 x 4096 
    
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1,1).expand(batch_size,3, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 

    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

        dist_c1c2 = torch.norm(centers_var[target.data[i]][0] - centers_var[target.data[i]][1], 2) 
        dist_hinge1 = torch.clamp(-dist_c1c2  + margin, min=0.0)
        dist_c2c3 = torch.norm(centers_var[target.data[i]][1] - centers_var[target.data[i]][2], 2) 
        dist_hinge2 = torch.clamp(-dist_c2c3  + margin, min=0.0)
        dist_c1c3 = torch.norm(centers_var[target.data[i]][0] - centers_var[target.data[i]][2], 2)
        dist_hinge3 = torch.clamp(-dist_c1c3  + margin, min=0.0)
        loss = dist_hinge1 + dist_hinge2 + dist_hinge3
        loss.backward() 
        centers[target.data[i]] -= centers_var.grad.data[target.data[i]]

    centers_var.requires_grad=False

    return center_loss, centers

###########################################################
## start of centers_loss margin 
###########################################################
def similarity_matrix(feats_bzxD):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(feats_bzxD, feats_bzxD.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    # D[i, j]: similarity of sample i-th feature feats_bzxD[i] 
    # in the batch and feats_bzxD[j]
    D = diag + diag.t() - 2*r + 1e-6
    return D.sqrt() # no square in the orignal paper
def convert_y2(y):
    bz = y.size(0) # batch size (number of samples in the batch) 
    y_expand = y.unsqueeze(0).expand(bz, bz)
    Y = y_expand.eq(y_expand.t())
    return Y

###########################################
##
###########################################
def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod) 
    res = (norm + norm.t() - 2 * prod).clamp(min = 0) 
    return res if squared else (res + eps).sqrt() + eps 
def pdist2(A, B, squared=False, eps=1e-4):
    """
    input: 
        A: bz x D
        B: bz x D
    output: 
        C: bz x bz  
    """
    m = A.size(0)
    mmp1 = torch.stack([A]*m)
    mmp2 = torch.stack([B]*m).transpose(0,1)
    C = torch.sum((mmp1-mmp2)**2,2).squeeze()

    return C if squared else (C + eps).sqrt() + eps

def get_center_loss_single_center_each_class_margin_hard(centers, features, target, alpha, num_classes, margin=1.0):
    batch_size = target.size(0)
    features_dim = features.size(1)
    
    target_expand = target.view(batch_size,1).expand(batch_size, features_dim)
    centers_var = Variable(centers, requires_grad=False)

    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features
    
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    #############################################
    # additional 
    #############################################
    centers_var.requires_grad = True 
    # nothing 
    # else 
    # centers_batch_b1 = centers_var.gather(0,target_expand)
    feats_centers_dist = pdist2(features, centers_batch)
    # else 
    # feats_centers_dist = pdist2(features, centers_batch_b1)

    # normalize data 
    norms = feats_centers_dist.norm(2, 1)
    feats_centers_dist = feats_centers_dist / norms.unsqueeze(1).repeat(1, feats_centers_dist.size(1)) 

    pos = torch.eq(*[target.unsqueeze(dim).expand_as(feats_centers_dist) for dim in [0, 1]]).type_as(features)
    pd, _ = (pos * feats_centers_dist).max(1)
    n_pos = pos.eq(0).float() 
    nd, _= (feats_centers_dist * n_pos).masked_fill_(pos.byte(), float('inf')).min(1)
    dist_m = pd + margin - nd
    
    loss_mh = torch.clamp(dist_m, min=0.0).mean(0).squeeze()

    center_loss_mh = loss_mh + center_loss

    return center_loss_mh, centers # , centers_var
    
def get_centers_loss_margin_hard(centers, features, target, alpha, num_classes, margin=1.0):
    batch_size = target.size(0)
    features_dim = features.size(1)
    num_centers = centers.size(1)

    # bz x 3 x 4096
    features_view = features.unsqueeze(1).expand(batch_size, 3, features_dim) # bz x 3 x feat_Dim
    target_expand = target.view(batch_size,1, 1).expand(batch_size,3, features_dim)
    centers_var = Variable(centers, requires_grad=False)
    centers_batch = centers_var.gather(0,target_expand) # bz x 3 x feat_Dim

    criterion = nn.MSELoss()
    center_loss = criterion(features_view,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features_view # bz x 3 x 4096 

    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1,1).expand(batch_size,3, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    centers_var.requires_grad=True
    New_Feats = features.view(batch_size, 1, 1, features_dim).expand(batch_size, num_classes, num_centers, features_dim)
    centers_var_view = centers_var.view(1, num_classes, num_centers, features_dim).expand(batch_size, num_classes, num_centers, features_dim)
    sim_D = torch.pow(New_Feats - centers_var, 2).sum(3).sqrt()  # bz x num_class x K 
    norms = sim_D.norm(2, 2)
    sim_D = sim_D / norms.unsqueeze(2).repeat(1,1,sim_D.size(2)) 
    # make index 
    pos_mask = Variable(torch.ByteTensor().resize_(sim_D.size()).zero_())
    for i in range(batch_size):
        pos_mask.data[i][target.data[i]][:].fill_(1) 

    pd, _= sim_D.masked_select(pos_mask.cuda()).view(pos_mask.size(0), -1).max(1)
    neg_mask = pos_mask.eq(0)
    nd, _= sim_D.masked_select(neg_mask.cuda()).view(neg_mask.size(0), -1).min(1)
    margin_tensor = Variable(torch.Tensor([margin]).expand(nd.size(0)).cuda())
    diff_margin = pd + margin_tensor - nd
    loss_mh = torch.clamp(diff_margin, min=0.0).mean(0).squeeze()

    # debug_here() 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    # debug_here() 
    center_loss_mh = loss_mh + center_loss
     
    return center_loss_mh, centers

def get_centers_loss_margin_hard_v2(centers, features, target, alpha, num_classes, margin=1.0):
    batch_size = target.size(0)
    features_dim = features.size(1)
    num_centers = centers.size(1)

    # bz x 3 x 4096
    features_view = features.unsqueeze(1).expand(batch_size, 3, features_dim) # bz x 3 x feat_Dim
    target_expand = target.view(batch_size,1, 1).expand(batch_size,3, features_dim)
    centers_var = Variable(centers, requires_grad=False)
    centers_batch = centers_var.gather(0,target_expand) # bz x 3 x feat_Dim

    criterion = nn.MSELoss()
    center_loss = criterion(features_view,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features_view # bz x 3 x 4096 

    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1,1).expand(batch_size,3, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    New_Feats = features.unsqueeze(1).expand(batch_size, num_centers, features_dim).contiguous().view(batch_size*num_centers, features_dim)
    centers_batch = centers_batch.view(batch_size*num_centers, features_dim)
    feats_centers_dist_raw = pdist2(New_Feats, centers_batch)
    # declare a mask: 24 x 24: every three element contains a one 
    L = batch_size * num_centers # 8(batch_size) * 3(number_centers) 
    Mask_Three = torch.arange(0, L*L).view(L, L)
    Mask_Three = Mask_Three.apply_(lambda x: 1 if x%3 == 0  else 0).byte()
    Mask_Three = Variable(Mask_Three).cuda() 

    feats_centers_dist = feats_centers_dist_raw.masked_select(Mask_Three).view(batch_size, batch_size,num_centers)
    norms = feats_centers_dist.norm(2, 2)
    feats_centers_dist = feats_centers_dist / norms.unsqueeze(2).repeat(1, 1, feats_centers_dist.size(2)) 

    # make index 
    pos_mask = Variable(torch.ByteTensor().resize_(feats_centers_dist.size()).zero_()).float().cuda() 
    for i in range(batch_size):
        pos_mask.data[i][i][:].fill_(1) 

    # debug_here() 
    neg_mask = pos_mask.eq(0).float() 
    pd_1, _ = (feats_centers_dist * pos_mask).max(1)
    pd, _ = pd_1.max(1)

    nd_1, _ = (feats_centers_dist * neg_mask).masked_fill_(pos_mask.byte(), float('inf')).min(1)
    nd, _ = nd_1.min(1)

    margin_tensor = Variable(torch.Tensor([margin]).expand(nd.size(0)).cuda())
    diff_margin = pd + margin_tensor - nd
    loss_mh = torch.clamp(diff_margin, min=0.0).mean(0).squeeze()

    debug_here() 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    center_loss_mh = loss_mh + center_loss
     
    return center_loss_mh, centers

#########################################################################################
## end of centers_loss_margin_hard
#########################################################################################

# update nearest center 
def get_center_loss_nn(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)
    # bz x 3 x 4096
    features_view = features.unsqueeze(1).expand(batch_size, 3, features_dim)

    target_expand = target.view(batch_size,1, 1).expand(batch_size,3, features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features_view,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features_view # bz x 3 x 4096 
    
    # debug_here() 
    norm_diff_3dim = torch.norm(diff.data, 2, 2)
    _, min_idx = torch.min(norm_diff_3dim, 1)

    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1,1).expand(batch_size,3, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 

    for i in range(batch_size):
        centers[target.data[i]][min_idx[i]] -= diff_cpu[i][min_idx[i]].type(centers.type())
    """
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())
    """
    return center_loss, centers


def get_center_loss_single_center_each_class(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)
    
    target_expand = target.view(batch_size,1).expand(batch_size, features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features
    
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers



# this loss will try to drag the center away from each other 
def get_contrastive_center_loss(centers, targets):

    num_classes = centers.size(0) # for shapenet55, it is 55
    l2_norm = centers.norm(2)  # normalize the input
    centers = centers.div_(l2_norm)

    centers_var = Variable(centers, requires_grad = True) 
    centers_var_stack = torch.stack([centers_var]*num_classes) 
    centers_var_stack_t = torch.stack([centers_var]*num_classes).transpose(0, 1)

    # zero out coresponding centers which are not updated during this iterations 
    distance_map = torch.sum((centers_var_stack - centers_var_stack_t)**2, 2).squeeze() 

    mask = torch.zeros(num_classes, num_classes).long()
    mask[targets.data.cpu(), :] = 1
    mask = mask.type_as(centers)
    distance_map.data = distance_map.data * mask

    # different classes are different, so enforce their distance to 1 
    # we should normalize centers 
    cross_target = 1 - np.identity(num_classes) 
    cross_target = torch.from_numpy(cross_target).type_as(centers)
    cross_target = cross_target * mask

    # target =  np.ones((num_classes, num_classes))
    cross_target = Variable(cross_target)
    
    criterion = nn.MSELoss()

    # if we want distance d12 to be 1, then we need to |x1 - x2| to be reach 1
    contrastive_center_loss = criterion(distance_map, cross_target)
    # print(contrastive_center_loss)

    # based on the input centers, we update its centers 
    contrastive_center_loss.backward() 

    centers_var.grad.data = centers_var.grad.data * mask 
    # up : 0.01
    centers_var.data -= 0.01 * centers_var.grad.data

    # resume
    centers = centers.mul_(l2_norm)

    return centers

