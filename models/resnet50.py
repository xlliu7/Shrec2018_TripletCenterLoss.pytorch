import torch 
import torch.nn as nn
from torchvision import models 
# import ipdb
from torch.autograd import Variable


'''
ResNet(
    0: conv1
    1: bn1
    2: relu
    3: maxpool
    4: layer1
    5: layer2
    6: layer3
    7: layer4
    8: avgpool
    9: fc
)
'''
nclasses = 90
original_model = models.resnet50(pretrained=True)
# original_model.classifier._modules['6'] = nn.Linear(4096, nclasses)
def Net_Classifier(nfea=2048, nclasses=90):
    return nn.Linear(nfea, nclasses)

class Net_Prev_Pool(nn.Module):  
    def __init__(self, pool_idx=7): 
        super(Net_Prev_Pool, self).__init__() 
        self.Prev_Pool_Net =  nn.Sequential(
            # use bottom layers, suppose pool_idx = 1, 
            # then we use the bottomest layer(i.e, first layer)
            *list(original_model.children())[:pool_idx] 
            )
    def forward(self, x): 
        x = self.Prev_Pool_Net(x) 
        return x 


# this layer has no parameters
class View_And_Pool(nn.Module):  
    def __init__(self): 
        super(View_And_Pool, self).__init__() 
        # note that in python, dimension idx starts from 1
        # self.Pool_Net =  legacy_nn.Max(1)
        # only max pool layer, we will use view in forward function
        # self.w = nn.Parameter(torch.ones(1, 12, 1, 1, 1), requires_grad=True)
        # self. = nn.Parameter(torch.zeros(12, 4096), requires_grad=True)

    def forward(self, x): 
        # view x ( (bz*12) x C x H x W) ) as 
        # bz x 12 x C x H x W 
        # transform each view: 12 x C x H x W -> 12 X C x H x W
        x = x.view(-1, 12, x.size()[1], x.size()[2], x.size()[3])
        # using average pool instead of max pool 
        x, _= torch.max(x, 1)
        
        return x

class Net_After_Pool(nn.Module):  
    def __init__(self, pool_idx=7): 
        super(Net_After_Pool, self).__init__() 
        self.After_Pool_Net =  nn.Sequential(
            # use top layers, suppose pool_idx = 1, 
            # then we use from 2 layer up to the topest layer 
            *list(original_model.children())[pool_idx:-1]
            )
        #self.modules_list = nn.ModuleList([module for module in original_model.fc.children()])


    def forward(self, x): 
        # forward through all left layers except the classifier
        # however, an additional reshape operation is needed
        x = self.After_Pool_Net(x)
        # need to insert a view layer so that we can feed it to classification layers 
        x = x.view(x.size()[0], -1)

        #x = self.modules_list[0](x) 
        #x = self.modules_list[1](x)
        #x = self.modules_list[2](x)
        #x = self.modules_list[3](x) 
        #x = self.modules_list[4](x)
        #out1 = self.modules_list[5](x)
        # out2 = self.modules_list[6](out1)
        return x #[out1, out2]

class Net_Whole(nn.Module):
    def __init__(self, nclasses=90, use_finetuned=False):
        super(Net_Whole, self).__init__()
        if use_finetuned:
            net = models.resnet50(num_classes=250)
            d = torch.load('../sketch_finetune/resnet50_sketch.pth.tar')
            sd = d['state_dict']
            od = net.state_dict()
            for sk, ok in zip(sd.keys(), od.keys()):
                od[ok] = sd[sk]
            #net.load_state_dict(d['state_dict'])
        else:
            net = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            *list(original_model.children())[:-1])
        #classifier = net.classifier
        # classifier._modules['6'] = nn.Linear(4096, nclasses)
        #self.modules_list = nn.ModuleList([module for module in classifier.children()])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        
        # out2 = self.modules_list[6](out1)
        return x #[out1, out2]


# no use
class zzc_maxpooling(nn.Module):
    def __init__(self):
        super(zzc_maxpooling, self).__init__()
        net = models.alexnet(pretrained=False)
        self.features = net.features
        self.classifier = net.classifier

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.features(x)
        x = x.view(-1, 12, x.size()[1], x.size()[2], x.size()[3])
        x, _= torch.max(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    '''
    pool_idx = 13
    # avoid  pool at relu layer, because if relu is inplace, then 
    # may cause misleading
    model_prev_pool = Net_Prev_Pool(pool_idx).cuda()
    view_and_pool = View_And_Pool().cuda()
    # ipdb.set_trace()
    x = Variable(torch.rand(12*2, 3, 224, 224).cuda())
    model_after_pool = Net_After_Pool(pool_idx).cuda()
    bp = model_prev_pool(x)
    ap = view_and_pool(bp)
    o1 = model_after_pool(ap)

    whole = Net_Whole().cuda()
    ipdb.set_trace()

    x = Variable(torch.rand(12*2, 3, 224, 224).cuda())
    o2 = whole(x)
    '''
    m = zzc_maxpooling()
    x = Variable(torch.rand(2, 12, 3, 224, 224).cuda())
    o = m(x)
    ipdb.set_trace()



