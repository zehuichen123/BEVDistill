import torch
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule
from ..registry import CL_HEADS
from mmdet3d.models.builder import build_loss

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'cross_entropy':
        z = z.detach()
        z_softmax = F.softmax(z, dim=-1)
        p_logsoftmax = F.log_softmax(p, dim=-1)
        return - (z_softmax * p_logsoftmax).sum(dim=-1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024, norm_cfg=None):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=1024, norm_cfg=None): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
        
        
@CL_HEADS.register_module()
class SimSiamCLHead(BaseModule):
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels=64,
                 hidden_dim=256,
                 img_proj_num=1,
                 pts_proj_num=1,
                 norm_cfg=None,
                 loss_cl_type='simplified',
            ):
        super().__init__()
        img_input_channels = img_channels
        pts_input_channels = pts_channels
        img_projs = []; pts_projs = []
        for ii in range(img_proj_num):
            img_proj =  nn.Sequential(
                nn.Linear(img_input_channels, mid_channels),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True)
            )
            img_input_channels = mid_channels
            img_projs.append(img_proj)
        for ii in range(pts_proj_num):
            pts_proj =  nn.Sequential(
                nn.Linear(pts_input_channels, mid_channels),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True)
            )
            pts_input_channels = mid_channels
            pts_projs.append(pts_proj)
        self.img_projs = nn.ModuleList(img_projs)
        self.pts_projs = nn.ModuleList(pts_projs)
        self.projector = projection_MLP(mid_channels, mid_channels, mid_channels, norm_cfg=norm_cfg)
        self.predictor = prediction_MLP(mid_channels, hidden_dim, mid_channels, norm_cfg=norm_cfg)
        
        self.loss_cl_type = loss_cl_type

    # @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def loss(self, pts_feats, img_feats):
        pts_z = self.projector(pts_feats)
        img_z = self.projector(img_feats)

        pts_p = self.predictor(pts_z)
        img_p = self.predictor(img_z)

        L = D(pts_p, img_z, self.loss_cl_type) / 2 + D(img_p, pts_z, self.loss_cl_type) / 2
        return L

    def forward(self, img_feats, pts_feats):
        for pts_proj in self.pts_projs:
            pts_feats = pts_proj(pts_feats)
        for img_proj in self.img_projs:
            img_feats = img_proj(img_feats)

        loss_cl = self.loss(pts_feats, img_feats)
        return loss_cl
