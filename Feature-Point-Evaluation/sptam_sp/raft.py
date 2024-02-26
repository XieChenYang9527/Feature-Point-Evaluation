import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)


    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        # with torch.no_grad():
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            x1 = torch.split(x1, [batch_dim, batch_dim], dim=0)
        return x, x1

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim+input_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        self.norm = torch.nn.InstanceNorm2d(hidden_dim+input_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = self.conv1(hx)
        z = self.norm(z)
        z = self.relu(z)
        z = self.conv2(z)
        return z

class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()

        self.cor1 = nn.Conv2d(1, 32, 3, padding=1)
        self.cor1_norm = torch.nn.InstanceNorm2d(32)
        self.cor1_relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.cor2 = nn.Conv2d(32, 32, 3, padding=1)
        self.cor2_norm = torch.nn.InstanceNorm2d(32)
        self.cor2_relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)


        self.f1 = nn.Conv2d(512, 32, 3, padding=1)
        self.f11 = nn.Conv2d(128, 32, 3, padding=1)
        self.f1_norm = torch.nn.InstanceNorm2d(32)
        self.f1_relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.f2 = nn.Conv2d(32, 32, 3, padding=1)
        self.f2_norm = torch.nn.InstanceNorm2d(32)
        self.f2_relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)


        self.conv = nn.Conv2d(64, 32, 3, padding=1)
        self.conv_norm = torch.nn.InstanceNorm2d(64)
        self.conv_relu = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

    def forward(self, corr,  feature, scale):

        corr = self.cor1(corr)
        corr = self.cor1_norm(corr)
        corr = self.cor1_relu(corr)


        if scale == 0:
             feature = self.f1(feature)
        if scale == 1:
             feature = self.f11(feature)
        feature = self.f1_norm(feature)
        feature = self.f1_relu(feature)

        feature = self.f2(feature)
        feature = self.f2_norm(feature)
        feature = self.f2_relu(feature)

        motion = self.conv( torch.cat( (corr,feature), 1 ) )
        motion = self.conv_norm(motion)
        motion = self.conv_relu(motion)

        return motion

class BasicUpdateBlock(nn.Module):
    def __init__(self,  hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=32)

    def forward(self, delta_flow, corr, feature, scale, upsample=True):
        #with torch.no_grad():
        motion_features = self.encoder(corr,  feature, scale)
        delta_flow = self.gru(delta_flow, motion_features )
        return delta_flow

class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()

        self.unfold = torch.nn.Unfold(kernel_size=(10, 10), stride=10)
        self.fold = torch.nn.Fold(output_size=(160, 320), kernel_size=(10, 10), stride=10)
        self.hidden_dim = hdim = 1
        self.rand = torch.arange(start=0, end=0.01, step=0.0001).to("cuda").view(1, 1, 100)

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.update_block = BasicUpdateBlock( hidden_dim=hdim)
        self.norm = torch.nn.BatchNorm2d(256,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)
        self.norm1 = torch.nn.BatchNorm2d(64,
                                         eps=1e-05,
                                         momentum=0.1,
                                         affine=True,
                                         track_running_stats=True)

        self.instance = nn.InstanceNorm2d(6)
        self.batch = nn.BatchNorm2d(6)




    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)

        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def index_feature(self,feature,coords,r):
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
        # 中心坐标
        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2).transpose(1,2)
        coords_lvl = centroid_lvl + delta_lvl
        coords_lvl = coords_lvl.reshape(batch,-1,(2*r+1),2)
        corr = bilinear_sampler(feature, coords_lvl)
        return corr

    def corr(self, fmap1, fmap2):
        batch1, point, dim1, ht1, wd1 = fmap1.shape
        batch2, point, dim2, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch1, point, dim1, ht1 * wd1)
        fmap2 = fmap2.view(batch2, point, dim2, ht2 * wd2)

        corr = torch.matmul(fmap1.transpose(2, 3), fmap2)
        corr = corr.view(batch1, point, ht1 * wd1, ht2, wd2)
        return corr

    def forward(self, image1, image2, mode, cod ):
        """ Estimate optical flow between pair of frames """

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        image = torch.cat((image1,image2),1)
        # image = self.instance(image)
        image1, image2 = torch.split(image,3,1)
        x, x1 = self.fnet([image1, image2])
        f1,f2 = x
        f11,f21 = x1

        N = f1.shape[0]


        point = torch.ones(1, 1, 10, 20)
        point1 = torch.zeros_like(point).to("cuda")
        point1[:, :, 2:8, 2:18] = point[:, :, 2:8, 2:18]
        point1 = point1.bool()

        coord = coords_grid(N, 10, 20).to("cuda")
        coords = torch.masked_select(coord, point1)
        coords = coords.view(N, 2, -1).view(N, 2, 1, -1).to("cuda")


        flow = torch.zeros_like(coords).to("cuda")
        flow_predictions = []
        num_point = coords.shape[3]
        coords = coords*32

        if mode == 1:
            coords = cod
            flow = torch.zeros_like(coords).to("cuda")
            num_point = coords.shape[3]

        coords_c = coords+flow
        flow = flow

        rad = 5
        dx = torch.linspace(-rad, rad, 2 * rad + 1)
        dy = torch.linspace(-rad, rad, 2 * rad + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device).view(1, 2 * rad + 1, 2 * rad + 1, 2).transpose(1,2)
        delta = delta.permute(0,3,1,2)
        delta_flow = torch.ones(N * num_point, 1, 2 * rad + 1, 2 * rad + 1).to("cuda")

        F1 = self.index_feature(f1, coords / 16, rad).contiguous()
        F1 = F1.view(N, -1, num_point, 2 * rad + 1, 2 * rad + 1).transpose(1, 2).contiguous()
        F1 = F.normalize(F1, p=2, dim=2)

        F11 = F1[:,:,:,rad,rad].view(N,num_point,-1,1,1)+0
        F1 = F1.reshape(N * num_point, -1, 2 * rad + 1, 2 * rad + 1)

        for j in range(2):

            F2 = self.index_feature(f2, coords_c / 16, rad)
            F2 = F2.view(N, -1, num_point, 2*rad+1, 2*rad+1).transpose(1, 2).contiguous()
            F2 = F.normalize(F2, p=2, dim=2)

            corr_f = self.corr(F11,F2)
            corr_f = corr_f.view(N*num_point, 1, 2*rad+1, 2*rad+1)
            F2 = F2.reshape(N * num_point, -1, 2 * rad + 1, 2 * rad + 1) + 0

            F1 = self.norm(F1)
            F2 = self.norm(F2)


            feature = torch.cat((F1, F2), 1)

            delta_flow = self.update_block( delta_flow,corr_f, feature,0)




            delta_flow1 = delta_flow.view(N*num_point,1, (2*rad+1)*(2*rad+1))
            delta_flow1 = torch.nn.functional.softmax(delta_flow1*2, dim=2)
            delta_flow1 = delta_flow1.view(N * num_point, 1, 2 * rad + 1, 2 * rad + 1)
            delta_flow1 = delta*delta_flow1

            delta_flow1 = delta_flow1.reshape(N*num_point,2,(2*rad+1)*(2*rad+1)).sum(2)
            delta_flow1 = delta_flow1.view(N,num_point,2).transpose(1,2).view(N,2,1,-1)
            flow = flow + delta_flow1*16
            coords_c = coords + flow

        corr1 = corr_f+0

        rad = 3
        dx = torch.linspace(-rad, rad, 2 * rad + 1)
        dy = torch.linspace(-rad, rad, 2 * rad + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device).view(1, 2 * rad + 1, 2 * rad + 1,
                                                                                    2).transpose(1, 2)
        delta = delta.permute(0, 3, 1, 2)
        delta_flow = torch.ones(N * num_point, 1, 2 * rad + 1, 2 * rad + 1).to("cuda")

        F1 = self.index_feature(f11, coords/2, rad).contiguous()
        F1 = F1.view(N, -1, num_point, 2 * rad + 1, 2 * rad + 1).transpose(1, 2).contiguous()
        F1 = F.normalize(F1, p=2, dim=2)

        F11 = F1[:, :, :, rad, rad].view(N, num_point, -1, 1, 1) + 0
        F1 = F1.reshape(N * num_point, -1, 2 * rad + 1, 2 * rad + 1)


        for j in range(1):
            F2 = self.index_feature(f21, coords_c/2, rad)
            F2 = F2.view(N, -1, num_point, 2 * rad + 1, 2 * rad + 1).transpose(1, 2).contiguous()
            F2 = F.normalize(F2, p=2, dim=2)

            corr_f = self.corr(F11, F2)
            corr_f = corr_f.view(N * num_point, 1, 2 * rad + 1, 2 * rad + 1)
            F2 = F2.view(N*num_point, -1, 2 * rad + 1, 2 * rad + 1)

            F1 = self.norm1(F1)
            F2 = self.norm1(F2)

            feature = torch.cat((F1, F2), 1)

            delta_flow = self.update_block(delta_flow, corr_f, feature,1)

            delta_flow1 = delta_flow.view(N * num_point, 1, (2 * rad + 1) * (2 * rad + 1))
            delta_flow1 = torch.nn.functional.softmax(delta_flow1*2, dim=2)
            delta_flow1 = delta_flow1.view(N * num_point, 1, 2 * rad + 1, 2 * rad + 1)
            delta_flow1 = delta * delta_flow1

            delta_flow1 = delta_flow1.reshape(N * num_point, 2, (2 * rad + 1) * (2 * rad + 1)).sum(2)
            delta_flow1 = delta_flow1.view(N, num_point, 2).transpose(1, 2).view(N, 2, 1, -1)
            flow = flow + delta_flow1*2
            coords_c = coords + flow

        #corr1 = corr_f + 0

        coords_c = coords_c.view(N, 2, num_point)
        coords = coords.view(N,2,num_point)
        end.record()
        torch.cuda.synchronize()
        return coords_c, coords, corr1


