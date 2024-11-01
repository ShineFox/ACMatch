import torch
import torch.nn as nn
import numpy as np


class GridPosition(nn.Module):
    def __init__(self, grid_num, use_gpu = True):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.use_gpu = use_gpu

    def forward(self, batch_size):
        grid_center_x = torch.linspace(-1.+2./self.grid_num/2,1.-2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num)
        grid_center_y = torch.linspace(1.-2./self.grid_num/2,-1.+2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num)
        # BCHW, (b,:,h,w)->(x,y)
        grid_center_position_mat = torch.reshape(
            torch.cartesian_prod(grid_center_x, grid_center_y),
            (1, self.grid_num, self.grid_num, 2)
        ).permute(0,3,2,1)
        # BCN, (b,:,n)->(x,y), left to right then up to down
        grid_center_position_seq = grid_center_position_mat.reshape(1, 2, self.grid_num*self.grid_num)
        return grid_center_position_seq.repeat(batch_size, 1, 1)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, :]

def get_graph_feature(x, k=8, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
  
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    delta_feats = x - feature 
    conv = nn.Conv2d(128, 1, kernel_size=1)
    feature1 = delta_feats.transpose(1,3)
    conv = conv.cuda()
    feature_conv = conv(feature1).transpose(1,3)
    feature_softmax = torch.softmax(feature_conv,dim=2)
    feature = torch.cat((x, feature_softmax*feature), dim=3).permute(0, 3, 1, 2).contiguous()   # equ.(3)
    return feature


class KNNFeats(nn.Module):
    def __init__(self,in_dim, out_dim,use_bn=True):
        super().__init__()
        self.cat_filter = nn.Sequential(
            nn.Conv2d(2*out_dim, 2*out_dim, kernel_size=1),
            nn.BatchNorm2d(2*out_dim), nn.ReLU(),
            nn.Conv2d(2*out_dim, out_dim, kernel_size=1),
        )
        if use_bn:
            self.mlp=nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
                nn.Conv2d(out_dim, out_dim, 1),
            )
        else:
            self.mlp=nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 1),
            )

    def forward(self, feats):
        nn_feats=get_graph_feature(feats) # b*c*n*k
        nn_feats = self.cat_filter(nn_feats)
        nn_feats_out=self.mlp(nn_feats) # b*c*n*k 
        feats_out=torch.max(nn_feats_out, 3, keepdim=True)[0] # b*c*n*1
        return feats_out # b * 8 * n * 1

class LCLayer(nn.Module):
    def __init__(self,in_dim,knn_dim):
        super().__init__()
        self.knn_feats=KNNFeats(in_dim,knn_dim)
        self.conv_out=nn.Conv2d(knn_dim,in_dim,1)

    def forward(self, feats):
        feats_knn=self.knn_feats(feats)
        return self.conv_out(feats_knn).squeeze(dim=3) # b*128*n


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new


class ImportancePrediction(nn.Module):
    def __init__(self, in_channels, grid_size):
        super(ImportancePrediction, self).__init__()
        self.grid_size = grid_size
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),  
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  
            nn.Sigmoid()
        )

    def forward(self, features):
        batch_size, channels, _ = features.size()
        features = features.reshape(batch_size, channels, self.grid_size, -1) # b*c*（k*k） -> b*(k*k)*c
        importance_scores = self.mlp(features)  # Predict importance scores
        # print('importance_scores ', importance_scores.shape) # b*1*k*k
        return importance_scores


class ResBlock(nn.Module):
    def __init__(self, channels,grid_num):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.grid_num = grid_num
        self.weight = ImportancePrediction(channels, grid_num) # 16 is grid_num for simplification

    def forward(self, x):
        # BCHW -> BCN
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num*self.grid_num)        
        weight = self.weight(x)
        # print(weight.shape) #16*1*16*16 
        # BCN -> BCHW
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x = self.net(weight*x) + x
        return x


class Filter(nn.Module):
    def __init__(self, channels, grid_num):
        nn.Module.__init__(self)
        self.resnet = nn.Sequential(*[ResBlock(channels, grid_num) for _ in range(3)])
        self.scale =  nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        x = self.scale(self.resnet(x))
        return x


class FilterNet(nn.Module):
    def __init__(self, grid_num, channels):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.filter = Filter(channels, grid_num)

    def forward(self, x):
        # BCN -> BCHW
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x = self.filter(x)
        # BCHW -> BCN
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num*self.grid_num)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class LayerBlock(nn.Module):
    def __init__(self, channels, head, grid_num):
        nn.Module.__init__(self)
        self.align = AttentionPropagation(channels, head)
        self.lcl = LCLayer(channels, 128)
        self.filter = FilterNet(grid_num, channels)
        self.dealign = AttentionPropagation(channels, head)
        self.inlier_pre = InlinerPredictor(channels)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )
    def forward(self, xs, d, grid_pos_embed):
        # xs: B1N4
        d_loc = self.lcl(d)
        d = d + self.cat_filter(torch.cat((d, d_loc), dim=1))  
        grid_d = self.align(grid_pos_embed, d)
        # weight = self.weight(grid_d)
        grid_d = self.filter(grid_d)
        d_new = self.dealign(d, grid_d)
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d_new - d), 1)
        e_hat = weighted_8points(xs, logits)
        return d_new, logits, e_hat


class ACMatch(nn.Module):
    def __init__(self, config, use_gpu=True):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num

        self.grid_center = GridPosition(config.grid_num, use_gpu=use_gpu)
        self.pos_embed = PositionEncoder(config.net_channels)
        self.grid_pos_embed = PositionEncoder(config.net_channels)
        self.init_project = InitProject(config.net_channels)
        self.layer_blocks = nn.Sequential(
            *[LayerBlock(config.net_channels, config.head, config.grid_num) for _ in range(self.layer_num)]
        )

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1,3).squeeze(3)
        x1, x2 = input[:,:2,:], input[:,2:,:]
        motion = x2 - x1
        # print(x1)

        pos = x1 # B2N
        grid_pos = self.grid_center(batch_size) # B2N

        pos_embed = self.pos_embed(pos) # BCN
        grid_pos_embed = self.grid_pos_embed(grid_pos)

        d = self.init_project(motion) + pos_embed # BCN

        res_logits, res_e_hat = [], []
        for i in range(self.layer_num):
            d, logits, e_hat = self.layer_blocks[i](data['xs'], d, grid_pos_embed) # BCN
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat 


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

