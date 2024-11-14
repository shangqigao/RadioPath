import torch
import torch.nn.functional as F

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_x, pow_tensor
from models.attention import  AttentionLayer


class ScoreNetworkA_eigen(torch.nn.Module):

    def __init__(self, max_feat_num, nhid, depth=3):

        super(ScoreNetworkA_eigen, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.row_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.row_layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.row_layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.row_final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nhid,
                            use_bn=False, activate_func=F.elu)
        
        self.col_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.col_layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.col_layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.col_final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nhid,
                            use_bn=False, activate_func=F.elu)

        self.final_with_eigen = MLP(num_layers=2, input_dim=2, hidden_dim=self.nhid, output_dim=1,
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags, u, la):
        # get row vectors
        row_x = x
        row_list = [row_x]
        for _ in range(self.depth):
            row_x = self.row_layers[_](row_x, adj)
            row_x = self.activation(row_x)
            row_list.append(row_x)

        row_x = torch.cat(row_list, dim=-1)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        row_x = self.row_final(row_x).view(*out_shape)
        row_x = mask_x(row_x, flags)
        # get column vectors
        col_x = x
        col_list = [col_x]
        for _ in range(self.depth):
            col_x = self.col_layers[_](col_x, adj)
            col_x = self.activation(col_x)
            col_list.append(col_x)

        col_x = torch.cat(col_list, dim=-1)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        col_x = self.col_final(col_x).view(*out_shape)
        col_x = mask_x(col_x, flags)
        row_x = row_x.transpose(-2, -1).unsqueeze(-2)
        col_x = col_x.transpose(-2, -1).unsqueeze(-1)
        # compute eigens of low-rank approximation
        u = u.unsqueeze(1)
        row_x = torch.matmul(row_x, u).squeeze(-2)
        col_x = torch.matmul(u.transpose(-2, -1), col_x).squeeze(-1)
        x = torch.sum(row_x * col_x, dim=-2)
        x = torch.stack((la, x), dim=-1)
        x = self.final_with_eigen(x)

        assert not torch.any(x.isnan())
        return x.squeeze(-1)


class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init,
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid,
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid,
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num,
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x
