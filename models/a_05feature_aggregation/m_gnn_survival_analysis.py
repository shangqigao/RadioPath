import json
import joblib
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbnn as bnn

from torch.nn import BatchNorm1d, Linear, ReLU, Dropout
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, softmax, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import EdgeConv, GINConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, TopKPooling
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, AttentionalAggregation
    
class SurvivalGraphDataset(Dataset):
    """loading graph data for survival analysis
    """
    def __init__(self, info_list, mode="train", preproc=None, 
                 data_types=["radiomics", "pathomics"],
                 sampling_rate=1.0,
                 max_num_nodes=1e4
        ):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.data_types = data_types
        self.sampling_rate = sampling_rate
        self.max_num_nodes = max_num_nodes
    
    def get(self, idx):
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            graph_path, label = info
            label = torch.tensor(label).unsqueeze(0)
        else:
            graph_path = info
        
        if len(self.data_types) == 1:
            key = self.data_types[0]
            with pathlib.Path(graph_path[key]).open() as fptr:
                graph_dict = json.load(fptr)
            graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}
            if key == "radiomics": graph_dict["edge_index"] = graph_dict["edge_index"].T

            if self.preproc is not None:
                graph_dict["x"] = self.preproc[key](graph_dict["x"])

            graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
            if any(v in self.mode for v in ["train", "valid"]):
                graph_dict.update({"y": label})

            if self.sampling_rate < 1 and key == "radiomics":
                num_nodes = len(graph_dict["x"])
                if num_nodes > self.max_num_nodes:
                    num_sampled = int(num_nodes*self.sampling_rate)
                    start = torch.randperm(num_nodes - num_sampled)[0]
                    subset = torch.arange(start, start + num_sampled)
                    edge_index, _ = subgraph(subset, graph_dict["edge_index"], relabel_nodes=True)
                    graph_dict["edge_index"] = edge_index
                    graph_dict["x"] = graph_dict["x"][subset]

            graph_dict = {k: v.type(torch.float32) for k, v in graph_dict.items()}
            graph_dict["edge_index"] = graph_dict["edge_index"].type(torch.int64)
            graph = Data(**graph_dict)
        else:
            graph_dict = {}
            for key in self.data_types:
                with pathlib.Path(graph_path[key]).open() as fptr:
                    subgraph_dict = json.load(fptr)
                subgraph_dict = {k: np.array(v) for k, v in subgraph_dict.items() if k != "cluster_points"}
                if key == "radiomics": subgraph_dict["edge_index"] = subgraph_dict["edge_index"].T

                if self.preproc is not None:
                    subgraph_dict["x"] = self.preproc[key](subgraph_dict["x"])

                subgraph_dict = {k: torch.tensor(v) for k, v in subgraph_dict.items()}
                if any(v in self.mode for v in ["train", "valid"]):
                    subgraph_dict.update({"y": label})

                if self.sampling_rate < 1 and key == "radiomics":
                    num_nodes = len(subgraph_dict["x"])
                    if num_nodes > self.max_num_nodes:
                        num_sampled = int(num_nodes*self.sampling_rate)
                        start = torch.randperm(num_nodes - num_sampled)[0]
                        subset = torch.arange(start, start + num_sampled)
                        edge_index, _ = subgraph(subset, subgraph_dict["edge_index"], relabel_nodes=True)
                        subgraph_dict["edge_index"] = edge_index
                        subgraph_dict["x"] = subgraph_dict["x"][subset]
                        del edge_index

                edge_dict = {"edge_index": subgraph_dict["edge_index"].type(torch.int64)}
                subgraph_dict = {k: v.type(torch.float32) for k, v in subgraph_dict.items() if k != "edge_index"}

                graph_dict.update({key: subgraph_dict, (key, "to", key): edge_dict})
                del edge_dict
                del subgraph_dict
            
            graph = HeteroData(graph_dict)
        return graph
    
    def len(self):
        return len(self.info_list)

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Bayes_Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1, Bayes_std=0.1):
        super(Bayes_Attn_Net_Gated, self).__init__()
        self.attention_a = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = bnn.BayesLinear(0, Bayes_std, D, n_classes)
        # self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Score_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Score_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D)]

        self.attention_b = [nn.Linear(L, D)]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class ImportanceScoreArch(nn.Module):
    """define importance score architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.num_layers = len(self.embedding_dims)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                # BatchNorm1d(out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        self.tail = Linear(self.embedding_dims[-1], dim_target)  

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embedding_dims[1:]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
                self.linears.append(Linear(out_emb_dim, out_emb_dim))
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(create_block(out_emb_dim, out_emb_dim))
            else:
                subnet = create_block(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(nn.Sequential())
                
            input_emb_dim = out_emb_dim

    def forward(self, feature, edge_index):
        feature = self.head(feature)
        for layer in range(1, self.num_layers):
            if self.conv_name in ["MLP"]:
                feature = self.convs[layer - 1](feature)
            else:
                feature = self.convs[layer - 1](feature, edge_index)
            feature = self.linears[layer - 1](feature)
        output = self.tail(feature)
        return output
    
class DownsampleArch(nn.Module):
    """define downsample architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_hidden,
            dropout=0.0,
            pool_ratio=0.1,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        for out_emb_dim in [dim_hidden, dim_hidden]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
            else:
                subnet = create_block(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
            self.pools.append(TopKPooling(out_emb_dim, ratio=pool_ratio))    
            input_emb_dim = out_emb_dim

    def forward(self, feature, edge_index, batch):
        perm_list = []
        for l in range(2):
            if self.conv_name in ["MLP"]:
                feature = self.convs[l](feature)
            else:
                feature = self.convs[l](feature, edge_index)
            feature, edge_index, _, batch, perm, score = self.pools[l](feature, edge_index, batch=batch)
            perm_list.append(perm)
        return feature, edge_index, batch, perm_list, score
    
class OmicsIntegrationArch(nn.Module):
    """define importance score architecture
    """
    def __init__(
            self, 
            dim_teachers, 
            dim_student,
            dim_target,
            dropout=0.0,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims, normalize=False):
            if normalize:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    BatchNorm1d(out_dims),
                    ReLU(),
                    Dropout(self.dropout)
                )
            else:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    ReLU(),
                    Dropout(self.dropout)
                )
        
        def conv_block(in_dims, out_dims, name="GCNConv", normalize=False):
            conv_class, alpha = conv_dict[name]
            if name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * in_dims, out_dims, normalize)
                subnet = conv_class(block, **kwargs)
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * in_dims, out_dims)
            else:
                subnet = create_block(alpha * in_dims, out_dims, normalize)
            return subnet
        
        self.align_teachers = nn.ModuleList()
        for ch_t in dim_teachers:
            self.align_teachers.append(
                conv_block(ch_t, 2*dim_target, self.conv_name, normalize=True)
            )

        self.align_student = conv_block(dim_student, 2*dim_target, self.conv_name, normalize=True)

        self.extractor = conv_block(2*dim_target, dim_target, self.conv_name, normalize=True)

        self.recon_teachers = nn.ModuleList()
        for ch_t in dim_teachers:
            self.recon_teachers.append(
                conv_block(dim_target, ch_t, self.conv_name, normalize=False)
            )

    def forward(self, ft, et, fs, es):
        assert len(ft) == len(et)
        aligned_t = [self.align_teachers[i](ft[i], et[i]) for i in range(len(ft))]
        aligned_s = self.align_student(fs, es)
        ht = [self.extractor(f, e) for f, e in zip(aligned_t, et)]
        hs = self.extractor(aligned_s, es)
        ft_ = [self.recon_teachers[i](ht[i], et[i]) for i in range(len(ht))]
        ht = [torch.concat(ht, axis=0)]
        return (hs, ht), (ft_, ft)

class SurvivalGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    Args:
        aggregation: attention-based multiple instance learning (ABMIL) 
        or stochastic importance score informed regression (SISIR)
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            pool_ratio={"radiomics": 0.7, "pathomics": 0.2},
            conv="GINConv",
            keys=["radiomics", "pathomics"],
            aggregation="SISIR",
            mu0=0, 
            lambda0=1, 
            alpha0=2, 
            beta0=1e-2,
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.pool_ratio = pool_ratio
        self.embedding_dims = layers
        self.keys = keys
        self.aggregation = aggregation
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.num_layers = len(self.embedding_dims)
        self.enc_convs = nn.ModuleList()
        self.enc_linears = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_linears = nn.ModuleList()
        self.conv_name = conv
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                # BatchNorm1d(out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        if len(keys) > 1:
            out_emb_dim = self.embedding_dims[0]
            self.enc_branches = nn.ModuleDict({k: create_block(dim_features[k], out_emb_dim) for k in keys})
            self.dec_branches = nn.ModuleDict({k: Linear(out_emb_dim, dim_features[k]) for k in keys})
            input_emb_dim = out_emb_dim
        elif len(keys) == 1:
            input_emb_dim = dim_features[keys[0]]
            out_emb_dim = self.embedding_dims[0]
            self.head = create_block(input_emb_dim, out_emb_dim)
            self.tail = Linear(out_emb_dim, input_emb_dim)
            input_emb_dim = out_emb_dim
        else:
            raise NotImplementedError

        if aggregation == "ABMIL":
            self.gate_nn_dict = nn.ModuleDict(
                {
                    k: Attn_Net_Gated(L=input_emb_dim, D=256, dropout=0.25, n_classes=1)
                    for k in self.keys
                }
            )
            self.Aggregation = SumAggregation()
        elif aggregation == "SISIR":
            # dim_teachers = [dim_features[k] for k in keys]
            # dim_student = input_emb_dim

            # self.integration_nn = OmicsIntegrationArch(
            #     dim_teachers=dim_teachers,
            #     dim_student=dim_student,
            #     dim_target=out_emb_dim,
            #     conv=self.conv_name
            # )
            # self.gate_nn_dict = nn.ModuleDict(
            #     {
            #         k: Attn_Net_Gated(L=out_emb_dim, D=64, dropout=0.25, n_classes=1)
            #         for k in self.keys
            #     }
            # )
            # self.Aggregation = SumAggregation()
            hid_emb_dim = self.embedding_dims[1]
            self.downsample_nn_dict = nn.ModuleDict(
                {
                    k: DownsampleArch(
                        dim_features=input_emb_dim,
                        dim_hidden=hid_emb_dim,
                        dropout=0.0,
                        pool_ratio=self.pool_ratio[k],
                        conv=self.conv_name
                    )
                    for k in self.keys
                }
            )
            out_emb_dim = hid_emb_dim
            input_emb_dim = hid_emb_dim
            self.gate_nn = Attn_Net_Gated(
                L=input_emb_dim, 
                D=256, 
                dropout=0.25, 
                n_classes=2
            )
            # self.score_nn = ImportanceScoreArch(
            #     dim_features=input_emb_dim,
            #     dim_target=2 * hid_emb_dim,
            #     layers=self.embedding_dims[1:],
            #     dropout=self.dropout,
            #     conv=self.conv_name
            # )
            input_emb_dim = 1
            hid_emb_dim = self.embedding_dims[0]
            self.inverse_score_nn = ImportanceScoreArch(
                dim_features=input_emb_dim,
                dim_target=hid_emb_dim,
                layers=self.embedding_dims[1:],
                dropout=0.0,
                conv=self.conv_name
            )
            # self.Aggregation = MeanAggregation()
            self.Aggregation = SumAggregation()
        else:
            raise NotImplementedError
        self.classifier = Linear(len(self.keys)*out_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def sampling(self, data):
        loc, logvar = torch.chunk(data, chunks=2, dim=-1)
        logvar = torch.clamp(logvar, min=-20, max=20)
        gauss = torch.distributions.Normal(loc, torch.exp(0.5*logvar))
        return gauss.sample(), loc, logvar

    def precision(self, loc, logvar):
        alpha = 2*self.alpha0 + 1
        beta = 2*self.beta0 + self.lambda0*(loc - self.mu0)**2 + self.lambda0*torch.exp(logvar)
        return alpha, beta

    def forward(self, data):
        if len(self.keys) > 1:
            x_dict = data.x_dict
            edge_index_dict = {k: data.edge_index_dict[k, "to", k] for k in self.keys}
            batch_dict = data.batch_dict
            enc_list = [x_dict[k] for k in self.keys]
            feature_dict = {k : self.enc_branches[k](e) for k, e in zip(self.keys, enc_list)}
            # # graph concatenation
            # feature = torch.concat(list(feature_dict.values()), dim=0)
            # edge_index_list = [edge_index_dict[k, "to", k] for k in self.keys]
            # # check = [[len(f), e.max().item()] for f, e in zip(feature_list, edge_index_list)]
            # # print("Number of nodes and edge index: ", check)
            # edge_index = [e.clone() for e in edge_index_list]
            # ins = [0]
            # index_shift = 0
            # for i in range(1, len(self.keys)): 
            #     index_shift += len(feature_dict[self.keys[i-1]])
            #     edge_index[i] += index_shift
            #     ins.append(index_shift)
            # ins.append(len(feature))
            # edge_index = torch.concat(edge_index, dim=1)
            # batch_list = [batch_dict[k] for k in self.keys]
            # batch = torch.concat(batch_list, dim=0)
        else:
            feature, edge_index, batch = data.x, data.edge_index, data.batch
            enc_list = [data.x]
            feature = self.head(feature)
            feature_dict = {self.keys[0]: feature}
            edge_index_dict = {self.keys[0]: edge_index}
            batch_dict = {self.keys[0]: batch}

        if self.aggregation == "ABMIL":
            gate_dict = {}
            for k in self.keys:
                gate = self.gate_nn_dict[k](feature_dict[k])
                gate = softmax(gate, index=batch_dict[k], dim=-2)
                gate_dict.update({k: gate})
            VIparas = None
            KAparas = None
            atte_dict = gate_dict
        elif self.aggregation == "SISIR":
            # encoder
            feature_list, edge_index_list, batch_list = [], [], []
            perm_dict, score_list = {}, []
            for k in self.keys:
                feature, edge_index, batch, perm_list, score = self.downsample_nn_dict[k](
                    feature = feature_dict[k], 
                    edge_index = edge_index_dict[k], 
                    batch = batch_dict[k]
                )
                feature_list.append(feature)
                edge_index_list.append(edge_index)
                batch_list.append(batch)
                perm_dict.update({k: perm_list})
                score_list.append(score)
            
            # graph concatenation
            feature = torch.concat(feature_list, dim=0)
            edge_index = [e.clone() for e in edge_index_list]
            ins = [0]
            index_shift = 0
            for i in range(1, len(self.keys)): 
                index_shift += len(feature_list[i - 1])
                edge_index[i] += index_shift
                ins.append(index_shift)
            ins.append(len(feature))
            edge_index = torch.concat(edge_index, dim=1)
            batch = torch.concat(batch_list, dim=0)
            batch_dict = {k: b for k, b in zip(self.keys, batch_list)}
            score = torch.concat(score_list, dim=0)

            # feature alignment by knowledge amalgamation
            # (feature, ht), (ft_, ft) = self.integration_nn(enc_list, edge_index_list, feature, edge_index)
            # KAparas = [feature, ht, ft_, ft]
            KAparas = None

            # feature_dict = {k: feature[ins[i]:ins[i+1]] for i, k in enumerate(self.keys)}
            # gate_dict = {}
            # for k in self.keys:
            #     gate = self.gate_nn_dict[k](feature_dict[k])
            #     gate = softmax(gate, index=batch_dict[k], dim=-2)
            #     gate_dict.update({k: gate})
            # VIparas = None
            
            # encoder
            # encode = self.score_nn(feature, edge_index)
            # gate = softmax(self.gate_nn(feature), index=batch, dim=-2)
            # VIparas = None
            # reparameterization
            sample, loc, logvar = self.sampling(self.gate_nn(feature))
            VIparas = [loc, logvar]

            # caculate normalized inverse precision
            alpha, beta = self.precision(loc, logvar)
            inv_precision = beta / alpha
            N = maybe_num_nodes(batch)
            inv_precision_sum = scatter(inv_precision, index=batch, dim=-2, dim_size=N, reduce='sum')
            inv_precision_sum = inv_precision_sum.index_select(-2, batch)
            gate = inv_precision / inv_precision_sum
            precision_list = [alpha / beta, self.lambda0, self.mu0]
            VIparas.append(precision_list)
            
            feature_dict = {k: feature[ins[i]:ins[i+1]] for i, k in enumerate(self.keys)}
            # feature_dict = {k: v.clone().detach() for k, v in feature_dict.items()}
            gate_dict = {k: gate[ins[i]:ins[i+1]] for i, k in enumerate(self.keys)}
            # # decoder
            decode = self.inverse_score_nn(sample, edge_index)

            if len(self.keys) > 1:
                dec_list = [self.dec_branches[k](decode[ins[i]:ins[i+1], ...]) for i, k in enumerate(self.keys)]
            else:
                dec_list = [self.tail(decode)]

            # get references of downsampled graphs
            ds_enc_list = []
            atte_dict = {}
            for i, k in enumerate(self.keys):
                enc = enc_list[i]
                msk_list = []
                msk = torch.zeros(enc.shape[0], 1).to(enc.device)
                msk_list.append(msk)
                perm_list = perm_dict[k]
                for p in perm_list: 
                    enc = enc[p]
                    msk = torch.zeros(enc.shape[0], 1).to(enc.device)
                    msk_list.append(msk)
                ds_enc_list.append(enc)
                for n, msk in enumerate(msk_list[::-1]):
                    if n == 0:
                        atte = gate_dict[k]
                    else:
                        perm = perm_list[-n]
                        msk[perm] = atte
                        atte = msk
                atte_dict.update({k: atte})
            VIparas.append(ds_enc_list)
            VIparas.append(dec_list)
        feature_list = []
        for k in self.keys:
            feature_list.append(
                self.Aggregation(feature_dict[k] * gate_dict[k], index=batch_dict[k])
            )
        feature = torch.concat(feature_list, dim=-1)
        atte_list = [atte_dict[k] for k in self.keys]
        attention = torch.concat(atte_list, dim=0)
        output = self.classifier(feature)
        return output, VIparas, KAparas, feature, attention
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl=None):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        wsi_outputs, VIparas, KAparas, _, _ = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        if hasattr(wsi_graphs, "y_dict"):
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].squeeze()
        elif hasattr(wsi_graphs, "y"):
            wsi_labels = wsi_graphs.y.squeeze()
        else:
            raise AttributeError
        loss = loss(wsi_outputs, wsi_labels[:, 0], wsi_labels[:, 1], VIparas, KAparas)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        model.eval()
        with torch.inference_mode():
            wsi_outputs, _, _, features, attention = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        features = features.cpu().numpy()
        attention = attention.cpu().numpy()
        wsi_labels = None
        try:
            wsi_labels = wsi_graphs.y_dict[model.keys[0]].cpu().numpy()
        except:
            print("no y dict!")
        if hasattr(wsi_graphs, "y"):
            if wsi_graphs.y is not None:
                wsi_labels = wsi_graphs.y.cpu().numpy()
        if wsi_labels is not None:
            return [wsi_outputs, wsi_labels]
        else:
            # return [wsi_outputs, features]
            return [wsi_outputs, features, attention]

class SurvivalBayesGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            Bayes_std=0.1,
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.num_layers = len(self.embedding_dims)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [None, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim
        out_emb_dim = self.embedding_dims[-1]

        # input_emb_dim = out_emb_dim
        # for out_emb_dim in self.embedding_dims[1:]:
        #     conv_class, alpha = conv_dict[self.conv_name]
        #     if self.conv_name in ["GINConv", "EdgeConv"]:
        #         block = create_block(alpha * input_emb_dim, out_emb_dim)
        #         subnet = conv_class(block, **kwargs)
        #         self.convs.append(subnet)
        #         self.linears.append(bnn.BayesLinear(0, Bayes_std, input_emb_dim, out_emb_dim))
        #     elif self.conv_name in ["GCNConv", "GATConv"]:
        #         subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(create_block(out_emb_dim, out_emb_dim))
        #     else:
        #         subnet = create_block(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(nn.Sequential())
                
        #     input_emb_dim = out_emb_dim
            
        self.gate_nn = Bayes_Attn_Net_Gated(
            L=input_emb_dim,
            D=out_emb_dim,
            dropout=0.25,
            n_classes=1,
            Bayes_std=Bayes_std
        )
        self.global_attention = AttentionalAggregation(
            gate_nn=self.gate_nn,
            nn=None
        )
        self.classifier = Linear(input_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, data):
        feature, edge_index, batch = data.x, data.edge_index, data.batch

        feature = self.head(feature)
        # for layer in range(1, self.num_layers):
        #     feature = F.dropout(feature, p=self.dropout, training=self.training)
        #     if self.conv_name in ["MLP"]:
        #         feature = self.convs[layer - 1](feature)
        #     else:
        #         feature = self.convs[layer - 1](feature, edge_index)
        #     feature = self.linears[layer - 1](feature)

        feature = self.global_attention(feature, index=batch)
        output = self.classifier(feature)
        return output
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels[:, 0], wsi_labels[:, 1])
        kl_loss, kl_weight = kl["loss"](model)[0], kl["weight"]
        loss = loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.cpu().numpy()

            return [wsi_outputs, wsi_labels]
        return [wsi_outputs]

class ScalarMovingAverage:
    """Class to calculate running average."""

    def __init__(self, alpha=0.95):
        """Initialize ScalarMovingAverage."""
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self, step_output):
        """ScalarMovingAverage instances behave and can be called like a function."""
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # Calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # Init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value

class VILoss(nn.Module):
    """ variational inference loss
    """
    def __init__(self, tau_ae=1e-3):
        super(VILoss, self).__init__()
        self.tau_ae = tau_ae

    def forward(self, loc, logvar, pre_list, enc_list, dec_list):
        precision, lambda0, mu0 = pre_list
        precision = precision.clone().detach()
        loss_kl = 0.5*torch.mean(lambda0*precision*((loc - mu0)**2 + torch.exp(logvar)) - logvar)
        loss_ae = 0.0
        for enc, dec in zip(enc_list, dec_list):
            # subset = torch.randperm(len(enc))[:len(dec)]
            # enc = enc[subset]
            loss_ae += F.mse_loss(enc, dec)
        print(loss_ae.item(), loss_kl.item(), precision.min().item(), precision.max().item())
        return self.tau_ae * loss_ae + loss_kl
    
def calc_mmd(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return mmd_rbf2(f1, f2, sigmas=sigmas)


def mmd_rbf2(x, y, sigmas=None):
    N, _ = x.shape
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = L = P = 0.0
    XX2 = rx.t() + rx - 2*xx
    YY2 = ry.t() + ry - 2*yy
    XY2 = rx.t() + ry - 2*zz

    if sigmas is None:
        sigma2 = torch.mean((XX2.detach()+YY2.detach()+2*XY2.detach()) / 4)
        sigmas2 = [sigma2/4, sigma2/2, sigma2, sigma2*2, sigma2*4]
        alphas = [1.0 / (2 * sigma2) for sigma2 in sigmas2]
    else:
        alphas = [1.0 / (2 * sigma**2) for sigma in sigmas]

    for alpha in alphas:
        K += torch.exp(- alpha * (XX2.clamp(min=1e-12)))
        L += torch.exp(- alpha * (YY2.clamp(min=1e-12)))
        P += torch.exp(- alpha * (XY2.clamp(min=1e-12)))

    beta = (1./(N*(N)))
    gamma = (2./(N*N))

    return F.relu(beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P))

class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht, ft_, ft):
        mmd_loss = 0.0
        mse_loss = 0.0
        # random sampling if hs is large-scale
        if len(hs) > 1e5:
            num_sampled = int(len(hs) * 0.01)
            subset = torch.randperm(len(hs))[:num_sampled]
            hs = hs[subset]
            ht = [ht_i[subset] for ht_i in ht]
        for ht_i in ht:
            mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])
        
        return mmd_loss + self.beta*mse_loss
    
class CoxSurvLoss(nn.Module):
    def __init__(self, tau_vi=1e-1, tau_ka=1e-2):
        super(CoxSurvLoss, self).__init__()
        self.tau_vi = tau_vi
        self.tau_ka = tau_ka
        self.loss_vi = VILoss()
        self.loss_ka = CFLoss()

    def forward(self, hazards, time, c, VIparas, KAparas):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        # c = torch.tensor(c).to(hazards.device)
        R_mat = torch.tensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * c)
        if VIparas is not None:
            loss_vi = self.loss_vi(*VIparas)
            loss_cox = loss_cox + self.tau_vi * loss_vi
        if KAparas is not None:
            loss_ka = self.loss_ka(*KAparas)
            loss_cox = loss_cox + self.tau_ka * loss_ka
        return loss_cox
    
