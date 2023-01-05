import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool


class histoGCN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_channels=256):
        super(histoGCN, self).__init__()
        torch.manual_seed(12345)
        self.fc_emb = Linear(input_dim, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, 64)
        self.predict = Linear(64, 1)
        
    def forward(self, x, edge_index, batch):
        embedded = self.fc_emb(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        
        x = self.predict(x)
        x = F.sigmoid(x)
        
        return x

class histoGIN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels=64):
        super(histoGIN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GINConv(embedding_dim, hidden_channels)
        self.conv2 = GINConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, 32)
        self.predict = Linear(32, 1)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        
        x = self.predict(x)
        x = F.sigmoid(x)
        
        return x
    