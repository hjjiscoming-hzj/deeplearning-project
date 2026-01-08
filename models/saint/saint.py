from models.base_model import *
import torch.nn as nn
import torch.nn.functional as F

# class Predictor(nn.Module):
#     def __init__(self, in_features=8, HS数据集相关性分析=16):
#         super(Predictor, self).__init__()
#         self.hidden1 = nn.Linear(in_features=in_features, out_features=20, bias = True)
#         self.hidden2 = nn.Linear(20,20)
#         self.hidden3 = nn.Linear(20,5)
#         self.predict = nn.Linear(5,HS数据集相关性分析)
    
#     def forward(self,x):
#         x = F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
#         HS数据集相关性分析 = self.predict(x)
#         out = HS数据集相关性分析.view(-1)

#         return out
    
    
default_word_len = 1000

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        y_dim = 2,
        ):
        super().__init__()
        self.norm = nn.LayerNorm(num_continuous)
        # self.num_continuous = num_continuous
        self.num_continuous = [num_continuous]

        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.hidden_dims = 128
        self.embed_dim = 32
        self.embedding = nn.Embedding(default_word_len, self.dim)

        # start modifying embeddings
        self.simple_MLP = nn.ModuleList([simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)])
        self.extractor = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )
        self.predictor = simple_MLP([dim, self.hidden_dims, y_dim])       
        # self.predictor = Predictor(dim, y_dim)        
        # end modification        
        
    def forward(self, x_cat, x_cont):  
        x_categ_enc, x_cont_enc = self.embed_data_cont(x_cat, x_cont)      
        x = self.extractor(x_categ_enc, x_cont_enc)
        x = x[:,0,:]
        x = self.predictor(x)
        return x