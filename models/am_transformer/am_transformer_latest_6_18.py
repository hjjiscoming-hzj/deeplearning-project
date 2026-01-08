from models.am_transformer.Attention.blocks import *
from models.am_transformer.Attention.RowColAttentionBlock import *
from einops import rearrange, repeat
from models.base_model import simple_MLP, simple_KAN
from models.src_kan.efficient_kan import KAN
from models.am_transformer.Fusion.fusion import StaticFeatureFusion
# 特征编码后，kan和trans并行，静态特征融合层+mlp输出

# transformer
class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            attn_dropout,
            ff_dropout,
            use_cls_token,
            groups,
            sum_num_per_group,
            prod_num_per_group,
            cluster,
            target_mode,
            token_num,
            token_descent=False,
            use_prod=True,
            qk_relu=False,
            use_row_col_attention=True,
    ):
        super().__init__()
        self.use_prod = use_prod
        self.use_row_col_attention = use_row_col_attention
        # 注意力机制可学习权重（加法、乘法、行列）
        self.attention_weights = nn.Parameter(torch.ones(3))
        self.layers = nn.ModuleList([])

        flag = int(use_cls_token)

        if not token_descent:
            groups = [token_num for _ in groups]

        for i in range(depth):
            token_num = token_num if i == 0 else groups[i - 1]
            self.layers.append(nn.ModuleList([
                # prod memory block
                MemoryBlock(
                    token_num=token_num,
                    heads=heads,
                    dim=dim,
                    attn_dropout=attn_dropout,
                    cluster=cluster,
                    target_mode=target_mode,
                    groups=groups[i],
                    num_per_group=prod_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='prod',
                    qk_relu=qk_relu) if use_prod else nn.Identity(),

                # sum memory block
                MemoryBlock(
                    token_num=token_num,
                    heads=heads,
                    dim=dim,
                    attn_dropout=attn_dropout,
                    cluster=cluster,
                    target_mode=target_mode,
                    groups=groups[i],
                    num_per_group=sum_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='sum',
                    qk_relu=qk_relu) if token_descent else Attention(heads=heads, dim=dim, dropout=attn_dropout),

                # row-col memory block
                RowColAttentionBlock(
                    dim=dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    feature_groups=groups[i],
                    use_cls_token=use_cls_token,
                    qk_relu=qk_relu
                ) if use_row_col_attention else nn.Identity(),
                nn.Linear(2 * (groups[i] + flag), groups[i] + flag),
                nn.Linear(token_num + flag, groups[i] + flag) if token_descent else nn.Identity(),
                FeedForward_MLP(dim, dropout=ff_dropout),
            ]))

    def forward(self, x):

        for toprod, tosum, torowcol, down, downx, ff in self.layers:

            # sum attention
            attn_out = tosum(x)

            # prod attention
            if self.use_prod:
                prod = toprod(x)
                attn_out = down(torch.cat([attn_out, prod], dim=1).transpose(2,1)).transpose(2,1)

            # row-col attention
            if self.use_row_col_attention:
                rowcol_out = torowcol(x)
                attn_out = down(torch.cat([attn_out, rowcol_out], dim=1).transpose(2,1)).transpose(2,1)

            x = attn_out + downx(x.transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x
        return x

    def forward_learn(self, x):
        for toprod, tosum, torowcol, down, downx, ff in self.layers:
            # 初始化注意力输出列表
            attn_outputs = []

            # sum attention
            sum_attn = tosum(x)
            attn_outputs.append(sum_attn)

            # prod attention
            if self.use_prod:
                prod_attn = toprod(x)
                attn_outputs.append(prod_attn)

            # row-col attention
            if self.use_row_col_attention:
                rowcol_attn = torowcol(x)
                attn_outputs.append(rowcol_attn)

            # 对注意力输出进行加权求和
            if len(attn_outputs) > 1:
                # 归一化权重
                weights = F.softmax(self.attention_weights[:len(attn_outputs)], dim=0)

                # 加权求和并保持维度
                weighted_attn = sum(w * out for w, out in zip(weights, attn_outputs))
            else:
                weighted_attn = attn_outputs[0]

            # 使用 down() 进行降维
            attn_out = down(torch.cat([x, weighted_attn], dim=1).transpose(2, 1)).transpose(2, 1)
            x = attn_out + downx(x.transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x
        return x


# numerical embedder
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


# main class
class AMTransformer(nn.Module):
    def __init__(
            self,
            args,
    ):
        super().__init__()
        '''
        dim: token dim
        depth: Attention block numbers
        heads: heads in multi-head attn
        attn_dropout: dropout in attn
        ff_dropout: drop in ff in attn
        use_cls_token: use cls token in FT-transformer but autoint it should be False
        groups: used in Memory block --> how many cluster prompts
        sum_num_per_group: used in Memory block --> topk to sum in each sum cluster prompts
        prod_num_per_group: used in Memory block --> topk to sum in each prod cluster prompts
        cluster: if True, prompt --> q, False, x --> q
        target_mode: if None, prompt --> q, if mix, [prompt, x] --> q
        token_num: how many token in the input x
        token_descent: use in MUCH-TOKEN dataset
        use_prod: use prod block
        use_row_col_attention: 是否使用行列注意力机制
        num_special_tokens: =2
        categories: how many different cate in each cate ol
        out: =1 if regressioin else =cls number
        self.num_cont: how many cont col
        num_cont = args.num_cont
        num_cate: how many cate col
        '''
        dim = args.dim
        depth = args.depth
        heads = args.heads
        attn_dropout = args.attn_dropout
        ff_dropout = args.ff_dropout
        self.use_cls_token = args.use_cls_token
        groups = args.groups
        sum_num_per_group = args.sum_num_per_group
        prod_num_per_group = args.prod_num_per_group
        cluster = args.cluster
        target_mode = args.target_mode
        token_num = args.num_cont + args.num_cate
        token_descent = args.token_descent
        use_prod = args.use_prod
        use_row_col_attention = args.use_row_col_attention
        num_special_tokens = args.num_special_tokens
        categories = args.categories
        self.num_cont = args.num_cont
        num_cont = args.num_cont
        num_cate = args.num_cate
        self.use_sigmoid = args.use_sigmoid
        qk_relu = args.qk_relu
        # 设置输出层为 kan/mlp
        predictor = args.predictor
        # 设置输出维度
        out_dim = args.out_dim
        # 设置静态融合层的 kan：trans比例
        fusion_weight = args.static_fusion

        self.args = args
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_cont > 0, 'input shape must not be null'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        total_tokens = self.num_unique_categories + num_special_tokens + 1

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(args.categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        if self.num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_cont)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer_path = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
            use_row_col_attention=use_row_col_attention,
            qk_relu=qk_relu,
        )

        # KAN 路径
        # self.kan_path = KAN([dim, dim, dim])
        self.kan_path = simple_KAN(dims=[dim, 128, dim])

        # 特征融合层
        self.fusion = StaticFeatureFusion(dim, fusion_weight)
        # self.fusion = DynamicFeatureFusion(dim)
        # self.fusion = HybridFeatureFusion(dim)
        # self.fusion = StaticFeatureFusion_norm(dim)

        if predictor == 'simple_MLP':
            self.predictor = simple_MLP(dims=[dim, 128, out_dim])
        elif predictor == 'simple_KAN':
            self.predictor = simple_KAN(dims=[dim, 128, out_dim])
        elif predictor == 'KAN':
            self.predictor = KAN([dim, 2 * dim + 1, 1])
        args = self.predictor.get_info()
        print('predictor: {}'.format(args.name))

    def model_name(self):
        return 'am_trans'

    def forward(self, conts, x_cat=None):
        xs = []

        # add numerically embedded tokens
        if self.num_cont > 0:
            conts = self.numerical_embedder(conts)
            xs.append(conts)

        # concat categorical and numerical
        x = torch.cat(xs, dim=1)

        # append cls tokens
        b = x.shape[0]
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        # transformer 路径处理
        transformer_out = self.transformer_path(x)

        # KAN 路径处理
        batch_size, seq_len, feature_size = x.shape
        x = x.reshape(-1, feature_size)
        kan_out = self.kan_path(x)
        kan_out = kan_out.reshape(batch_size, seq_len, feature_size)

        # 提取特征
        if self.use_cls_token:
            transformer_out = transformer_out[:, 0]
        else:
            transformer_out = transformer_out.mean(dim=1)
        kan_out = kan_out.mean(dim=1)

        # 特征融合
        fused = self.fusion(transformer_out, kan_out)

        # 最终预测
        x = self.predictor(fused)
        return x

    @classmethod
    def make_default(cls, n_num_features, cat_cardinalities, token_dim, out_dim):
        args_usrdefine = {
            'name': 'AMFormer',
            'dim': token_dim,  # 模型的隐藏维度
            'depth': 2,  # Transformer 编码器层数
            'heads': 4,  # 多头注意力机制中的头数
            'attn_dropout': 0.4,  # 注意力层的 dropout 比率
            'ff_dropout': 0.4,  # 前馈网络层的 dropout 比率
            'use_cls_token': True,  # 是否使用 [CLS] token 作为分类任务的输入
            'groups': [54, 54, 54],  # 分组数量（如果模型中涉及分组的话）
            'sum_num_per_group': [32, 16, 8],  # 每个分组内求和的数量
            'prod_num_per_group': [6, 6, 6],  # 每个分组内求积的数量
            'cluster': 3,  # 聚类数量（如果模型中涉及聚类的话）
            'target_mode': 'regression',  # 目标模式：'classification' 或 'regression'
            'num_cont': n_num_features,  # 连续特征的数量
            'num_cate': len(cat_cardinalities),  # 类别特征的数量
            'token_descent': False,  # 是否启用 token 下降策略
            'use_prod': True,  # 是否使用乘法操作
            'use_row_col_attention': True,  # 是否使用行列注意力机制
            'num_special_tokens': 1,  # 特殊 token 的数量（例如 [CLS], [SEP] 等）
            'categories': cat_cardinalities,  # 类别特征的总类别数
            'use_sigmoid': True,  # 是否在输出层使用 Sigmoid 函数
            'qk_relu': False,  # 在计算 Q 和 K 向量时是否应用 ReLU 激活函数
            'out_dim': out_dim,  # 输出维度
            'predictor': 'simple_MLP' , # 输出层选择 simple_MLP / simple_KAN / KAN
            'static_fusion': {'kan_weight': 0.45, 'trans_weight': 0.55}
        }
        args_usrdefine['name'] = ('{0}_{1}_dim{2}_depth{3}_heads{4}_dropout{5}_fusion{6}_{7}'.
                                  format(args_usrdefine['name'], args_usrdefine['predictor'], args_usrdefine['dim'],
                                         args_usrdefine['depth'], args_usrdefine['heads'],args_usrdefine['attn_dropout'],
                                         args_usrdefine['static_fusion']['kan_weight'],args_usrdefine['static_fusion']['trans_weight']))
        import types
        args = types.SimpleNamespace(**args_usrdefine)
        return AMTransformer(args)

    def get_info(self):
        return self.args