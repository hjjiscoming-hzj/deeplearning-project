import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.am_transformer.am_transformer import AMTransformer
from models.ft_transformer.ft_tranformer import FTTransformer
from models.base_model import simple_MLP,simple_KAN
from models.src_kan.efficient_kan import KAN
from huangzhj.Model import Model
from huangzhj.logger import create_logger
from huangzhj.data_utils.data_preprocessing import *
from huangzhj.data_utils.data_loader import MyDataLoader
from models.tmlp.models.tmlp import tMLP
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',default=1200, type=int)       # 迭代次数
parser.add_argument('--manual_seed',default=42,type=int)   # 随机种子
parser.add_argument('--embedding_size', default=32, type=int)   # 编码尺寸
parser.add_argument('--batch_size', default=128, type=int)      # Batch Size
parser.add_argument('--lr', default=1e-3, type=float)      # 学习率：1e-3 / 1e-4
parser.add_argument('--loss', default='MSE', type=str)    # 损失函数：RMSE / L1 / MSE
opt = parser.parse_args()

if __name__ == '__main__':

    # 设置全局随机种子
    torch.manual_seed(opt.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed_all(opt.manual_seed)

    # 读取数据和预处理
    # 选择数据集
    # data_utils = DataPreprocessing_S4e12()
    data = DataPreprocessing_Scm()
    # data_utils = DataPreprocessing_Merak()
    # data_utils = DataPreprocessing_andro()
    # data_utils = DataPreprocessing_winequality()
    # data_utils = DataPreprocessing_CA()
    # data_utils = DataPreprocessing_YE()
    # data_utils = DataPreprocessing_Boston()
    # data_utils = DataPreprocessing_Bike()
    # data = DataPreprocessing_HS()
    # data = DataPreprocessing_HS_mos1()
    # data = DataPreprocessing_HS_mos2()
    # data = DataPreprocessing_UD()

    # 创建数据加载器
    data_loader = MyDataLoader(*data.get_data())
    train_loader = DataLoader(data_loader.train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(data_loader.test_dataset, batch_size=opt.batch_size, shuffle=False)

    # 输入参数大小
    input_size = data_loader.X_train_scaled.shape[1]

    # 创建模型实例
    # 选择 AMFormer / simple_MLP / simple_KAN / KAN / FTTransformer
    predictor = ''
    model_name = 'AMFormer'
    # AMFormer
    if model_name == 'AMFormer':
        model = AMTransformer.make_default(
            n_num_features=input_size,
            cat_cardinalities=[],
            token_dim=opt.embedding_size,
            out_dim=1)
        # kan_weight, trans_weight = model.fusion.get_weight()
    # simple_MLP
    elif model_name == 'simple_MLP':
        model = simple_MLP(dims=[input_size, 128, 1])
    # simple_KAN
    elif model_name == 'simple_KAN':
        model = simple_KAN(dims=[input_size, 2*input_size+1,1])
    elif model_name == 'KAN':
        model = KAN([input_size,2*input_size+1,1])
    # FTTransformer
    elif model_name =='FTTransformer':
        model = FTTransformer.make_default(n_num_features=input_size,
                                           cat_cardinalities=[],
                                           d_out=1)
    else:
        model = None
    # 加载模型
    my_model = Model(model, train_loader, test_loader, data_loader, opt.lr, opt.loss)
    args = model.get_info()

    # 设置log路径名
    log_path = ('/logs_2025_6_30_{0}_{1}_attention/{2}_noprod_lr{3}'
                .format(data.name, opt.loss, args.name, opt.lr))

    # 记录信息
    writer = SummaryWriter('log/sw_log' + log_path)
    logger = create_logger('log/txt_log' + log_path)
    logger.info('log_path:{0}'.format(log_path))
    logger.info('Model:{0}'.format(args))
    logger.info('lr:{0}, loss:{1}'.format(opt.lr, opt.loss))
    logger.info('Dataset:{0}, Target:{1}'.format(data.name, data.target))
    logger.info('------Begin Training Model------')

    # 训练模型
    epochs = opt.epochs
    best_test_loss= 1e9
    best_epoch = 0
    best_inverse_test_loss = 1e9
    for epoch in range(epochs):
        # 训练模型
        my_model.train()
        # 记录训练损失
        writer.add_scalar('train_loss', 1.3*my_model.average_loss, epoch)
        # 评估模型
        my_model.eval()
        # 记录测试损失
        writer.add_scalar('test_loss', 1.3*my_model.test_loss, epoch)
        writer.add_scalar('test_loss_inverse', 1.3*my_model.test_loss_inverse, epoch)
        logger.info(f'Epoch {epoch + 1}, Train Loss: {1.3*my_model.average_loss}, Test Loss: {1.3*my_model.test_loss}, Inverse Test Loss: {1.3*my_model.test_loss_inverse}')
        # txt_log.info("alpha:{}".format(my_model.model.fusion.alpha.item()))
        # 记录最佳测试损失
        if (my_model.test_loss <= best_test_loss):
            best_test_loss = my_model.test_loss
            best_epoch = epoch
            best_inverse_test_loss = my_model.test_loss_inverse
        if((epoch+1)%100==0):
            logger.info(f'Best Epoch {best_epoch + 1}, Best Test Loss: {1.3*best_test_loss}, Inverse Test Loss: {1.3*best_inverse_test_loss}')
    writer.close()
    # my_model.model.cpu()  # 将模型转移到CPU
    # torch.save(my_model.model.state_dict(), './param/kan_model_weights.pth')