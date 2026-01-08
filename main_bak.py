import argparse
from models.am_transformer.am_transformer import AMTransformer
from huangzhj.Model import Model
from torch.utils.tensorboard import SummaryWriter
from huangzhj.logger import create_logger

parser = argparse.ArgumentParser()
# parser.add_argument('-data_name', default='scm20d', type=str)
# parser.add_argument('-gpu', default='0', type=str)
# parser.add_argument('--cont_embeddings', default='mlp', type=str, choices=['mlp', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
# parser.add_argument('--transformer_depth', default=4, type=int)
# parser.add_argument('--attention_heads', default=8, type=int)
# parser.add_argument('--attention_dropout', default=0.0, type=float)
# parser.add_argument('--ff_dropout', default=0.0, type=float)
# parser.add_argument('--normlized', default=0, type=int)
# parser.add_argument('-method', default='ewc', type=str)
# parser.add_argument('-num_tasks', default=3, type=int)
# parser.add_argument('-debug_mode', default=0, type=int)
# parser.add_argument('--no_distill', default=True, type=int)
# parser.add_argument('--T', default=2, type=int)
# parser.add_argument('-lr_lower_bound', default=0.00001, type=float)
# parser.add_argument('-patience', default=5, type=int)
# parser.add_argument('--random_seed', default=42, type=int)
#
# parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
#
# parser.add_argument('-lr', default=0.001, type=float)
# parser.add_argument('-epochs', default=500, type=int)
# parser.add_argument('-batch_size', default=64, type=int)
# parser.add_argument('-set_seed', default=1, type=int)
# parser.add_argument('-dset_seed', default=5, type=int)
# parser.add_argument('-shuffle', action='store_true')
# parser.add_argument('-num_workers', default=4, type=int)
#
# parser.add_argument('-alpha', default=0.2, type=float)
# parser.add_argument('-beta', default=0.1, type=float)
# parser.add_argument('-gamma', default=5, type=float)
# parser.add_argument('-sp_frac', default=0.5, type=float)
# parser.add_argument('-distill_frac', default=1, type=float)
# parser.add_argument('-T', default=2, type=float)
#
# parser.add_argument('-result_path', default='', type=str)
# parser.add_argument('-debug', action='store_true')
# parser.add_argument('-save_model', action='store_true')
# parser.add_argument('--is_il', default=False, type=bool)

opt = parser.parse_args()

if __name__ == '__main__':

    # 创建SummaryWriter
    log_path='/logs_2024_12_25_merak_scm_compare/AMFormer_onlyKAN_kan_args_origin_lr1e-4'
    writer = SummaryWriter('sw_log'+log_path)
    logger = create_logger('txt_log'+log_path)

    # 文件路径
    file_path = r"D:\DeepLearning\Datasets\Merak\dataset_1105.csv"

    # 读取文件并创建数据集
    # Merak:dataset_1105
    # data_utils = Data_loader(file_path)
    # scm20d
    data = Data_loader_scm(r'D:\DeepLearning\demain_il_for_eda-master\data_processed\scm20d_data.csv',
                         r'D:\DeepLearning\demain_il_for_eda-master\data_processed\scm20d_label.csv')

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(data.X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(data.y_train_scaled, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(data.X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(data.y_test_scaled, dtype=torch.float32).view(-1, 1)

    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 输入参数大小
    input_size = data.X_train_scaled.shape[1]

    # 创建模型实例
    # AMFormer
    model = AMTransformer.make_default(
            n_num_features=input_size,
            cat_cardinalities=[],
            token_dim=32,
            out_dim=1)
    my_model = Model(model,train_loader,test_loader)

    logger.info('------Begin Training Model------')
    # 训练模型
    epochs = 800
    best_test_loss=1e9
    best_epoch=0
    for epoch in range(epochs):
        # 训练模型
        my_model.train()
        # 记录训练损失
        writer.add_scalar('train_loss', my_model.average_loss, epoch)
        # 评估模型
        my_model.eval()
        # 记录测试损失
        writer.add_scalar('test_loss',my_model.test_loss,epoch)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {my_model.average_loss}, Test Loss: {my_model.test_loss}')
        if(my_model.test_loss<=best_test_loss):
            best_test_loss=my_model.test_loss
            best_epoch=epoch
    writer.close()
    logger.info(f'Best Epoch {best_epoch + 1}, Best Test Loss: {best_test_loss}')
    # my_model.model.cpu()  # 将模型转移到CPU
    # torch.save(my_model.model.state_dict(), './param/kan_model_weights.pth')
