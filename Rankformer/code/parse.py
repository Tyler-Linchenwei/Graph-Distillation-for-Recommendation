import torch
import numpy as np
import random
import argparse
import os
# 设置并行计算线程数，优化性能（针对多线程库如OpenMP、MKL、OpenBLAS）
os.environ["OMP_NUM_THREADS"] = "10"  # 设置OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "10"  # 设置MKL线程数
os.environ["OPENBLAS_NUM_THREADS"] = "10"  # 设置OpenBLAS线程数


"""
parse.py - Rankformer模型的参数配置文件

该文件负责解析命令行参数，设置随机种子，配置设备，并打印模型设置信息。
包含以下主要部分：
1. 参数解析：定义并解析所有模型相关参数
2. 设备设置：自动检测并配置GPU/CPU设备
3. 随机种子设置：确保实验结果可复现
4. 设置信息打印：输出所有配置参数，便于实验记录和调试
"""


def parse_args():
    """解析命令行参数，定义所有模型配置参数
    
    Returns:
        args: 包含所有配置参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="Rankformer模型参数配置")
    
    # ========================= 基础参数 =========================
    parser.add_argument('--seed', type=int, default=12345, help="随机种子，用于控制实验的可复现性")
    parser.add_argument('--device', type=int, default=0, help="GPU设备编号，若为-1则使用CPU")
    parser.add_argument('--hidden_dim', type=int, default=64, help="用户和物品嵌入向量的隐藏维度")
    
    # ========================= GCN相关参数 =========================
    parser.add_argument('--use_gcn', action='store_true', help="是否在模型中使用图卷积网络(GCN)层")
    parser.add_argument('--gcn_layers', type=int, default=1, help="GCN的层数")
    parser.add_argument('--gcn_left', type=float, default=1.0, help="GCN中左侧权重系数，用于平衡原始特征和传播特征")
    parser.add_argument('--gcn_right', type=float, default=0.0, help="GCN中右侧权重系数，用于控制剩余连接的权重")
    parser.add_argument('--gcn_mean', action='store_true', help="是否对多层GCN的输出进行平均融合，否则只使用最后一层的输出")
    
    # ========================= Rankformer相关参数 =========================
    parser.add_argument('--use_rankformer', action='store_true', help="是否在模型中使用Rankformer层")
    parser.add_argument('--rankformer_layers', type=int, default=1, help="Rankformer的层数")
    parser.add_argument('--rankformer_tau', type=float, default=0.5, help="Rankformer中更新权重的系数，控制当前层与上一层输出的融合比例")
    parser.add_argument('--rankformer_alpha', type=float, default=2, help="正负样本平衡系数，用于控制正负样本对的相对重要性")
    parser.add_argument('--rankformer_clamp_value', type=float, default=0, help="防止数值不稳定的截断值，当计算结果小于该值时被截断")
    
    # ========================= 对比学习(CL)相关参数 =========================
    parser.add_argument('--use_cl', action='store_true', help="是否在模型中使用对比学习损失")
    parser.add_argument('--cl_layer', type=int, default=1, help="应用对比学习的层索引")
    parser.add_argument('--cl_lambda', type=float, default=0.05, help="对比学习损失在总损失中的权重系数")
    parser.add_argument('--cl_eps', type=float, default=0.1, help="对比学习中添加的噪声强度，用于生成正样本对")
    parser.add_argument('--cl_tau', type=float, default=0.15, help="对比学习中的温度参数，控制softmax分布的平滑程度")
    
    # ========================= 训练相关参数 =========================
    parser.add_argument('--learning_rate', type=float, default=1e-1, help="优化器的学习率")
    parser.add_argument('--reg_lambda', type=float, default=1e-4, help="L2正则化系数，防止模型过拟合")
    parser.add_argument('--loss_batch_size', type=int, default=0, help="损失计算的批次大小，0表示全量计算")
    parser.add_argument('--max_epochs', type=int, default=2000, help="最大训练轮数")
    parser.add_argument('--show_loss_interval', type=int, default=1, help="每多少轮打印一次训练损失")
    
    # ========================= 测试相关参数 =========================
    parser.add_argument('--topks', type=str, default='[20]', help="评估指标的top-k值，如[20, 50]表示计算Recall@20和Recall@50")
    parser.add_argument('--test_batch_size', type=int, default=1000, help="测试时的批次大小，控制内存使用")
    parser.add_argument('--valid_interval', type=int, default=20, help="每多少轮进行一次验证")
    parser.add_argument('--stopping_step', type=int, default=10, help="早停机制的停止步数，若连续stopping_step轮验证指标未提升则停止训练")
    
    # ========================= 数据相关参数 =========================
    parser.add_argument('--data', type=str, default="Ali-Display", help="数据集名称，用于指定数据路径")
    
    # ========================= 实验设置相关参数 =========================
    parser.add_argument('--del_neg', action='store_true', help="是否在Rankformer中删除负样本对的影响")
    parser.add_argument('--del_benchmark', action='store_true', help="是否在Rankformer中删除基准分数项")
    parser.add_argument('--del_omega_norm', action='store_true', help="是否在Rankformer中删除Omega权重的归一化")
    parser.add_argument('--save_emb', action='store_true', help="是否保存训练后的用户和物品嵌入")
    parser.add_argument('--load_emb', action='store_true', help="是否加载预训练的用户和物品嵌入")
    
    return parser.parse_args()


args = parse_args()
args.topks = eval(args.topks)  # 将字符串形式的列表转换为实际列表

# ========================= 设备设置 =========================
# 自动检测是否有可用的GPU，并根据设备编号设置计算设备
if torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.device:d}')  # 设置为指定的GPU设备
    print(f"Running on GPU: {args.device}")
else:
    args.device = torch.device('cpu')  # 若没有GPU可用，则使用CPU
    print("GPU not found. Running on CPU.")

# ========================= 随机种子设置 =========================
# 设置所有相关库的随机种子，确保实验结果的可复现性
if args.seed != -1:
    torch.manual_seed(args.seed)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU的随机种子
    random.seed(args.seed)  # 设置Python内置random库的随机种子
    np.random.seed(args.seed)  # 设置NumPy的随机种子
    torch.backends.cudnn.deterministic = True  # 启用CUDA的确定性算法，避免随机性
    print(f'seed: {args.seed:d}')

print('Using', args.device)

# ========================= 打印配置信息 =========================
# 打印模型结构相关的设置
print('\n' + '='*50)
print('Model Setting')
print('-'*50)
print(f'    hidden dim: {args.hidden_dim:d}')
if args.use_gcn:
    print(f'    Using {args.gcn_layers:d} layers GCN.')
    print(f'      gcn left = {args.gcn_left:f}')
    print(f'      gcn right = {args.gcn_right:f}')
    if args.gcn_mean:
        print(f'      Z = Mean( Z(0~{args.gcn_layers:d}) )  # 对多层GCN输出进行平均')
    else:
        print(f'      Z = Z({args.gcn_layers:d})  # 只使用最后一层GCN的输出')
if args.use_rankformer:
    print(f'    Using {args.rankformer_layers:d} layers Rankformer:')
    print(f'      rankformer alpha = {args.rankformer_alpha:f}  # 正负样本平衡系数')
    print(f'      rankformer tau = {args.rankformer_tau:f}  # 更新权重系数')
    print(f'      rankformer clamp value = {args.rankformer_clamp_value:f}  # 数值稳定性截断值')
if args.use_cl:
    print(f'    Using CL Loss:')
    print(f'      cl layer: {args.cl_layer:d}  # 应用对比学习的层索引')
    print(f'      cl lambda: {args.cl_lambda:f}  # 对比学习损失权重')
    print(f'      cl eps: {args.cl_eps:f}  # 噪声强度')
    print(f'      cl tau: {args.cl_tau:f}  # 温度参数')

# 打印训练相关的设置
print('\nTrain Setting')
print('-'*50)
print(f'    learning rate: {args.learning_rate:f}  # 优化器学习率')
print(f'    reg_lambda: {args.reg_lambda:f}  # L2正则化系数')
print(f'    loss batch size: {args.loss_batch_size:d}  # 损失计算批次大小(0表示全量)')
print(f'    max epochs: {args.max_epochs:d}  # 最大训练轮数')
print(f'    show loss interval: {args.show_loss_interval:d}  # 显示损失的间隔轮数')

# 打印测试相关的设置
print('\nTest Setting')
print('-'*50)
print(f'    topks: {args.topks}  # 评估指标的top-k值')
print(f'    test batch size: {args.test_batch_size:d}  # 测试批次大小')
print(f'    valid interval: {args.valid_interval:d}  # 验证间隔轮数')
print(f'    stopping step: {args.stopping_step:d}  # 早停机制步数')

# 打印数据相关的设置
print('\nData Setting')
print('-'*50)
args.data_dir = "data/"  # 数据根目录
args.train_file = os.path.join(args.data_dir, args.data, f'train.txt')  # 训练数据路径
args.valid_file = os.path.join(args.data_dir, args.data, f'valid.txt')  # 验证数据路径
args.test_file = os.path.join(args.data_dir, args.data, f'test.txt')  # 测试数据路径
print(f'    data: {args.data:s}')
print(f'    train: {args.train_file:s}')
print(f'    valid: {args.valid_file:s}')
print(f'    test: {args.test_file:s}')

# 打印实验设置，用于消融研究
print('\nExperiment Setting')
print('-'*50)
print(f'    |                   Ablation Study Setting                 |')
print(f'    | Negative pairs | Benchmark | Offset | Normalize of Omega |')
print(f'    |        {"N" if args.del_neg else "Y"}       |     {"N" if args.del_benchmark else "Y"}     |   {"N" if args.rankformer_alpha==0 else "Y"}    |          {"N" if args.del_omega_norm else "Y"}         |')
if args.rankformer_alpha < 2:
    print('    Warning: Setting args.rankformer_alpha < 2 may violate some assumptions of the code.')
    print('             If the experimental results do not converge, please try increasing args.rankformer_clamp_value.')
if args.save_emb or args.load_emb:
    args.user_emb_path = f'saved/{args.data:s}_user.pt'  # 用户嵌入保存路径
    args.item_emb_path = f'saved/{args.data:s}_item.pt'  # 物品嵌入保存路径
    if args.save_emb:
        print(f'    Initial features will be saved to: {args.user_emb_path} and {args.item_emb_path}')
    if args.load_emb:
        print(f'    Initial features will be loaded from: {args.user_emb_path} and {args.item_emb_path}')

print('='*50 + '\n')
