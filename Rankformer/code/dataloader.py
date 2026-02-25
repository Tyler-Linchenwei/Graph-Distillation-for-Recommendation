import torch
from torch.utils.data import Dataset
import pandas as pd
from parse import args
import numpy as np
import torch.nn.functional as F


class MyDataset(Dataset):
    """自定义数据集类，用于加载和处理推荐系统数据
    
    加载训练、验证和测试数据集，统计用户和物品数量，并构建批次数据
    """
    def __init__(self, train_file, valid_file, test_file, device):
        """初始化数据集
        
        Args:
            train_file: 训练数据文件路径
            valid_file: 验证数据文件路径
            test_file: 测试数据文件路径
            device: 数据加载的目标设备（CPU或GPU）
        """
        self.device = device
        
        # ========== 训练数据加载与预处理 ==========
        train_data = pd.read_table(train_file, header=None, sep=' ')  # 读取训练数据
        train_data = torch.from_numpy(train_data.values).to(self.device)  # 转换为Tensor并移动到指定设备
        self.train_data = train_data[torch.argsort(train_data[:, 0]), :]  # 按用户ID排序
        self.train_user, self.train_item = self.train_data[:, 0], self.train_data[:, 1]  # 分割用户和物品ID
        
        # ========== 验证数据加载与预处理 ==========
        valid_data = pd.read_table(valid_file, header=None, sep=' ')  # 读取验证数据
        valid_data = torch.from_numpy(valid_data.values).to(self.device)  # 转换为Tensor并移动到指定设备
        self.valid_data = valid_data[torch.argsort(valid_data[:, 0]), :]  # 按用户ID排序
        self.valid_user, self.valid_item = self.valid_data[:, 0], self.valid_data[:, 1]  # 分割用户和物品ID
        
        # ========== 测试数据加载与预处理 ==========
        test_data = pd.read_table(test_file, header=None, sep=' ')  # 读取测试数据
        test_data = torch.from_numpy(test_data.values).to(self.device)  # 转换为Tensor并移动到指定设备
        self.test_data = test_data[torch.argsort(test_data[:, 0]), :]  # 按用户ID排序
        self.test_user, self.test_item = self.test_data[:, 0], self.test_data[:, 1]  # 分割用户和物品ID
        
        # ========== 统计用户和物品数量 ==========
        self.num_users = max(self.train_user.max(), self.valid_user.max(), self.test_user.max()).cpu() + 1
        self.num_items = max(self.train_item.max(), self.valid_item.max(), self.test_item.max()).cpu() + 1
        self.num_nodes = self.num_users + self.num_items  # 总节点数（用户+物品）
        
        # 打印数据统计信息
        print(f'{self.num_users:d} users, {self.num_items:d} items.')
        print(f'train: {self.train_user.shape[0]:d}, valid: {self.valid_user.shape[0]:d}, test: {self.test_user.shape[0]:d}.')
        
        # 构建批次数据
        self.build_batch()

    def build_batch(self):
        """构建批次数据，用于模型训练和评估
        
        计算每个用户的交互次数，构建用户批次和对应的数据批次
        """
        # 计算每个用户的训练交互次数
        self.train_degree = torch.zeros(self.num_users).long().to(args.device)
        self.train_degree = self.train_degree.index_add(0, self.train_user, torch.ones_like(self.train_user))
        
        # 计算每个用户的测试交互次数
        self.test_degree = torch.zeros(self.num_users).long().to(args.device)
        self.test_degree = self.test_degree.index_add(0, self.test_user, torch.ones_like(self.test_user))
        
        # 计算每个用户的验证交互次数
        self.valid_degree = torch.zeros(self.num_users).long().to(args.device)
        self.valid_degree = self.valid_degree.index_add(0, self.valid_user, torch.ones_like(self.valid_user))
        
        # 构建用户批次（按test_batch_size分割）
        self.batch_users = [torch.arange(i, min(i+args.test_batch_size, self.num_users)).to(args.device) 
                          for i in range(0, self.num_users, args.test_batch_size)]
        
        # 构建训练数据批次（根据每个用户批次的交互次数分割）
        self.train_batch = list(self.train_data.split(
            [self.train_degree[batch_user].sum() for batch_user in self.batch_users]))
        
        # 构建测试数据批次（根据每个用户批次的交互次数分割）
        self.test_batch = list(self.test_data.split(
            [self.test_degree[batch_user].sum() for batch_user in self.batch_users]))
        
        # 构建验证数据批次（根据每个用户批次的交互次数分割）
        self.valid_batch = list(self.valid_data.split(
            [self.valid_degree[batch_user].sum() for batch_user in self.batch_users]))


# 全局数据集实例
dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
