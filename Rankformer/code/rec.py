import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import args


def sparse_sum(values, indices0, indices1, n):
    """稀疏求和函数 - 实现高效的分组聚合操作
    
    该函数用于将values中的元素根据indices1的分组信息进行聚合求和，类似于稀疏矩阵的行求和
    当提供indices0时，可以实现从源索引到目标索引的映射聚合
    
    数学公式：
        result[i] = sum_{j: indices1[j] = i} values[indices0[j]]  (如果indices0不为None)
        result[i] = sum_{j: indices1[j] = i} values[j]          (如果indices0为None)
    
    Args:
        values: 要聚合的数值张量，形状为[num_elements, ...]或[num_entities, ...]
        indices0: 源索引（可选），形状为[num_elements]，用于从values中选择元素
        indices1: 目标索引，形状为[num_elements]，指定每个元素应该聚合到哪个目标位置
        n: 目标维度大小，即结果张量的第一维大小
        
    Returns:
        result: 聚合后的张量，形状为[n, ...]，其中result[i]是所有目标索引为i的元素的和
    """
    # 参数验证
    if indices0 is None:
        # 直接聚合values中的元素到indices1指定的位置
        assert len(indices1.shape) == 1 and values.shape[0] == indices1.shape[0]
    else:
        # 通过indices0从values中选择元素，然后聚合到indices1指定的位置
        assert len(indices0.shape) == 1 and len(indices1.shape) == 1
        assert indices0.shape[0] == indices1.shape[0]
    
    # 创建结果张量并使用index_add进行高效聚合
    # 1. 创建形状为[n, ...]的全零张量
    # 2. 使用index_add将values或values[indices0]聚合到indices1指定的位置
    return torch.zeros(
        [n] + list(values.shape)[1:],  # 结果形状：[n, ...]
        device=values.device,          # 保持与输入相同的设备
        dtype=values.dtype             # 保持与输入相同的数据类型
    ).index_add(0, indices1, values if indices0 is None else values[indices0])


def rest_sum(values, indices0, indices1, n):
    """计算剩余求和（总和减去稀疏求和）
    
    该函数用于计算"非分组"元素的总和，即所有元素的总和减去通过sparse_sum聚合的部分
    常用于计算负样本的聚合（所有样本减去正样本）
    
    数学公式：
        result[i] = sum(values) - sum_{j: indices1[j] = i} values[indices0[j]]
    
    Args:
        values: 要聚合的数值张量，形状为[num_entities, ...]
        indices0: 源索引，形状为[num_elements]，用于从values中选择正样本元素
        indices1: 目标索引，形状为[num_elements]，指定正样本的目标位置
        n: 目标维度大小
        
    Returns:
        result: 剩余聚合后的张量，形状为[n, ...]
    """
    # 计算所有元素的总和（在第一个维度上求和）
    total_sum = values.sum(0).unsqueeze(0)  # 形状从[...,]变为[1, ...]
    
    # 计算正样本的稀疏求和
    positive_sum = sparse_sum(values, indices0, indices1, n)
    
    # 剩余部分 = 总和 - 正样本和
    return total_sum - positive_sum


class GCN(nn.Module):
    """图卷积网络（Graph Convolutional Network）类
    
    实现了基于用户-物品交互图的图卷积操作，用于学习用户和物品的嵌入表示。
    该实现采用了对称的图卷积结构，支持用户和物品的双向信息传播。
    """
    def __init__(self, dataset, alpha=1.0, beta=0.0):
        """初始化GCN层
        
        Args:
            dataset: 数据集对象，包含用户数和物品数等元数据
            alpha: GCN左参数，控制用户度数的影响权重
            beta: GCN右参数，控制物品度数的影响权重
        """
        super(GCN, self).__init__()
        self.dataset = dataset
        self.alpha = alpha  # 用户度数的幂次参数
        self.beta = beta    # 物品度数的幂次参数

    def forward(self, x, u, i):
        """GCN层前向传播 - 执行图卷积操作
        
        实现了对称的图卷积传播，公式如下：
        zu_i = sum_{j∈N(u_i)} (xi_j / (du_i^α * di_j^β))
        zi_j = sum_{i∈N(i_j)} (xu_i / (du_i^β * di_j^α))
        
        其中：
        - N(u_i)是用户u_i交互过的物品集合
        - N(i_j)是物品i_j被交互过的用户集合
        - du_i是用户u_i的度数（交互物品数）
        - di_j是物品i_j的度数（被交互次数）
        
        Args:
            x: 用户和物品的嵌入矩阵，形状为[num_users+num_items, hidden_dim]
            u: 训练数据中的用户索引，形状为[num_interactions]
            i: 训练数据中的物品索引，形状为[num_interactions]
            
        Returns:
            updated_emb: 更新后的用户和物品嵌入矩阵，形状为[num_users+num_items, hidden_dim]
        """
        # 获取用户数和物品数
        n, m = self.dataset.num_users, self.dataset.num_items
        
        # 计算每个用户和物品的度数（交互次数）
        # du: [num_users] - 每个用户的交互物品数
        du = sparse_sum(torch.ones_like(u), None, u, n).clamp(1)  # clamp(1)确保度数至少为1
        # di: [num_items] - 每个物品的被交互次数
        di = sparse_sum(torch.ones_like(u), None, i, m).clamp(1)  # clamp(1)避免除零错误
        
        # 计算图卷积的权重
        # w1: [num_interactions, 1] - 用户→物品传播的权重
        w1 = (torch.ones_like(u) / du[u].pow(self.alpha) / di[i].pow(self.beta)).unsqueeze(-1)
        # w2: [num_interactions, 1] - 物品→用户传播的权重
        w2 = (torch.ones_like(u) / du[u].pow(self.beta) / di[i].pow(self.alpha)).unsqueeze(-1)
        
        # 将嵌入分为用户和物品两部分
        # xu: [num_users, hidden_dim] - 用户嵌入
        # xi: [num_items, hidden_dim] - 物品嵌入
        xu, xi = torch.split(x, [n, m])
        
        # 执行图卷积传播
        # zu: [num_users, hidden_dim] - 更新后的用户嵌入
        zu = sparse_sum(xi[i] * w1, None, u, n)  # 物品信息传播给用户
        # zi: [num_items, hidden_dim] - 更新后的物品嵌入
        zi = sparse_sum(xu[u] * w2, None, i, m)  # 用户信息传播给物品
        
        # 合并更新后的用户和物品嵌入
        return torch.concat([zu, zi], 0)


class Rankformer(nn.Module):
    """Rankformer层实现
    
    基于排序目标的图Transformer层，同时考虑正样本和负样本的聚合。
    该层设计用于直接优化排序性能，通过对比正样本和负样本来学习更好的嵌入表示。
    
    核心思想：
    1. 同时聚合正样本（用户交互过的物品）和负样本（用户未交互过的物品）
    2. 使用注意力机制来权衡不同样本的重要性
    3. 直接优化排序目标，而非传统的点积相似度
    """
    def __init__(self, dataset, alpha):
        """初始化Rankformer层
        
        Args:
            dataset: 数据集对象，包含用户数和物品数等元数据
            alpha: 正负样本平衡系数，控制正负样本的权重比例
        """
        super(Rankformer, self).__init__()
        self.dataset = dataset
        self.my_parameters = []  # 模型参数列表
        self.alpha = alpha      # 正负样本平衡参数

    def forward(self, x, u, i):
        """Rankformer层前向传播 - 执行基于排序的图Transformer操作
        
        实现了基于正负样本聚合的注意力传播机制，公式复杂，核心步骤如下：
        1. 计算正样本和负样本的聚合嵌入
        2. 计算基准分数作为注意力的参考
        3. 计算注意力权重并进行信息聚合
        4. 合并正负样本的结果得到最终嵌入
        
        Args:
            x: 用户和物品的嵌入矩阵，形状为[num_users+num_items, hidden_dim]
            u: 训练数据中的用户索引，形状为[num_interactions]
            i: 训练数据中的物品索引，形状为[num_interactions]
            
        Returns:
            updated_emb: 更新后的用户和物品嵌入矩阵，形状为[num_users+num_items, hidden_dim]
        """
        # 获取用户数和物品数
        n, m = self.dataset.num_users, self.dataset.num_items  # 用户数和物品数
        
        # ========== 步骤1：计算正负样本数量 ==========
        # 计算每个用户交互过的物品数量(dui)和未交互过的物品数量(duj)
        dui = sparse_sum(torch.ones_like(u), None, u, n)  # [num_users] - 每个用户的正样本数
        duj = m - dui  # [num_users] - 每个用户的负样本数
        # 确保数量至少为1，并扩展维度以便后续计算
        dui, duj = dui.clamp(1).unsqueeze(1), duj.clamp(1).unsqueeze(1)
        
        # ========== 步骤2：准备嵌入向量 ==========
        # 分割嵌入向量为用户和物品两部分
        # xu, xi: 归一化后的嵌入，用于计算注意力权重
        # vu, vi: 原始嵌入，用于信息传递和更新
        xu, xi = torch.split(F.normalize(x), [n, m])  # 归一化后的嵌入
        vu, vi = torch.split(x, [n, m])              # 原始嵌入
        
        # ========== 步骤3：计算用户-物品交互分数 ==========
        # xui: [num_interactions, 1] - 用户u和物品i的交互分数（归一化嵌入的点积）
        xui = (xu[u] * xi[i]).sum(1).unsqueeze(1)
        
        # ========== 步骤4：正样本聚合 ==========
        # 使用sparse_sum计算用户交互过的物品嵌入的聚合
        sxi = sparse_sum(xi, i, u, n)  # [num_users, hidden_dim] - 正样本物品的归一化嵌入聚合
        svi = sparse_sum(vi, i, u, n)  # [num_users, hidden_dim] - 正样本物品的原始嵌入聚合
        
        # ========== 步骤5：负样本聚合 ==========
        # 计算用户未交互过的物品嵌入的聚合（通过总和减去正样本聚合）
        sxj = xi.sum(0) - sxi  # [hidden_dim] - 负样本物品的归一化嵌入聚合（总和-正样本）
        svj = vi.sum(0) - svi  # [hidden_dim] - 负样本物品的原始嵌入聚合（总和-正样本）
        
        # ========== 步骤6：计算基准分数 ==========
        # 基准分数表示用户对正/负样本的平均偏好，用于注意力计算
        b_pos = (xu * sxi).sum(1).unsqueeze(1) / dui  # [num_users, 1] - 正样本的平均分数
        b_neg = (xu * sxj).sum(1).unsqueeze(1) / duj  # [num_users, 1] - 负样本的平均分数
        
        # 根据参数决定是否使用基准分数
        if args.del_benchmark:
            b_pos, b_neg = 0, 0
        
        # ========== 步骤7：计算交叉项 ==========
        # 计算物品之间的外积，用于后续的注意力聚合
        xxi = xi.unsqueeze(1) * xi.unsqueeze(2)  # [num_items, hidden_dim, hidden_dim] - 物品-物品归一化嵌入外积
        xvi = xi.unsqueeze(1) * vi.unsqueeze(2)  # [num_items, hidden_dim, hidden_dim] - 物品归一化嵌入与原始嵌入外积
        
        # ========== 步骤8：计算注意力权重分母 ==========
        # 计算注意力权重的分母部分，用于归一化注意力分数
        du1 = (xu * sxi).sum(1).unsqueeze(1) / dui - b_neg + self.alpha  # 用户侧正样本分母
        du2 = -(xu * sxj).sum(1).unsqueeze(1) / duj + b_pos + self.alpha  # 用户侧负样本分母
        
        # 物品侧的分母计算，考虑了用户的分布
        di1 = (xi * sparse_sum(xu / dui, u, i, m)).sum(1).unsqueeze(1) + sparse_sum((-b_neg + self.alpha) / dui, u, i, m)
        di2 = -(xi * rest_sum(xu / duj, u, i, m)).sum(1).unsqueeze(1) + rest_sum((b_pos + self.alpha) / duj, u, i, m)
        
        # ========== 步骤9：注意力计算核心 ==========
        # 通过注意力机制聚合正样本的信息
        A = sparse_sum(xui * vi[i], None, u, n)  # [num_users, hidden_dim] - 正样本的注意力聚合
        
        # ========== 步骤10：计算更新后的嵌入 ==========
        # 用户嵌入更新 - 正样本部分
        zu1 = A / dui - svi * (b_neg - self.alpha) / dui
        
        # 用户嵌入更新 - 负样本部分
        zu2 = (torch.mm(xu, (xvi).sum(0)) - A) / duj - svj * (b_pos + self.alpha) / duj
        
        # 物品嵌入更新 - 正样本部分
        zi1 = sparse_sum(xui * vu[u] / dui[u], None, i, m) - sparse_sum(vu * (b_neg - self.alpha) / dui, u, i, m)
        
        # 物品嵌入更新 - 负样本部分
        zi2 = torch.mm(xi, ((xu / duj).unsqueeze(2) * vu.unsqueeze(1)).sum(0)) \
            - sparse_sum(xui * (vu / duj)[u], None, i, m) \
            - rest_sum(vu * (b_pos + self.alpha) / duj, u, i, m)
        
        # ========== 步骤11：合并结果 ==========
        # 合并正样本和负样本的结果
        z1 = torch.concat([zu1, zi1], 0)  # 正样本的最终嵌入
        z2 = torch.concat([zu2, zi2], 0)  # 负样本的最终嵌入
        d1 = torch.concat([du1, di1], 0).clamp(args.rankformer_clamp_value)  # 正样本的分母（限制最小值避免除零）
        d2 = torch.concat([du2, di2], 0).clamp(args.rankformer_clamp_value)  # 负样本的分母（限制最小值避免除零）
        
        # 根据参数决定是否使用负样本
        if args.del_neg:
            z2, d2 = 0, 0
        
        # 计算最终的嵌入更新
        z, d = z1 + z2, d1 + d2
        
        # 根据参数决定是否使用归一化
        if args.del_omega_norm:
            return z
        
        return z / d
