from parse import args
import torch
from torch import nn
import torch.nn.functional as F
import rec
from rec import GCN, Rankformer
import numpy as np


def InfoNCE(x, y, tau=0.15, b_cos=True):
    """实现InfoNCE对比学习损失
    
    Args:
        x: 第一组嵌入向量
        y: 第二组嵌入向量
        tau: 温度参数，控制分布平滑程度
        b_cos: 是否对输入向量进行余弦归一化
        
    Returns:
        InfoNCE损失值
    """
    if b_cos:
        x, y = F.normalize(x), F.normalize(y)
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()


def test(pred, test, recall_n):
    """计算推荐系统评价指标
    
    Args:
        pred: 预测的物品ID矩阵
        test: 测试集中的真实物品ID矩阵
        recall_n: 每个用户的测试集物品数量
        
    Returns:
        样本数量、precision列表、recall列表、ndcg列表
    """
    pred = torch.isin(pred[recall_n > 0], test)
    recall_n = recall_n[recall_n > 0]
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        right_pred = pred[:, :k].sum(1)
        recall_k = recall_n.clamp(max=k)
        # precision
        pre.append((right_pred/k).sum())
        # recall
        recall.append((right_pred/recall_k).sum())
        # ndcg
        dcg = (pred[:, :k]/torch.arange(2, k+2).to(args.device).unsqueeze(0).log2()).sum(1)
        d_val = (1/torch.arange(2, k+2).to(args.device).log2()).cumsum(0)
        idcg = d_val[recall_k-1]
        ndcg.append((dcg / idcg).sum())
    return recall_n.shape[0], torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg)


def multi_negative_sampling(u, i, m, k):
    """批量负采样函数，确保生成的负样本不在正样本中
    
    该函数实现了高效的负采样机制，用于推荐系统中的BPR损失计算。
    核心思想是：生成随机负样本，并确保它们不是用户的历史交互物品（正样本）
    
    数学原理：
    - 每个用户u的正样本i必须对应k个负样本j，满足j不在u的交互历史中
    - 使用edge_id = u * m + i将(user, item)对唯一映射为整数，便于快速查找
    
    Args:
        u: 用户ID张量，形状为[batch_size]，包含批次中所有用户的ID
        i: 正样本物品ID张量，形状为[batch_size]，包含与用户交互过的物品ID
        m: 物品总数（整数），用于确定采样范围[0, m-1]
        k: 整数，每个正样本需要生成的负样本数量
        
    Returns:
        j: 负样本物品ID张量，形状为[batch_size, k]，每个正样本对应k个有效负样本
    """
    # 步骤1: 创建正样本的唯一边标识符
    # 计算公式: edge_id = user_id * 物品总数 + item_id
    # 目的: 将(user, item)对唯一编码为整数，便于O(1)时间复杂度的成员检查
    edge_id = u * m + i
    
    # 步骤2: 随机生成初始负样本矩阵
    # 生成形状为[batch_size, k]的随机物品ID，取值范围[0, m-1]
    # 确保负样本与输入数据在同一设备上（GPU/CPU）
    j = torch.randint(0, m, (i.shape[0], k), device=u.device)
    
    # 步骤3: 检查并标记无效负样本
    # 计算负样本的edge_id: u.unsqueeze(1)扩展为[batch_size, 1]，与[j]广播相乘
    # 使用torch.isin()检查每个负样本是否是用户的历史交互物品（即是否在正样本edge_id中）
    # mask: 布尔矩阵，形状[batch_size, k]，True表示对应位置的负样本无效（是正样本）
    mask = torch.isin(u.unsqueeze(1) * m + j, edge_id)
    
    # 步骤4: 迭代替换无效负样本
    # 只要还有无效负样本（mask中True的数量>0），就继续循环
    while mask.sum() > 0:
        # 对mask为True的位置重新生成随机物品ID
        j[mask] = torch.randint_like(j[mask], 0, m)
        # 重新计算所有负样本的有效性
        mask = torch.isin(u.unsqueeze(1) * m + j, edge_id)
    
    # 返回最终的有效负样本矩阵
    return j


def negative_sampling(u, i, m):
    """负采样函数，确保负样本不在正样本中
    
    Args:
        u: 用户ID
        i: 正样本物品ID
        m: 物品总数
        
    Returns:
        负样本物品ID
    """
    edge_id = u*m+i
    j = torch.randint_like(i, 0, m)
    mask = torch.isin(u*m+j, edge_id)
    while mask.sum() > 0:
        j[mask] = torch.randint_like(j[mask], 0, m)
        mask = torch.isin(u*m+j, edge_id)
    return j


class Model(nn.Module):
    """Rankformer推荐模型主类
    
    结合GCN和Rankformer层实现基于图的推荐系统
    """
    def __init__(self, dataset):
        """初始化模型
        
        Args:
            dataset: 数据集对象，包含用户和物品信息
        """
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = args.hidden_dim
        
        # 用户和物品嵌入层
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        
        # 初始化嵌入权重
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # 模型参数配置
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]
        self.layers = []
        
        # GCN和Rankformer组件
        self.GCN = GCN(dataset, args.gcn_left, args.gcn_right)
        self.Rankformer = Rankformer(dataset, args.rankformer_alpha)
        
        # 保存计算得到的嵌入
        self._users, self._items, self._users_cl, self._items_cl = None, None, None, None
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)
        self.loss_func = self.loss_bpr

    def computer(self):
        """计算用户和物品的最终嵌入向量
        
        该方法是模型的核心特征传播函数，负责结合GCN和Rankformer层
        对用户和物品的原始嵌入进行特征增强和表示学习。
        
        核心流程：
        1. 合并用户和物品的原始嵌入
        2. （可选）通过GCN层进行图卷积特征传播
        3. （可选）通过Rankformer层进行基于注意力的特征增强
        4. 保存最终嵌入和用于对比学习的中间嵌入
        
        注意：
        - 支持单独使用GCN或Rankformer，也支持两者结合
        - 对比学习噪声仅在指定的GCN层添加
        """
        # 步骤1: 准备输入数据
        # 获取训练集的用户-物品交互对
        u_train, i_train = self.dataset.train_user, self.dataset.train_item
        
        # 获取用户和物品的原始嵌入矩阵
        users_emb_original = self.embedding_user.weight  # [num_users, hidden_dim]
        items_emb_original = self.embedding_item.weight  # [num_items, hidden_dim]
        
        # 合并用户和物品嵌入，形成完整的节点嵌入矩阵
        # 顺序: [用户嵌入在前，物品嵌入在后]
        all_emb = torch.cat([users_emb_original, items_emb_original], dim=0)  # [num_users+num_items, hidden_dim]
        
        # 初始化用于对比学习的嵌入（默认使用原始嵌入）
        emb_cl = all_emb
        
        # 步骤2: GCN层特征传播（若启用）
        if args.use_gcn:
            # 保存各层GCN嵌入用于可选的平均聚合
            gcn_embs = [all_emb]
            
            # 执行指定层数的GCN传播
            for layer_idx in range(args.gcn_layers):
                # 通过GCN层进行一次特征传播
                all_emb = self.GCN(all_emb, u_train, i_train)
                
                # （可选）添加对比学习噪声
                if args.use_cl:
                    # 生成与嵌入同形状的随机噪声
                    random_noise = torch.rand_like(all_emb)
                    # 计算噪声的方向并归一化，然后乘以噪声强度参数
                    noise_direction = torch.sign(all_emb) * F.normalize(random_noise, dim=-1)
                    all_emb += noise_direction * args.cl_eps
                
                # 保存指定层的嵌入用于对比学习
                if layer_idx == args.cl_layer - 1:  # 注意：args.cl_layer是从1开始计数
                    emb_cl = all_emb
                
                # 保存当前层的嵌入
                gcn_embs.append(all_emb)
            
            # （可选）对GCN各层嵌入进行平均聚合
            if args.gcn_mean:
                # 将各层嵌入堆叠并在最后一维求平均
                all_emb = torch.stack(gcn_embs, dim=-1).mean(dim=-1)
        
        # 步骤3: Rankformer层特征增强（若启用）
        if args.use_rankformer:
            # 执行指定层数的Rankformer传播
            for _ in range(args.rankformer_layers):
                # 通过Rankformer层计算增强后的嵌入
                rec_emb = self.Rankformer(all_emb, u_train, i_train)
                # 使用残差连接更新嵌入：all_emb = (1-tau)*all_emb + tau*rec_emb
                all_emb = (1 - args.rankformer_tau) * all_emb + args.rankformer_tau * rec_emb
        
        # 步骤4: 分割并保存最终嵌入
        # 将合并的嵌入分割回用户嵌入和物品嵌入
        # 分割点: [num_users, num_items]
        self._users, self._items = torch.split(
            all_emb, 
            [self.dataset.num_users, self.dataset.num_items],
            dim=0
        )
        
        # 分割并保存用于对比学习的嵌入
        self._users_cl, self._items_cl = torch.split(
            emb_cl, 
            [self.dataset.num_users, self.dataset.num_items],
            dim=0
        )

    def evaluate(self, test_batch, test_degree):
        """评估模型性能
        
        该方法用于在验证集或测试集上评估模型的推荐性能，支持多个Top-K值的同时计算。
        评估指标包括Precision@K、Recall@K和NDCG@K。
        
        核心流程：
        1. 确保模型处于评估模式并计算好嵌入
        2. 对用户进行批次处理，避免内存溢出
        3. 计算用户-物品评分矩阵并排除训练集交互
        4. 获取Top-K推荐结果并计算评估指标
        5. 汇总并平均所有批次的评估结果
        
        Args:
            test_batch: 测试集批次列表，每个批次包含用户的测试物品ID
            test_degree: 每个用户的测试集物品数量张量，用于归一化指标
            
        Returns:
            all_pre: 各Top-K值的Precision列表
            all_recall: 各Top-K值的Recall列表
            all_ndcg: 各Top-K值的NDCG列表
        """
        # 步骤1: 准备评估环境
        # 设置模型为评估模式（关闭Dropout等训练专用层）
        self.eval()
        
        # 确保已经计算了最新的嵌入向量
        if self._users is None:
            self.computer()
        
        # 获取用户和物品的最终嵌入
        user_emb = self._users  # [num_users, hidden_dim]
        item_emb = self._items  # [num_items, hidden_dim]
        
        # 确定需要计算的最大Top-K值
        max_K = max(args.topks)
        
        # 初始化评估指标累加器
        all_pre = torch.zeros(len(args.topks), device=user_emb.device)
        all_recall = torch.zeros(len(args.topks), device=user_emb.device)
        all_ndcg = torch.zeros(len(args.topks), device=user_emb.device)
        all_cnt = 0  # 记录处理的用户总数
        
        # 步骤2: 批次处理用户并计算评估指标
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
            # 遍历所有用户批次
            for batch_users, batch_train, ground_true in zip(
                self.dataset.batch_users,  # 用户ID批次
                self.dataset.train_batch,   # 训练集交互批次
                test_batch                  # 测试集真实标签批次
            ):
                # 获取当前批次用户的嵌入向量
                user_e = user_emb[batch_users]  # [batch_size, hidden_dim]
                
                # 步骤3: 计算用户-物品评分矩阵
                # 使用点积计算用户对所有物品的评分
                rating = torch.mm(user_e, item_emb.t())  # [batch_size, num_items]
                
                # 步骤4: 排除训练集交互物品
                # 将训练集中的物品评分设置为负无穷，确保它们不会出现在推荐列表中
                # 计算批次内索引: batch_train[:, 0] - batch_users[0]
                rating[batch_train[:, 0] - batch_users[0], batch_train[:, 1]] = -(1 << 10)  # 约等于负无穷
                
                # 步骤5: 获取Top-K推荐结果
                # 对每个用户取评分最高的max_K个物品
                _, pred_items = torch.topk(rating, k=max_K, dim=1)  # [batch_size, max_K]
                
                # 步骤6: 计算评估指标
                # 将用户ID和物品ID组合成edge_id，便于与ground_true比较
                user_edge_ids = batch_users.unsqueeze(1) * self.dataset.num_items + pred_items
                true_edge_ids = ground_true[:, 0] * self.dataset.num_items + ground_true[:, 1]
                
                # 调用test函数计算当前批次的评估指标
                cnt, pre_batch, recall_batch, ndcg_batch = test(
                    user_edge_ids,        # 推荐结果的edge_id
                    true_edge_ids,        # 真实交互的edge_id
                    test_degree[batch_users]  # 每个用户的测试物品数量
                )
                
                # 步骤7: 累加批次指标
                all_pre += pre_batch
                all_recall += recall_batch
                all_ndcg += ndcg_batch
                all_cnt += cnt
            
            # 步骤8: 计算平均指标
            all_pre /= all_cnt
            all_recall /= all_cnt
            all_ndcg /= all_cnt
        
        # 返回平均后的评估指标
        return all_pre, all_recall, all_ndcg

    def valid_func(self):
        """验证集评估函数
        
        Returns:
            验证集上的precision、recall、ndcg
        """
        return self.evaluate(self.dataset.valid_batch, self.dataset.valid_degree)

    def test_func(self):
        """测试集评估函数
        
        Returns:
            测试集上的precision、recall、ndcg
        """
        return self.evaluate(self.dataset.test_batch, self.dataset.test_degree)

    def train_func(self):
        """模型训练主函数
        
        支持全量训练和批量训练
        
        Returns:
            平均训练损失
        """
        self.train()
        if args.loss_batch_size == 0:
            # 全量训练
            return self.train_func_one_batch(self.dataset.train_user, self.dataset.train_item)
        train_losses = []
        # 随机打乱训练数据
        shuffled_indices = torch.randperm(self.dataset.train_user.shape[0], device=args.device)
        train_user = self.dataset.train_user[shuffled_indices]
        train_item = self.dataset.train_item[shuffled_indices]
        # 批量训练
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            train_losses.append(self.train_func_one_batch(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size]))
        return torch.stack(train_losses).mean()

    def train_func_one_batch(self, u, i):
        """单批次训练函数
        
        Args:
            u: 用户ID
            i: 物品ID
            
        Returns:
            训练损失
        """
        self.computer()
        train_loss = self.loss_func(u, i)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        # memory_allocated = torch.cuda.max_memory_allocated(args.device)
        # print(f"Max memory allocated after backward pass: {memory_allocated} bytes = {memory_allocated/1024/1024:.4f} MB = {memory_allocated/1024/1024/1024:.4f} GB.")
        return train_loss

    def loss_bpr(self, u, i):
        """BPR损失函数实现（Bayesian Personalized Ranking）
        
        BPR损失是推荐系统中常用的排序损失，目标是让正样本的预测分数高于负样本。
        该实现支持基础BPR损失、L2正则化和对比学习损失的组合。
        
        数学公式：
        - BPR损失: E_{(u,i,j)} [ -ln(σ(score(u,i) - score(u,j))) ]
        - 正则化损失: (1/2) * λ * (||u_emb||² + ||i_emb||² + ||j_emb||²) / batch_size
        - 总损失: BPR_loss + λ * reg_loss + μ * cl_loss（若使用对比学习）
        
        Args:
            u: 用户ID张量，形状为[batch_size]，包含当前批次的用户ID
            i: 正样本物品ID张量，形状为[batch_size]，包含用户实际交互的物品ID
            
        Returns:
            loss_total: 总损失值张量，包含BPR损失、正则化损失和可选的对比学习损失
        """
        # 步骤1: 生成负样本
        # 调用negative_sampling函数，确保负样本j不在用户u的交互历史中
        j = negative_sampling(u, i, self.dataset.num_items)
        
        # 步骤2: 获取嵌入向量
        # u_emb0/i_emb0/j_emb0: 原始嵌入（用于正则化）
        # u_emb/i_emb/j_emb: GCN/Rankformer传播后的最终嵌入（用于评分计算）
        u_emb0, u_emb = self.embedding_user(u), self._users[u]
        i_emb0, i_emb = self.embedding_item(i), self._items[i]
        j_emb0, j_emb = self.embedding_item(j), self._items[j]
        
        # 步骤3: 计算用户-物品交互分数
        # 使用点积计算用户与物品的相似度分数
        # scores_ui: 正样本分数，形状为[batch_size]
        # scores_uj: 负样本分数，形状为[batch_size]
        scores_ui = torch.sum(torch.mul(u_emb, i_emb), dim=-1)
        scores_uj = torch.sum(torch.mul(u_emb, j_emb), dim=-1)
        
        # 步骤4: 计算BPR损失
        # 使用softplus函数实现稳定的BPR损失计算
        # softplus(x) = ln(1 + e^x)，当x < 0时近似为|x|，当x ≥ 0时近似为x
        # 目标: 最小化softplus(scores_uj - scores_ui)，等价于最大化scores_ui - scores_uj
        loss_bpr = torch.mean(F.softplus(scores_uj - scores_ui))
        
        # 步骤5: 计算L2正则化损失
        # 正则化仅作用于原始嵌入，防止过拟合
        # 计算每个嵌入的L2范数平方和，然后除以批次大小归一化
        reg_loss = (1/2) * (u_emb0.norm(2).pow(2) + i_emb0.norm(2).pow(2) + j_emb0.norm(2).pow(2)) / u.shape[0]
        
        # 步骤6: 计算对比学习损失（可选）
        if args.use_cl:
            # 获取所有用户和物品的最终嵌入及对比学习嵌入
            all_user_emb, all_item_emb = self._users, self._items
            cl_user_emb, cl_item_emb = self._users_cl, self._items_cl
            
            # 获取当前批次中唯一的用户和物品ID，避免重复计算
            u_unique = torch.unique(u)
            i_unique = torch.unique(i)
            
            # 计算用户和物品的对比学习损失
            cl_loss_user = InfoNCE(all_user_emb[u_unique], cl_user_emb[u_unique], args.cl_tau)
            cl_loss_item = InfoNCE(all_item_emb[i_unique], cl_item_emb[i_unique], args.cl_tau)
            cl_loss = cl_loss_user + cl_loss_item
            
            # 组合所有损失：BPR损失 + 正则化损失*权重 + 对比学习损失*权重
            return loss_bpr + args.reg_lambda * reg_loss + args.cl_lambda * cl_loss
        
        # 组合BPR损失和正则化损失并返回
        return loss_bpr + args.reg_lambda * reg_loss

    def train_func_batch(self):
        """批量训练函数（备用）
        
        Returns:
            平均训练损失
        """
        train_losses = []
        train_user = self.dataset.train_user
        train_item = self.dataset.train_item
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            self.computer()
            train_loss = self.loss_bpr(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size])
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss)
        return torch.stack(train_losses).mean()

    def save_emb(self):
        """保存用户和物品嵌入
        
        保存到指定路径，用于后续分析或模型加载
        """
        torch.save(self.embedding_user, f'../saved/{args.data:s}_user.pt')
        torch.save(self.embedding_item, f'../saved/{args.data:s}_item.pt')

    def load_emb(self):
        """加载用户和物品嵌入
        
        从指定路径加载预训练的嵌入向量
        """
        self.embedding_user = torch.load(f'../saved/{args.data:s}_user.pt').to(args.device)
        self.embedding_item = torch.load(f'../saved/{args.data:s}_item.pt').to(args.device)
