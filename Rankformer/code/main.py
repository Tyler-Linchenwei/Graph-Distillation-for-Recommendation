#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rankformer模型训练与评估主程序

该文件是整个Rankformer推荐系统的入口点，负责协调模型的训练、验证和测试流程。
核心功能包括：
1. 模型初始化与配置
2. 训练循环控制
3. 验证集评估与早停机制
4. 测试集性能报告
5. 最佳模型跟踪与嵌入保存

主要流程：
- 加载配置参数和数据集
- 初始化模型
- 预训练嵌入加载（可选）
- 初始验证
- 训练循环（包含训练和定期验证）
- 早停检查
- 最终结果报告
"""

# 导入必要的模块和组件
from parse import args  # 参数配置
from dataloader import dataset  # 数据集实例
from model import Model  # 模型定义
import torch  # PyTorch库
import numpy as np  # 数值计算库


def print_test_result():
    """打印测试结果函数
    
    在最佳验证结果对应的epoch上，打印测试集的性能指标。
    支持同时打印多个Top-K值的评估指标。
    
    全局变量依赖：
        best_epoch: 最佳验证结果对应的epoch数
        test_pre: 测试集Precision@K指标数组
        test_recall: 测试集Recall@K指标数组
        test_ndcg: 测试集NDCG@K指标数组
    """
    global best_epoch, test_pre, test_recall, test_ndcg
    print(f'===== Test Result(at {best_epoch:d} epoch) =====')
    for i, k in enumerate(args.topks):
        print(f'ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}')


def train():
    """模型训练函数
    
    执行一个epoch的模型训练，包括前向传播、损失计算、反向传播和参数更新。
    根据配置定期打印训练损失，以便监控训练过程。
    
    变量依赖：
        epoch: 当前训练的epoch数（全局变量）
        model: 模型实例（全局变量）
        args.show_loss_interval: 打印损失的间隔epoch数
    
    功能流程：
    1. 调用model.train_func()执行训练，返回训练损失
    2. 检查是否需要打印损失（当前epoch是否为show_loss_interval的倍数）
    3. 打印当前epoch和训练损失
    """
    # 执行一个epoch的训练，获取训练损失
    train_loss = model.train_func()
    
    # 每隔指定间隔打印一次训练损失
    if epoch % args.show_loss_interval == 0:  # 检查是否达到打印间隔
        print(f'epoch {epoch:d}, train_loss = {train_loss:f}.')


def valid(epoch):
    """模型验证和测试函数
    
    核心功能：
    1. 在验证集上评估模型性能
    2. 与当前最佳验证结果比较
    3. 若为新最佳结果，在测试集上评估并保存
    4. 控制早停机制的触发条件
    
    评估指标：
    - Precision@K: 推荐列表中相关物品的比例
    - Recall@K: 相关物品中被推荐的比例
    - NDCG@K: 考虑推荐顺序的归一化折扣累积增益
    
    Args:
        epoch: 当前训练的epoch数，用于打印和记录
        
    Returns:
        bool: 是否找到了新的最佳模型（True表示找到新最佳）
        
    全局变量修改：
        best_valid_ndcg: 最佳验证NDCG值
        best_epoch: 最佳验证结果对应的epoch
        test_pre: 测试集Precision@K结果
        test_recall: 测试集Recall@K结果
        test_ndcg: 测试集NDCG@K结果
    """
    global best_valid_ndcg, best_epoch, test_pre, test_recall, test_ndcg
    
    # 步骤1: 在验证集上评估模型性能
    valid_pre, valid_recall, valid_ndcg = model.valid_func()
    
    # 步骤2: 打印当前验证结果
    print(f'\n----- Epoch {epoch:d} Validation Results -----')
    for i, k in enumerate(args.topks):
        print(f'ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}')
    print('-------------------------------------')
    
    # 步骤3: 检查是否为新的最佳模型
    # 使用最大Top-K值的NDCG作为评价指标
    # 例如：若args.topks = [5, 10, 20]，则使用ndcg@20进行比较
    if valid_ndcg[-1] > best_valid_ndcg:
        # 更新最佳结果记录
        best_valid_ndcg = valid_ndcg[-1]
        best_epoch = epoch
        
        # 在测试集上评估模型（仅当找到新最佳模型时）
        test_pre, test_recall, test_ndcg = model.test_func()
        
        # 打印当前最佳测试结果
        print_test_result()
        
        # 保存嵌入（如果配置了保存选项）
        if args.save_emb:
            print(f'Saving embeddings to ../saved/{args.data:s}_*.pt')
            model.save_emb()
        
        # 返回True表示找到新的最佳模型
        return True
    
    # 返回False表示未找到新的最佳模型
    return False


# =============================================================================
# 主程序执行部分
# =============================================================================

# 步骤1: 初始化模型
# 创建Model实例并将其移动到指定设备（GPU或CPU）
print(f'Initializing model on {args.device}...')
model = Model(dataset).to(args.device)

# 步骤2: 加载预训练嵌入（可选）
if args.load_emb:
    print(f'Loading pre-trained embeddings from ../saved/{args.data:s}_*.pt')
    model.load_emb()

# 步骤3: 初始化全局跟踪变量
# 最佳验证结果记录
best_valid_ndcg = 0.0  # 最佳验证NDCG值（使用最大Top-K）
best_epoch = 0          # 最佳验证结果对应的epoch数

# 测试集结果记录（使用最佳模型）
test_pre = torch.zeros(len(args.topks), device=args.device)     # Precision@K
test_recall = torch.zeros(len(args.topks), device=args.device)   # Recall@K
test_ndcg = torch.zeros(len(args.topks), device=args.device)     # NDCG@K

# 步骤4: 训练前的初始验证
# 在训练开始前，先在验证集上评估初始模型性能
print('\n=== Initial Validation (Before Training) ===')
valid(epoch=0)

# 步骤5: 模型训练循环
print('\n=== Starting Training Loop ===')
print(f'Max epochs: {args.max_epochs}, Valid interval: {args.valid_interval}, Early stopping: {args.stopping_step}')
print('-' * 50)

epoch = 1  # 训练epoch计数器从1开始
while epoch <= args.max_epochs:
    # 执行一个epoch的训练
    train()
    
    # 定期验证模型性能
    if epoch % args.valid_interval == 0:
        # 在验证集上评估并检查是否为新最佳模型
        found_best = valid(epoch)
        
        # 早停机制检查
        # 条件：1. 未找到新最佳模型 2. 距离最佳epoch已超过指定步数
        if not found_best and (epoch - best_epoch) >= args.stopping_step * args.valid_interval:
            print(f'\n=== Early Stopping Triggered ===')
            print(f'No improvement for {args.stopping_step} validation intervals.')
            print(f'Best epoch: {best_epoch}, Current epoch: {epoch}')
            break
    
    # 进入下一个epoch
    epoch += 1

# 步骤6: 训练完成，输出最终结果
print('\n' + '-' * 50)
print('=== Training Completed ===')
print_test_result()  # 打印最佳模型的测试结果
# ==========================================
    # 【新增代码】 保存老师模型 (Teacher Embeddings)
    # 请把这段代码粘贴到 main.py 的最末尾
    # ==========================================
import os
import torch

print(">>> Starting to save Teacher Embeddings...")

# 1. 确保目录存在
save_dir = os.path.join('data', args.data)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'teacher_rankformer.pt')

# 2. 获取模型中的 Embedding
# 注意：通常变量名叫 embedding_user 或 user_emb，根据 Rankformer 的习惯通常是 embedding_user
# 如果报错说 'model' 没有 'embedding_user'，请检查 model.py 里的定义

# 确保切回 CPU 并且不带梯度
model.eval()
with torch.no_grad():
    teacher_data = {
        # 这里假设模型里的层叫 embedding_user 和 embedding_item
        # 如果你的代码里叫 user_emb，请把 .embedding_user 改成 .user_emb
        "user_emb": model.embedding_user.weight.detach().cpu(),
        "item_emb": model.embedding_item.weight.detach().cpu()
    }

# 3. 保存文件
torch.save(teacher_data, save_path)
print(f">>> Teacher Embeddings successfully saved to: {save_path}")
# 释放资源（可选）
torch.cuda.empty_cache() if torch.cuda.is_available() else None
