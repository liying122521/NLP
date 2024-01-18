
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from Auto_scoring.config import set_args
from Auto_scoring.model import Model
from torch.utils.data import DataLoader
from Auto_scoring.utils import compute_mse, compute_mae, compute_rmse, compute_corrcoef, l2_normalize, compute_pearsonr
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertTokenizer
from transformers import AutoTokenizer  # todo
from Auto_scoring.data_helper import load_data, SentDataSet, collate_func
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F  # todo
import time

def evaluate(model):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # todo
    model.eval()
    all_a_vecs, all_b_vecs = [], []
    all_labels = []

    for step, batch in tqdm(enumerate(val_dataloader)): # 返回每个批次的索引和对应的数据，step=一共有多少个批次。
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        s1_input_ids, s2_input_ids, label_id = batch # 从批次中解包出输入句子1的输入标识符（input IDs）、句子2的输入标识符和标签。
        if torch.cuda.is_available():
            s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()
        with torch.no_grad():
            s1_embeddings = model(s1_input_ids, encoder_type='last-avg')
            s2_embeddings = model(s2_input_ids, encoder_type='last-avg')
            s1_embeddings = s1_embeddings.cpu().numpy()
            # 将句子1的嵌入向量从 GPU 上移回 CPU，并将其转换为 NumPy 数组。
            s2_embeddings = s2_embeddings.cpu().numpy()
            label_id = label_id.cpu().numpy()

            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label_id)

    all_a_vecs = np.array(all_a_vecs) # (batch, dim)
    #  将 all_a_vecs 转换为 NumPy 数组。
    all_b_vecs = np.array(all_b_vecs)
    all_labels  = np.array(all_labels)

    all_a_vecs = torch.tensor(np.array(all_a_vecs))  # todo
    all_b_vecs = torch.tensor(np.array(all_b_vecs))
    all_labels = torch.tensor(all_labels) # todo

    # sims = (a_vecs * b_vecs).sum(axis=1)
    # sims = sims.numpy()


    sims =torch.cosine_similarity(all_a_vecs, all_b_vecs)
    sims = sims.numpy()

    # sims = torch.pairwise_distance(all_a_vecs, all_b_vecs, p=2)
    # min_value = torch.min(sims)
    # max_value = torch.max(sims)
    # sims = 1 - (sims - min_value) / (max_value - min_value)
    # sims = sims.numpy()

    # sims = torch.sum(torch.abs(all_a_vecs - all_b_vecs), dim=1)
    # min_value = torch.min(sims)
    # max_value = torch.max(sims)
    # sims = 1 - (sims - min_value) / (max_value - min_value)
    # sims = sims.numpy()

    mse = compute_mse(all_labels, sims)
    mae = compute_mae(all_labels, sims)
    rmse = compute_rmse(all_labels, sims)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return mse, mae, rmse, corrcoef, pearsonr


def calc_loss(s1_vec, s2_vec, true_label):
    loss_fct = nn.MSELoss()
    output = torch.cosine_similarity(s1_vec, s2_vec)
    output = output
    loss = loss_fct(output, true_label)
    return loss


if __name__ == '__main__':
    args = set_args() # 通过调用 set_args() 函数获取命令行参数或设置默认参数，并将其存储在 args 中。
    os.makedirs(args.output_dir, exist_ok=True) # 创建输出目录，如果目录已存在则忽略。
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)

    train_df = load_data(args.train_data_path) # 加载训练数据集，数据路径由 args.train_data_path 指定。
    train_dataset = SentDataSet(train_df, tokenizer) # 使用加载的训练数据集和分词器创建一个 SentDataSet 对象。

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_func)
    # 使用训练数据集创建一个训练数据加载器，用于批量加载和迭代训练数据。shuffle=True：表示在每个训练 epoch（迭代轮数）开始时是否对数据进行洗牌（随机重排），
    # 以增加训练的随机性。洗牌可以帮助模型更好地学习数据的不同特征。collate_func是一个自定义的函数，用于将不同长度的样本序列填充为相同长度，以便于模型的批量化处理。

    val_df = load_data(args.dev_data_path)
    val_dataset = SentDataSet(val_df, tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.dev_batch_size, collate_fn=collate_func)

    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    # 计算总的训练步数，基于训练数据集大小、训练批次大小、梯度累积步数和训练轮数。(lcqmc数据集计算num_train_steps=238766/32/1*5=37307)

    # 模型
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters()) # 获取模型的参数列表（包括enbedding层和12层encoder层的参数）。
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # 定义不进行权重衰减的参数名称。
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps # 计算学习率预热步数。
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8) # 使用 AdamW 优化器进行模型参数优化。
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps) # 创建学习率调度器，根据预热步数和总训练步数来调整学习率。

    for epoch in range(args.num_train_epochs): # 开始训练循环，迭代指定数量的训练轮数。
        for step, batch in enumerate(train_dataloader): # 在每个训练轮次中，迭代训练数据加载器以获取每个训练批次的数据。
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            s1_input_ids, s2_input_ids, label_id = batch
            if torch.cuda.is_available():
                s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()

            s1_vec = model(s1_input_ids, encoder_type='last-avg').float() # todo
            s2_vec = model(s2_input_ids, encoder_type='last-avg').float()
            label_id = label_id.float()
            loss = calc_loss(s1_vec, s2_vec, label_id) # 计算损失，根据模型预测的 s1 和 s2 的向量表示以及标签。

            if args.gradient_accumulation_steps > 1: # 如果梯度累积步数大于 1，则对损失进行平均。
                loss = loss / args.gradient_accumulation_steps

            print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        mse, mae, rmse, corrcoef, pearsonr = evaluate(model)
        ss = 'epoch:{},mse:{:10f},mae:{:10f}, rmse:{:10f}, spearmanr:{:10f}, pearsonr:{:10f}'.format(epoch, mse, mae, rmse, corrcoef, pearsonr)
        with open(args.output_dir + '/logs_50_gjc.txt', 'a+', encoding='utf8') as f:
            ss += '\n'
            f.write(ss) # 打开日志文件，将日志字符串写入文件。
        model.train() # 将模型设置为训练模式，以便进行下一轮训练。

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file) # 获取要保存的模型对象




