
import torch
from torch import nn
from Auto_scoring.config import set_args
from transformers.models.bert import BertModel, BertConfig
from transformers import AutoModel, AutoConfig # todo
import torch.nn.functional as F  # todo
from Auto_scoring.utils import l2_normalize # todo
import numpy as np # todo

args = set_args()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # 首先调用父类的初始化函数 super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrain_path)
        self.attention_weights = nn.Parameter(torch.randn(self.bert.config.hidden_size, self.bert.config.hidden_size))

    def forward(self, input_ids, encoder_type='cls'):
        # 在前向传播函数 forward 中，接收输入 input_ids 和一个名为 encoder_type 的参数，用于选择不同的编码方式。
        attention_mask = torch.ne(input_ids, 0) # 创建 attention_mask，通过与0进行比较来确定哪些位置是真实令牌，哪些位置是填充的。
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        # 使用 BERT 模型处理输入，output_hidden_states=True 会返回所有隐藏状态

        if encoder_type == 'fist-last-avg':
            # 第一层和最后一层的隐层取出
            first = output.hidden_states[1]   # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)   # 这行代码用于获取隐藏状态 first 在维度1上的长度，即序列的长度。

            # 对第一层和最后一层的隐藏状态进行平均池化操作
            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)

            return final_encoding

        if encoder_type == 'last-avg': # 获取最后一层的隐藏状态
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            # final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1) # 对最后一层的隐藏状态进行平均池化操作
            final_encoding = F.max_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1) # 对最后一层的隐藏状态进行最大池化操作 todo
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state  # [batchsize,sequench_len,dim]最后一层的输出的隐藏状态
            cls = sequence_output[:, 0]  # [batchsize,dim] 最后一层的输出的第一个位置的隐藏状态
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d] 获取 output 中的 pooler_output，它是经过模型的池化层计算得到的表示，形状为 (batch_size, hidden_size)
            return pooler_output
        """encoder_type =='last-avg' 使用平均池化操作对最后一层隐藏状态进行处理,而 encoder_type =='pooler' 则直接使用模型的池化层输出作为最终的编码结果。
        在BERT模型中，output.pooler_output 是经过池化层计算得到的表示。它使用了非线性激活函数（tanh）和全连接层，
        将最后一层的隐藏状态中的[CLS]标记对应的隐藏状态作为输入，经过线性变换和激活函数处理得到一个固定维度的向量作为整个序列的池化表示。
        因此，池化层的操作不是简单的平均池化，而是一种结合非线性激活函数和全连接层的处理方式，用于获得一个维度固定的序列表示。"""

        if encoder_type == 'last-attention':  # 获取最后一层的隐藏状态
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            cls_vector = sequence_output[:, 0, :]  # 获取CLS向量，维度为 (batch_size, hidden_size)
            word_vectors = sequence_output[:, 1:, :]  # 获取除CLS向量外的词向量，维度为 (batch_size, max_len - 1, hidden_size)

           # ***************内积相似度************************************************
            # similarities = torch.matmul(word_vectors, cls_vector.unsqueeze(-1))  # 内积计算词向量与CLS向量之间的相似度
            # attention_weights = F.softmax(similarities, dim=1)  # 对相似度进行softmax操作
            # weighted_sum = torch.sum(word_vectors * attention_weights,dim=1)  # 将词向量与注意力权重相乘并求和，维度为 (batch_size, hidden_size)
            # return weighted_sum


            # 对词向量进行归一化
            word_vectors_normalized = F.normalize(word_vectors, p=2, dim=-1) # (batch_size, max_len - 1, hidden_size)
            cls_vector_normalized = F.normalize(cls_vector, p=2, dim=-1) # (batch_size, hidden_size)
            # ***************余弦相似度************************************
            # 计算归一化后的词向量与CLS向量之间的余弦相似度
            similarities = F.cosine_similarity(word_vectors_normalized, cls_vector_normalized.unsqueeze(1), dim=-1) # (batch_size, max_len - 1)
            attention_weights = F.softmax(similarities, dim=1)  # 对相似度进行softmax操作
            # attention_weights = torch.softmax(similarities, dim=1)  # 对相似度进行softmax操作 todo
            weighted_sum = torch.sum(word_vectors * attention_weights.unsqueeze(-1), dim=1)
            return weighted_sum

        if encoder_type == 'fist-last-attention':
            first_output = output.hidden_states[1]
            last_output = output.last_hidden_state
            first_cls_vector = last_output[:, 0, :]
            first_word_vectors = last_output[:, 1:, :]
            last_cls_vector = last_output[:, 0, :]
            last_word_vectors = last_output[:, 1:, :]

            first_cls_vector_normalized = first_cls_vector / torch.norm(first_cls_vector, p=2, dim=-1, keepdim=True)
            first_word_vectors_normalized = first_word_vectors / torch.norm(first_word_vectors, p=2, dim=-1, keepdim=True)

            last_cls_vector_normalized = last_cls_vector / torch.norm(last_cls_vector, p=2, dim=-1, keepdim=True)
            last_word_vectors_normalized = last_word_vectors / torch.norm(last_word_vectors, p=2, dim=-1, keepdim=True)

            # ***************余弦相似度************************************
            first_similarities = torch.cosine_similarity(first_word_vectors_normalized, first_cls_vector_normalized.unsqueeze(1), dim=-1) # (batch_size, max_len - 1) todo
            last_similarities1 = torch.cosine_similarity(last_word_vectors_normalized, last_cls_vector_normalized.unsqueeze(1), dim=-1) # (batch_size, max_len - 1) todo

            first_attention_weights = torch.softmax(first_similarities, dim=1)
            last_attention_weights = torch.softmax(last_similarities1, dim=1)
            first_weighted_sum = torch.sum(first_word_vectors * first_attention_weights.unsqueeze(-1), dim=1)
            last_weighted_sum = torch.sum(last_word_vectors * last_attention_weights.unsqueeze(-1), dim=1)
            final_encoding = torch.avg_pool1d(
                torch.cat([first_weighted_sum.unsqueeze(1), last_weighted_sum.unsqueeze(1)], dim=1).transpose(1, 2),kernel_size=2).squeeze(-1)
            return final_encoding















