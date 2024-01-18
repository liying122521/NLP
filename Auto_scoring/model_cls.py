
import torch
import numpy as np # todo
from Auto_scoring.utils import l2_normalize
from torch import nn
from Auto_scoring.config import set_args
from transformers.models.bert import BertModel, BertConfig
from transformers import AutoModel, AutoConfig # todo
import torch.nn.functional as F  # todo

args = set_args()
class CustomPooler(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomPooler, self).__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 使用 [CLS] 标记对应的隐藏状态作为输入
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.config = BertConfig.from_pretrained(args.bert_pretrain_path)
        # self.bert = BertModel.from_pretrained(args.bert_pretrain_path)
        self.config = AutoConfig.from_pretrained(args.bert_pretrain_path)
        self.bert = AutoModel.from_pretrained(args.bert_pretrain_path)
        # 自定义的 pooler 层
        self.custom_pooler = CustomPooler(self.config.hidden_size, self.config.hidden_size)
        self.clssify = nn.Linear(self.config.hidden_size * 3, 5)
        self.cross_attention1 = nn.MultiheadAttention(self.config.hidden_size, self.config.num_attention_heads)
        self.cross_attention2 = nn.MultiheadAttention(self.config.hidden_size, self.config.num_attention_heads)


    def get_embedding(self, output, encoder_type):
        # print(output.shape)  # 打印输出类型
        if encoder_type == 'fist-last-avg':
            first = output.hidden_states[1]   # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)   # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            # final_encoding = F.max_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # 对最后一层的隐藏状态进行最大池化操作 todo
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output

        if encoder_type == 'last-attention':
            sequence_output = output.last_hidden_state
            seq_length = sequence_output.size(1)
            cls_vector = sequence_output[:, 0, :]
            word_vectors = sequence_output[:, 1:, :]
            word_vectors_normalized = F.normalize(word_vectors, p=2, dim=-1) # (batch_size, max_len - 1, hidden_size)
            cls_vector_normalized = F.normalize(cls_vector, p=2, dim=-1) # (batch_size, hidden_size)

            similarities = F.cosine_similarity(word_vectors_normalized, cls_vector_normalized.unsqueeze(1), dim=-1) # (batch_size, max_len - 1)
            attention_weights = F.softmax(similarities, dim=1)
            # attention_weights = torch.softmax(similarities, dim=1)

            weighted_sum = torch.sum(word_vectors * attention_weights.unsqueeze(-1), dim=1)
            # weighted_sum的形状 (batch_size, hidden_size)
            return weighted_sum

# **************************************************************** todo
    def get_embedding1(self, output, encoder_type):
        if encoder_type == "cls":
            cls = output[:, 0, :]  # [b,d]
            return cls
        if encoder_type == 'last-avg':
            sequence_output = output  # (batch_size, max_len, hidden_size) 经过交叉注意力计算之后得到的替换过的cls在output中
            seq_length = sequence_output.size(1) # sequence_output.size(1) 返回张量的第二个维度的大小，
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1) # (batch_size, hidden_size)
            # final_encoding = F.max_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # 对最后一层的隐藏状态进行最大池化操作 todo
            return final_encoding

        if encoder_type == 'last-attention':
            cls_vector = output[:, 0, :]
            word_vectors = output[:, 1:, :]
            word_vectors_normalized = F.normalize(word_vectors, p=2, dim=-1) # (batch_size, max_len - 1, hidden_size)
            cls_vector_normalized = F.normalize(cls_vector, p=2, dim=-1) # (batch_size, hidden_size)
            similarities = F.cosine_similarity(word_vectors_normalized, cls_vector_normalized.unsqueeze(1), dim=-1) # (batch_size, max_len - 1
            attention_weights = F.softmax(similarities, dim=1)  # 对相似度进行softmax操作
            weighted_sum = torch.sum(word_vectors * attention_weights.unsqueeze(-1), dim=1)  # 将词向量与注意力权重相乘并求和，维度为 (batch_size, hidden_size)
            return weighted_sum

    def forward(self, s1_input_ids, s2_input_ids, encoder_type='cls'):
        s1_attention_mask = torch.ne(s1_input_ids, 0)
        # torch.ne(s1_input_ids, 0) 执行了一个元素级别的比较，将 s1_input_ids 张量中所有等于 0 的元素替换为 False，其他不等于 0 的元素替换为 True。
        s2_attention_mask = torch.ne(s2_input_ids, 0)
        s1_output = self.bert(s1_input_ids, s1_attention_mask, output_hidden_states=True)
        s2_output = self.bert(s2_input_ids, s2_attention_mask, output_hidden_states=True)

    # **************************************************************************** todo
    #     sequence_output1 = s1_output.last_hidden_state  # (batch_size, max_len, hidden_size)
    #     sequence_output2 = s2_output.last_hidden_state  # (batch_size, max_len, hidden_size)
    #     cls_vector1 = sequence_output1[:, 0, :] # (batch_size, hidden_size)
    #     cls_vector2 = sequence_output2[:, 0, :] # (batch_size, hidden_size)
    #     attention_scores1_to_2 = torch.matmul(sequence_output1, sequence_output2.transpose(1, 2)) # (batch_size, max_len, max_len)
    #     attention_scores2_to_1 = torch.matmul(sequence_output2, sequence_output1.transpose(1, 2))
    #     attention_weights1_to_2 = F.softmax(attention_scores1_to_2, dim=-1)
    #     attention_weights2_to_1 = F.softmax(attention_scores2_to_1, dim=-1)
    #
    #     s1_output = torch.matmul(attention_weights1_to_2, sequence_output2)  # output1 包含了文本1中每个位置的交叉注意力输出
    #     s2_output = torch.matmul(attention_weights2_to_1, sequence_output1)  # output2 包含了文本2中每个位置的交叉注意力输出
    #     s1_embedding = self.get_embedding1(s1_output, encoder_type)
    #     s2_embedding = self.get_embedding1(s2_output, encoder_type)
    # *****************************************************************************

        s1_embedding = self.get_embedding(s1_output, encoder_type)
        # 使用self.get_embedding函数从s1_output中获取编码表示s1_embedding。具体的实现可能涉及对隐藏状态进行处理和选择。
        s2_embedding = self.get_embedding(s2_output, encoder_type)

        diff = torch.abs(s1_embedding - s2_embedding) # 计算s1_embedding和s2_embedding的差异，使用torch.abs函数获取绝对值。
        concat_vector = torch.cat([s1_embedding, s2_embedding, diff], dim=-1)
        logits = self.clssify(concat_vector)
        return logits
        
    def encode(self, input_ids, encoder_type='cls'):
        attention_mask = torch.ne(input_ids, 0)
        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        embedding = self.get_embedding(output, encoder_type)
        return embedding


























