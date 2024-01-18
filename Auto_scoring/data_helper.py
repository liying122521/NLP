import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_data(path):
    # print(path)
    sent1, sent2, label = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t') # strip() 方法用于移除字符串中的首尾空白字符（例如空格、制表符、换行符等）。
            sent1.append(line[1]) # todo
            sent2.append(line[2])
            try:

                # 回归
                # lab = int(line[3])
                # lab_data = lab / 5
                # label.append(lab_data)

                lab = int(line[3])
                lab_data = (lab-1)
                label.append(lab_data)  # todo

            except:
                print(line)
    df = pd.DataFrame({'sent1': sent1, 'sent2': sent2, 'label': label})
    # DataFrame是一个二维表格数据结构，类似于Excel中的电子表格或SQL中的表。它由行和列组成，每列可以包含不同类型的数据。
    # sent1、sent2 和 label变量对应DataFrame的一列,列名分别为 'sent1'、'sent2' 和 'label'
    return df


class SentDataSet(Dataset):
    """SentDataSet类用于加载文本数据集，并提供了方便的方法来获取编码后的输入序列和标签。这样，我们可以将该数据集作为输入提供给模型进行训练和评估。"""
    def __init__(self, data, tokenizer):
        # 构造函数，接受两个参数 data 和 tokenizer。data 是一个包含句子1、句子2和标签的数据集，使用 pandas DataFrame 来表示。
        # tokenizer 是一个用于将文本编码为输入序列的分词器。
        self.tokenizer = tokenizer
        self.data = data   # 将传入的数据集保存到类的属性中 # pd.DataFrame({'sent1': sent1, 'sent2': sent2, 'label': label})
        self.sent1 = self.data['sent1'] # 从数据集中获取列名为 'sent1' 的句子1列，并保存到类的属性中。
        self.sent2 = self.data['sent2']
        self.label = self.data['label'] # todo

    def __len__(self):
        """dataset = SentDataSet(data, tokenizer)
           length = len(dataset)  # 显式调用 __len__(self) 方法
           在上面的示例中，len(dataset) 会自动调用 SentDataSet 类中的 __len__() 方法来获取数据集的长度。"""
        return len(self.sent1) # 返回数据集的长度，这里以句子1的数量为准。

    def __getitem__(self, idx): # 根据给定的索引 idx，获取数据集中对应索引位置的样本。
        """dataset = SentDataSet(data, tokenizer)
           sample = dataset[idx]  # 自动调用 __getitem__(self, idx) 方法
           在上面的示例中，dataset[idx] 会自动调用 SentDataSet 类中的 __getitem__() 方法来获取指定索引位置 idx 的样本。
           需要注意的是，idx 是一个整数索引值，表示要获取的样本在数据集中的位置。你可以根据具体需求来使用不同的索引值，例如 dataset[0] 获取第一个样本"""
        s1_input_ids = self.tokenizer.encode(self.sent1[idx])
        # 使用分词器对句子1进行编码，得到输入序列的 ID 列表，并保存到 s1_input_ids 中。
        s2_input_ids = self.tokenizer.encode(self.sent2[idx])
        return {'s1_input_ids': s1_input_ids, 's2_input_ids': s2_input_ids, 'label': self.label[idx]}
        # 返回一个字典，包含了编码后的句子1和句子2的输入序列（s1_input_ids 和 s2_input_ids），以及对应的标签（label）。这个字典表示数据集中一个样本的信息。

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    # 用于将输入的 input_ids 序列填充或截断到指定的最大长度 max_len。
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_func(batch_data):
    # 用于将一个批次（batch_data）的数据进行整理和处理，以便在训练或推理过程中进行批处理。
    '''
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    :param batch_data: batch数据
    :return:
    '''
    s1_max_len = max([len(d['s1_input_ids']) for d in batch_data])
    # 计算批次中 s1_input_ids 和 s2_input_ids 的最大长度
    s2_max_len = max([len(d['s2_input_ids']) for d in batch_data])
    # ***************************************************************************todo
    total_max_len = max(s1_max_len, s2_max_len)  # todo
    # ***************************************************************************todo
    s1_input_ids_list, s2_input_ids_list, label_list = [], [], [] # 初始化空列表用于存储 s1_input_ids、s2_input_ids 和标签
    for item in batch_data:
        # 遍历批次中的每个样本
        # s1_input_ids_list.append(pad_to_maxlen(item['s1_input_ids'], max_len=s1_max_len))
        # # 将 s1_input_ids 和 s2_input_ids 填充到对应的最大长度
        # s2_input_ids_list.append(pad_to_maxlen(item['s2_input_ids'], max_len=s2_max_len))

        # ***************************************************************************todo
        s1_input_ids_list.append(pad_to_maxlen(item['s1_input_ids'], max_len=total_max_len))
        s2_input_ids_list.append(pad_to_maxlen(item['s2_input_ids'], max_len=total_max_len))
        # ***************************************************************************todo

        label_list.append(item['label']) # 将标签添加到标签列表中
    all_s1_input_ids = torch.tensor(s1_input_ids_list, dtype=torch.long) # 使用 torch.tensor() 将列表转换为 PyTorch 张量
    all_s2_input_ids = torch.tensor(s2_input_ids_list, dtype=torch.long)
    all_labels_id = torch.tensor(label_list, dtype=torch.long)   # 分类这里为torch.long  回归这里为torch.float
    # all_labels_id = torch.tensor(label_list, dtype=torch.float)   # 分类这里为torch.long  回归这里为torch.float todo
    return all_s1_input_ids, all_s2_input_ids, all_labels_id
    # 返回填充后的 s1_input_ids、s2_input_ids 和标签的张量

















