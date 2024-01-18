import argparse

def set_args():

    parser = argparse.ArgumentParser('--使用transformers实现Auto_scoring')
    # parser.add_argument('--train_data_path', default='../data/lcqmc/train.tsv', type=str, help='训练数据集')
    # parser.add_argument('--dev_data_path', default='../data/lcqmc/dev.tsv', type=str, help='测试数据集')
    parser.add_argument('--train_data_path', default='../data/XFDL/train_48_gjc.txt', type=str, help='训练数据集')
    parser.add_argument('--dev_data_path', default='../data/XFDL/dev_48_gjc.txt', type=str, help='测试数据集')
    parser.add_argument('--bert_pretrain_path', default='./bert_pretrain', type=str, help='预训练模型路径')
    # parser.add_argument('--roberta_pretrain_path', default='./roberta-wwm-ext', type=str, help='预训练模型路径')
    # parser.add_argument('--albert_pretrain_path', default='./albert-base-chinese', type=str, help='预训练模型路径')
    # parser.add_argument('--distilbert_pretrain_path', default='./distilbert-base-mutilingual-similarity', type=str, help='预训练模型路径')
    parser.add_argument('--train_batch_size', default=8, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=8, type=int, help='验证批次的大小')
    parser.add_argument('--output_dir', default='./output/XFDL_output/yuxian', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率大小')
    return parser.parse_args()
