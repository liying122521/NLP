
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from Auto_scoring.config import set_args
from Auto_scoring.model_cls import Model
from torch.utils.data import DataLoader
from Auto_scoring.utils import compute_corrcoef, l2_normalize, compute_pearsonr
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertTokenizer
from transformers import AutoTokenizer  # todo
from Auto_scoring.data_helper import load_data, SentDataSet, collate_func
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(model):
    model.eval()
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    all_probs = []
    for step, batch in tqdm(enumerate(val_dataloader)):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        s1_input_ids, s2_input_ids, label_id = batch
        if torch.cuda.is_available():
            s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()
        with torch.no_grad():
            label_id = label_id.cpu().numpy()
            all_labels.extend(label_id)
            # 计算模型的预测标签
            logits = model(s1_input_ids, s2_input_ids, encoder_type='last-avg')
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

            s1_embeddings = model.encode(s1_input_ids, encoder_type='last-avg').float() # todo
            s2_embeddings = model.encode(s2_input_ids, encoder_type='last-avg').float() # todo
            # label_id = label_id.float() # todo
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label_id = label_id.cpu().numpy()

            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label_id)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)
    #
    a_vecs = torch.from_numpy(all_a_vecs)  # todo
    b_vecs = torch.from_numpy(all_b_vecs)

    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)

    sims = (a_vecs * b_vecs).sum(axis=1)

    predicted_labels = np.argmax(all_probs, axis=1)
    accuracy = accuracy_score(all_labels, predicted_labels)
    # precision = precision_score(all_labels, predicted_labels, average='weighted/micro ')
    precision = precision_score(all_labels, predicted_labels, average='weighted', zero_division=1) # todo
    # precision = precision_score(all_labels, predicted_labels, average='weighted')
    recall = recall_score(all_labels, predicted_labels, average='weighted')
    f1 = f1_score(all_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1

    # return all_labels, all_preds # todo
    # corrcoef = compute_corrcoef(all_labels, sims)
    # pearsonr = compute_pearsonr(all_labels, sims)
    # return corrcoef, pearsonr

if __name__ == '__main__':
    args = set_args()
    # args.output_dir = 'output_last-avg'
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain_path)

    train_df = load_data(args.train_data_path)
    train_dataset = SentDataSet(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_func)

    val_df = load_data(args.dev_data_path)
    val_dataset = SentDataSet(val_df, tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.dev_batch_size, collate_fn=collate_func)
    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = Model()
    loss_fct = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
    #     loss_fct.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8) # 使用 AdamW 优化器进行模型参数优化。
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps) # 创建学习率调度器，根据预热步数和总训练步数来调整学习率。

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            s1_input_ids, s2_input_ids, label_id = batch
            if torch.cuda.is_available():
                s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()
            logits = model(s1_input_ids, s2_input_ids, encoder_type='last-attention')
            loss = loss_fct(logits, label_id)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # 一轮跑完 进行eval
        # corrcoef, pearsonr = evaluate(model)
        # ss = 'epoch:{}, spearmanr:{:10f}, pearsonr:{:10f}'.format(epoch, corrcoef, pearsonr)
        # with open(args.output_dir + '/logs_gjc.txt', 'a+', encoding='utf8') as f:
        #     ss += '\n'
        #     f.write(ss)
        # model.train()
        #
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        # torch.save(model_to_save.state_dict(), output_model_file)

        accuracy, precision, recall, f1 = evaluate(model)
        print('Epoch:{}, accuracy:{:10f}, Precision:{:10f}, Recall:{:10f}, F1:{:10f}'.format(epoch, accuracy, precision,
                                                                                             recall, f1))
        with open(args.output_dir + '/logs_gjc.txt', 'a+', encoding='utf8') as f:
            f.write('Epoch:{}, accuracy:{:10f}, Precision:{:10f}, Recall:{:10f}, F1:{:10f}\n'.format(epoch, accuracy, precision, recall, f1))

        model.train()
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
