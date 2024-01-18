# Requirements Specification Text Scoring
## Chinese data set description

This Chinese data set about requirements specification is designed, collected and constructed by us, and contains 12840 pieces of data after processing.

| Data set name | Training set (number) | Validation sets (number) |
| :-: | :-: | :-: |
| XFDL | 10700 | 2140 |

## pre-trained language model

Pre-training model download address：

Since the dataset we constructed is based on Chinese, we chose to train and validate it on a Chinese pre-trained model.

bert_base_chinese model: https://huggingface.co/bert-base-chinese/tree/main

albert_base_chinese model：https://huggingface.co/ckiplab/albert-base-chinese/tree/main

chinese_roberta_wwm_ext model：https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main

distilbert-base-multilingual model：https://huggingface.co/distilbert-base-multilingual-cased/tree/main

The downloaded Chinese pre-training model is placed in the Auto_scoring directory, and you can execute the code by placing it in the corresponding location. The following files are mainly needed：

- pytorch_model.bin
- config.json
- vocab.txt

## Description of evaluation indicators

- **Accuracy**: Accuracy is one of the most intuitive and commonly used performance evaluation metrics, reflecting the overall performance of the model.
- **F1 value**:F1 value combines precision and recall to fully evaluate the performance of the model.
- **Mean Absolute Error (MAE)**: the mean absolute error between the true rating value and the model-predicted rating value.
- **Root Mean Square Error (RMSE)**: the root mean square error between the true rating values and the model-predicted rating values.
- **Pearson's coefficient (pearsonr)**: It is a measure of the linear correlation between two continuous variables and is used to measure the linear relationship between predicted and actual values. 
- **Spearman's correlation coefficient (spearmanr)**: It is a measure of the monotonic relationship between two variables. Spearman's correlation coefficient is used to assess the performance of a model when dealing with ordered categories or hierarchical data.

