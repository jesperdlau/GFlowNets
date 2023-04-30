import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Initial pull from huggingface
# tokenizer = AutoTokenizer.from_pretrained("jiuyiyuyi/esm2_t12_35M_UR50D-Fluorescence-230313")
# model = AutoModelForSequenceClassification.from_pretrained("jiuyiyuyi/esm2_t12_35M_UR50D-Fluorescence-230313")
# tokenizer.save_pretrained("/data/gfp/esm_tokenizer")
# model.save_pretrained("/data/gfp/esm_model")

# Local 
tokenizer = AutoTokenizer.from_pretrained("GFlowNets/data/gfp/esm_tokenizer")  
model = AutoModelForSequenceClassification.from_pretrained("GFlowNets/data/gfp/esm_model") 



def tokens2tokens(torch_batch):
    pass


def oracle(batch):
    with torch.no_grad():
        pred = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    return pred.detach().numpy()



if __name__ == "__main__":
    from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names, Dataset

    dataset = load_dataset("fluorescence")
    dataset_train = dataset["train"]
    dataset_eval = dataset["validation"]
    dataset_test = dataset["test"]

    # Batch pred
    seq = dataset_test['primary'][:10]
    #labels = torch.tensor(dataset_test['log_fluorescence'][:10])
    tok = tokenizer(seq, max_length=300, padding='max_length', return_tensors='pt')

    pred = oracle(tok)
    print(pred)
    print()
