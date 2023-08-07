#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import torch
import sentencepiece
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from time import sleep


# In[2]:

print(torch.cuda.is_available())


# In[14]:

data = pd.read_csv("/home4/s4236599/speedrun_trial/squad_speedrun.csv")
data.columns = ["source_text", "target_text"]
data.sample(frac=1)
data = data.dropna()
data = Dataset.from_pandas(data)

data = data.train_test_split(test_size=0.3)
data["test"] = data["test"].train_test_split(test_size=0.5)


# In[15]:


train_source = data["train"]["source_text"]
train_target = data["train"]["target_text"]

test_source = data["test"]["train"]["source_text"]
test_target = data["test"]["train"]["target_text"]

val_source = data["test"]["test"]["source_text"]
val_target = data["test"]["test"]["target_text"]


# In[5]:


tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[16]:


train_encodings = tokenizer(train_source, truncation = True, max_length = 1024, padding=True)
train_decodings = tokenizer(train_target, truncation = True, padding=True)

test_encodings = tokenizer(test_source, truncation = True, max_length = 1024, padding=True)
test_decodings = tokenizer(test_target, truncation = True, padding=True)


# In[20]:


train_input_ids = torch.tensor(train_encodings["input_ids"])
train_attention_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_decodings['input_ids'])

train_data = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = torch.utils.data.RandomSampler(train_data)
train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=2)


# In[21]:


test_input_ids = torch.tensor(test_encodings["input_ids"])
test_attention_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_decodings['input_ids'])

test_data = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=2)


# In[22]:


train_loss_his = []
test_loss_his = []

    
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base")

epochs = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)

for epoch in range(epochs):
    print(str(epoch)+"\n")
    model.train()
    train_loss = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:    
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

            loss = outputs.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss= loss.item())
        train_loss /= len(train_dataloader)
        sleep(0.1)
    print(loss)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        with tqdm(test_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)
                outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                test_loss += loss.item()
                tepoch.set_postfix(loss= loss.item())

            # Compute the average testing loss for the epoch
            test_loss /= len(test_dataloader)
    print(loss)
    print(f'train_loss = {train_loss}\n test_loss = {test_loss}')

    train_loss_his.append(train_loss)
    test_loss_his.append(test_loss)
model.save_pretrained(f'/home4/s4236599/speedrun_trial/longt5/longt5_epoch{epoch+1}')
loss_df = pd.DataFrame(list(zip(train_loss_his, test_loss_his)))
loss_df.to_csv("/home4/s4236599/speedrun_trial/longt5_loss.csv", index = False)
