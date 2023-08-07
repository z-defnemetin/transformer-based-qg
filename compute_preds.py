#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
from transformers import (AutoTokenizer, AutoModelForTokenClassification, pipeline,
                          BartForConditionalGeneration, LongT5ForConditionalGeneration,
                         T5ForConditionalGeneration, PegasusForConditionalGeneration)
import pandas as pd
from datasets import load_metric
import rouge_score
import torch
from tqdm import tqdm


# In[21]:


print(torch.cuda.is_available())


# In[22]:


rootdir = '/home4/s4236599/speedrun_trial/longt5'
models = []

for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        models.append(d)


# In[23]:


print(models)


# In[24]:


final_df = pd.DataFrame(columns=['Model','ROUGE','Bleu','Meteor','Keyword'])


# In[25]:


validation_set = pd.read_csv("validation_set.csv")


# In[26]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

validation = validation_set.iloc[:,0]
references = validation_set.iloc[:,1]


# In[ ]:


for i in range(len(models)):
    print(models[i])
    
    if "bart" in models[i]:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        model_ft = BartForConditionalGeneration.from_pretrained(models[i])
    elif "pegasus" in models[i]:
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        model_ft = PegasusForConditionalGeneration.from_pretrained(models[i])        
    elif "epoch" in models[i]:
        tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
        model_ft = LongT5ForConditionalGeneration.from_pretrained(models[i])
#     elif "t5" in models[i]:
#         tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         model_ft = T5ForConditionalGeneration.from_pretrained(models[i])
    else:
        continue

    predictions = []
    model_ft.to(device)

    for item in tqdm(validation):
        input_ids = tokenizer(item, return_tensors='pt', truncation = True).to(device)
        question_ids = None
        question_ids = model_ft.generate(**input_ids, max_new_tokens = 256)
        question = tokenizer.batch_decode(question_ids, skip_special_tokens=True)
        predictions.append(question)
        
    print("predictions are done.\n")    
    eval_res = pd.DataFrame(list(zip(validation, predictions, references)), columns = ["article", "prediction", "reference"])
    eval_res.to_csv(f"{models[i]}_predict.csv", index = False)
