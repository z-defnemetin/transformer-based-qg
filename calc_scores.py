#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
from datasets import load_metric
from transformers import AutoTokenizer
import rouge_score
import torch
from tqdm import tqdm
import spacy
import spacy_ke


# In[2]:


print(torch.cuda.is_available())


# In[10]:


rootdir = '/home4/s4236599/Models/predictions'
preds = []

for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    preds.append(d)


# In[11]:


print(preds)


# In[5]:


final_df = pd.DataFrame(columns=['Model','ROUGE','Bleu','Meteor','Keyword'])


# In[26]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

rouge_list = []
bleu_list = []
meteor_list = []
keyword_list = []
model_list = []

rouge = load_metric("rouge")
bleu = load_metric("bleu")
meteor = load_metric("meteor")


# In[117]:


def keyword(article, prediction):
    total_score = 0
    
    for article_text, prediction_text in tqdm(zip(article, prediction)):
        match = 0
        key_source = extract_keywords(article_text)
        key_target = extract_keywords(prediction_text)
        
        for elem in key_target:
            if elem in key_target and elem in key_source:
                match += 1
                        
        if len(key_target) > 0:  
            match /= len(key_target)
            total_score += match
        print(total_score)
    return total_score

def extract_keywords(text):
    doc = nlp(str(text))
    temp = doc._.extract_keywords(n = len(text)//2)
    return list(dict.fromkeys(key[0].text for key in temp))


# In[118]:


def compute_scores(data):
    validation = data.iloc[:,0]
    predictions = data.iloc[:,1]
    references = data.iloc[:,2]
    
    prediction_tok = []
    for p in predictions:
        tokens = tokenizer.tokenize(p)
        prediction_tok.append(tokens)
        
    reference_tok = []
    for r in references:
        tokens = tokenizer.tokenize(r)
        reference_tok.append([tokens])

    rouge_score = rouge.compute(predictions=predictions, references=references)
    print("rouge metric is done\n")
    
    bleu_score = bleu.compute(predictions=prediction_tok, references=reference_tok)
    print("bleu metric is done\n")

    meteor_score = meteor.compute(predictions=predictions, references=references)
    print("meteor metric is done\n")

    keyword_score = keyword(validation, predictions)/len(predictions)
    print("keyword metric is done!")
    
    return rouge_score, bleu_score, meteor_score, keyword_score


# In[116]:


spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake")

tokenizer = AutoTokenizer.from_pretrained("t5-base")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

rouge_list = []
bleu_list = []
meteor_list = []
keyword_list = []
model_list = []

rouge = load_metric("rouge")
bleu = load_metric("bleu")
meteor = load_metric("meteor")

for pred in preds:
    model_list.append(pred)
    pred = pd.read_csv(pred)
    rouge_score, bleu_score, meteor_score, keyword_score = compute_scores(pred)
    rouge_list.append(rouge_score)
    bleu_list.append(bleu_score)
    meteor_list.append(meteor_score)
    keyword_list.append(keyword_score)

final_df["Meteor"] = meteor_list
final_df["Bleu"] = bleu_list
final_df["ROUGE"] = rouge_list
final_df["Model"] = preds
final_df["Keyword"] = keyword_list

final_df.to_csv("/home4/s4236599/Models/predictions/final_comparison.csv", index = False)

