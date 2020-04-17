from models.bert_squad import BERT_SQUAD
import json

bs = BERT_SQUAD()

with open('data/train-v2.0.json', 'r') as f:
    data = json.load(f)['data']

context0, question0 = data[0]['paragraphs'][0]['context'], data[0]['paragraphs'][0]['qas'][0]['question']
context1, question1 = data[1]['paragraphs'][0]['context'], data[1]['paragraphs'][0]['qas'][0]['question']
input = [[context0, question0], [context1, question1]]
bs(input)
