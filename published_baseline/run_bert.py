from models.bert_squad import BERT_SQUAD
from src.utils import get_qa_data
from torch import nn

import json
import random
import torch

bs = BERT_SQUAD()

data = get_qa_data(file='./data/train-v2.0.json')
squad1 = {
    key : data[key] for key in data if data[key]['is_impossible'] == False
}


for p in bs.bert_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(bs.parameters())

keys = squad1.keys()
num_epochs = 1

for epoch in range(num_epochs):
    for i in range(2700):
        bs.train()
        key_batch = random.sample(keys, 32)
        c_q_pairs = [[squad1[key]['context'], squad1[key]['question']] for key in key_batch]

        start_indices = torch.LongTensor([squad1[key]['start_index'] for key in key_batch])
        end_indices = torch.LongTensor([squad1[key]['end_index'] for key in key_batch])

        optimizer.zero_grad()

        loss = bs(c_q_pairs, start_indices, end_indices)
        print('loss on batch {} : {}'.format(i, loss.item()))
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print('-----------------------------------------------------')
            print('Results on two random questions from training set : ')
            key_batch = random.sample(keys, 2)

            c_q_pairs = [[squad1[key]['context'], squad1[key]['question']] for key in key_batch]
            bs.eval()
            answers = bs.predict(c_q_pairs)

            for key in key_batch:
                print('Context: {} \n'.format(squad1[key]['context']))
                print('Question: {} \n'.format(squad1[key]['context']))
                print('Answer: {} \n'.format(squad1[key]['context']))
                print('Predicted answer: {} \n'.format(answers[i]))

            print('-----------------------------------------------------')


torch.save(bs.state_dict(), 'bert_squad.pt')
