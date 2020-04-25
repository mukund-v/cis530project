import json
import torch

from distilbert_squad import DISTILBERT_SQUAD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer



device = torch.device('cuda')
logger = SummaryWriter('logs/distilbert_model')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
feature_processor = SquadV2Processor()
examples = feature_processor.get_train_examples('../data')

features, dataset = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=512,
    doc_stride=128,
    max_query_length=128,
    is_training=True,
    return_dataset="pt",
    threads=1
)

train_loader = DataLoader(dataset=dataset, batch_size=6, shuffle=True)
dev_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
dbs = DISTILBERT_SQUAD().to(device)
num_epochs = 2
optimizer = torch.optim.Adam(dbs.parameters(), lr=.00003)

for epoch in range(num_epochs):
  dbs.train()

  for i, batch in enumerate(train_loader):
    c_q_pairs = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_ids = batch[2].to(device)
    start_ind, end_ind = batch[3].to(device), batch[4].to(device)

    optimizer.zero_grad()
    loss = dbs(c_q_pairs, attention_mask, token_type_ids, start_ind, end_ind)
    logger.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)
    print('loss on batch {} : {}'.format(epoch * len(train_loader) + i, loss.item()))
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
      print('-----------------------------------------------------')
      print('Results on two random questions from training set : ')
      for i in range(2):
        batch = next(iter(dev_loader))
        c_q_pairs = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        start_ind, end_ind = batch[3].to(device), batch[4].to(device)
        s_ind = start_ind.item()
        e_ind = end_ind.item()

        dbs.eval()
        indices = dbs.predict(c_q_pairs, attention_mask, token_type_ids)
        start, end = indices[0][0], indices[0][1]
        print('Context: {} \n'.format(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_q_pairs.view(-1).tolist()))
            ))
        print('Answer: {} \n'.format(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_q_pairs.view(-1).tolist())[s_ind:e_ind+1]) if s_ind <= e_ind else None
        ))
        print('Predicted answer: {} \n'.format(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_q_pairs.view(-1).tolist())[start:end+1]) if start <= end else None
        ))


      print('-----------------------------------------------------')
      dbs.train()


torch.save(dbs.state_dict(), './distilbert-squad.pt')

examples = feature_processor.get_dev_examples('../data')

dbs.eval()

outputs = dict()

for i in range(len(examples)):
  q_id = examples[i].qas_id
  context = examples[i].context_text
  question = examples[i].question_text
  tokenized = tokenizer.encode_plus(question,
                               context,
                               max_length=512,
                               return_tensors='pt')
  c_q_pairs = tokenized['input_ids'].to(device)
  token_type_ids = tokenized['token_type_ids'].to(device)
  attention_mask = tokenized['attention_mask'].to(device)

  indices = dbs.predict(c_q_pairs, attention_mask, token_type_ids)
  start, end = indices[0][0], indices[0][1]
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_q_pairs.view(-1).tolist())[start:end+1]) if start <= end else ""
  if '[CLS]' in answer:
    answer = ""
  outputs[q_id] = answer

  if i % 100 == 0:
    print('done with example : {}'.format(i))

with open('distilbert1-dev-preds.json', 'w') as f:
  json.dump(outputs, f)
