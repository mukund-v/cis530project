import json
import torch

from bert_squad import BERT_SQUAD
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor
from transformers import BertModel, BertConfig, BertTokenizer


bs1 = BERT_SQUAD()
bs1.load_state_dict(torch.load('./bert-squad.pt'))

feature_processor = SquadV2Processor()
examples = feature_processor.get_dev_examples('../data')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bs1 = bs1.to(device)
bs1.eval()

outputs = dict()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for i in range(len(examples)):
    q_id = examples[i].qas_id
    context = examples[i].context_text
    question = examples[i].question_text
    tokenized = tokenizer.encode_plus(question, context, max_length=512, return_tensors='pt')
    c_q_pairs = tokenized['input_ids'].to(device)
    token_type_ids = tokenized['token_type_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    indices = bs1.predict(c_q_pairs, attention_mask, token_type_ids)
    start, end = indices[0][0], indices[0][1]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c_q_pairs.view(-1).tolist())[start:end+1]) if start <= end else ""
    if '[CLS]' in answer:
        answer = ""
    outputs[q_id] = answer

    if i % 100 == 0:
        print('done with example : {}'.format(i))

with open('bert1-dev-preds.json', 'w') as f:
    json.dump(outputs, f)
