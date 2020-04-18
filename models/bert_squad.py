import torch

from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer

class BERT_SQUAD(nn.Module):
    def __init__(self):
        super(BERT_SQUAD, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.fc_layers = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.criterion = nn.CrossEntropyLoss()

        #self.softmax



    def forward(self, c_q_pairs, start_indices, end_indices):

        encoded = self.tokenizer.batch_encode_plus(c_q_pairs, pad_to_max_length=True, return_attention_masks=True, return_tensors='pt', max_length=512)

        bert_encoded = self.bert_model(
            input_ids=encoded['input_ids'],
            token_type_ids=encoded['token_type_ids'],
            attention_mask=encoded['attention_mask']
        )[0]

        fc_output = self.fc_layers(bert_encoded)
        start_outputs, end_outputs = fc_output[:, :, 0].squeeze(-1), fc_output[:, :, 1].squeeze(-1)

        start_indices = (1 + start_indices).clamp(0, start_outputs.shape[1]-1)
        end_indices = (1 + end_indices).clamp(0, start_outputs.shape[1]-1)

        start_loss = self.criterion(start_outputs, start_indices)
        end_loss = self.criterion(end_outputs, end_indices)

        return start_loss + end_loss


    def predict(self, c_q_pairs):
        encoded = self.tokenizer.batch_encode_plus(c_q_pairs, pad_to_max_length=True, return_attention_masks=True, return_tensors='pt', max_length=512)

        bert_encoded = self.bert_model(
            input_ids=encoded['input_ids'],
            token_type_ids=encoded['token_type_ids'],
            attention_mask=encoded['attention_mask']
        )[0]

        fc_output = self.fc_layers(bert_encoded)
        start_outputs, end_outputs = fc_output[:, :, 0].squeeze(-1), fc_output[:, :, 1].squeeze(-1)

        starts, ind = start_outputs.max(1)
        ends, ind = end_outputs.max(1)


        answers = []
        for i in range(start_outputs.shape[0]):
            answer = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i])[starts[i]:ends[i]] if starts[i] <= ends[i] else None
            answers.append(answer)

        return answers
