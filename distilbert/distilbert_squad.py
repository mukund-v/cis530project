import torch

from torch import nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class DISTILBERT_SQUAD(nn.Module):
    def __init__(self):
        super(DISTILBERT_SQUAD, self).__init__()

        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.fc_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.criterion = nn.CrossEntropyLoss()



    def forward(self, c_q_pairs, attention_mask, token_type_ids, start_indices, end_indices):

        bert_encoded = self.distilbert_model(
            input_ids=c_q_pairs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]

        fc_output = self.fc_layers(bert_encoded)
        start_outputs, end_outputs = fc_output[:, :, 0].squeeze(-1), fc_output[:, :, 1].squeeze(-1)

        start_indices = (start_indices).clamp(0, start_outputs.shape[1]-1)
        end_indices = (end_indices).clamp(0, start_outputs.shape[1]-1)

        start_loss = self.criterion(start_outputs, start_indices)
        end_loss = self.criterion(end_outputs, end_indices)

        return 2*start_loss + end_loss


    def predict(self, c_q_pairs, attention_mask, token_type_ids):
        bert_encoded = self.distilbert_model(
            input_ids=c_q_pairs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]

        fc_output = self.fc_layers(bert_encoded)
        start_outputs, end_outputs = fc_output[:, :, 0].squeeze(-1), fc_output[:, :, 1].squeeze(-1)

        starts, s_ind = start_outputs.max(1)
        ends, e_ind = end_outputs.max(1)


        answers = []
        for i in range(start_outputs.shape[0]):
            start = s_ind[i].clamp(0, start_outputs.shape[1]-1).item()
            end = e_ind[i].clamp(0, start_outputs.shape[1]-1).item()
            answers.append([start, end])
        return answers
