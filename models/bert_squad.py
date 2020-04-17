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

        self.softmax



    def forward(self, c_q_pairs):

        encoded = self.tokenizer.batch_encode_plus(c_q_pairs, pad_to_max_length=True, return_tensors='pt')

        bert_encoded = self.bert_model(
            input_ids=encoded['input_ids'],
            token_type_ids=encoded['token_type_ids']
        )[0]

        fc_output = self.fc_layers(bert_encoded)
        start_outputs, end_outputs = fc_output[:, :, 0].squeeze(-1), fc_output[:, :, 1].squeeze(-1)

        start_probs =
