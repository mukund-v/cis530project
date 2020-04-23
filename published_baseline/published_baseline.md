# published_baseline.md

Our published baseline model that we implemented uses the pretrained bert base model
and fine tunes it to the SQuAD 2.0 dataset by adding two fully connected layers from
the bert encoding to a size two output for each token, denoting the logits for the token
to be the start and end of the answer. We then take the softmax over these values and output
the starting token with highest probability and the ending token with the highest probability.

We train the model using a cross entropy loss function, using the actual indices of the starting
and ending tokens in the answer as labels in this supervised learning task. To train unanswerable
questions, we simply train the model to output the start token of the sequence, ['CLS']. During inference
time, if the model only outputs the token ['CLS'], or an end token occurring prior to a start token,
we interpret this as unanswerable.


# Training the model from scratch
In this directory, we have provided a Python Notebook 'bert_squad.ipynb' which goes through the training of
the model. We found it to take quite awhile even on the GPU (around 3 hours for two  epochs).

# Downloading our pre-trained parameters
You can download our trained parameters for this model at the link:
'https://drive.google.com/file/d/1-4nyiZckhefZ-iIajpSAlROmLgx9WMrk/view?usp=sharing'

I'll also try uploading to gradescope in this submission, but the file is ~ half a gig.

# Running the baseline model
You can run our model using the 'bert_predictions.py' script. Please pip install torch and transformers libraries.

Note that the script requires:
- model parameters at the path './bert-squad.pt'
- 'dev-v2.0.json' dataset at the path '../data'

The script will save predictions in the file 'bert1-dev-preds.json', which you can run through the evaluation script.


# Model evaluation
The evaluation metrics on testing set for this published baseline model are listed below. Note metrics are given as percentage value.

```json
{
  "exact": 68.04514444538027,
  "f1": 72.53253939979204,
  "total": 11873,
  "HasAns_exact": 58.620107962213226,
  "HasAns_f1": 67.60776658126379,
  "HasAns_total": 5928,
  "NoAns_exact": 77.44322960470984,
  "NoAns_f1": 77.44322960470984,
  "NoAns_total": 5945
}
```

This blows our simple baseline out of the water!
