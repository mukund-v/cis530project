# simple-baseline.md

Our simple baseline simply outputs the most common one or two gram from the given
context paragraph as the answer to the question regardless of the given question.

Note that our model removes stopwords in the case of one-grams but not two-grams
as stopwords may be a part of a two-gram entity (whereas stopwords are very rarely
answers themselves in the case of one-grams).

# Running the baseline model
We can run the baseline using the following command:

```
python simple-baseline.py
```

Note that this file expects the input data at the filepath "../data/train-v2.0.json"
Change this in the main function as necessary. After running the baseline model on the
input data, the program will ask in the command line for the filepath at which you wish
to output the baseline predictions in JSON format. You can directly use the predictions file produced by this in the evaluation script.


# Model evaluation
The evaluation metrics for this simple baseline model are listed below. Note metrics are given as percentage value.

```json
{
  "exact": 0.5593965576776986,
  "f1": 3.3521969832557006,
  "total": 130319,
  "HasAns_exact": 0.8338996325773719,
  "HasAns_f1": 5.025914912992246,
  "HasAns_total": 86821,

  "NoAns_exact": 0.011494781369258357,
  "NoAns_f1": 0.011494781369258357,
  "NoAns_total": 43498
}
```
