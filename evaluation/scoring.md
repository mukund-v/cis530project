Evaluation

score.py - script to evaluate predictions. Comes from the website.






## Running the Evaluation Script

We can run the evaluation script on a given data and prediction file using the following
command line command:

```
python PATH/TO/score.py data_file prediction_file
```

# Input data file
the data_file is a JSON file containing context paragraph and q/a pairs structured as following:

```json
{
  "version": "v2.0",
  "data": [{
    title: ARTICLE_TITLE,
    paragraphs: [{
      qas: [{
        question: QUESTION_TEXT,
        id: HASH_ID,
        answers: [POSSIBLE_ANSWERS],
        is_impossible: BOOLEAN
      }],
      context: PARAGRAPH_CONTENTS
    }]
  }]
}
```

# Prediction file
the prediction file is a JSON file containing mappings from question id in the dataset to the answer produced by the model. the prediction model should simply be structured as below. See out baseline outputs for an example.
```json
{
  question_id : answer
}
```

# Metrics
The evaluation statistics are broken into two sets of scores and then combined using a weighted average: one for questions that are answerable with a certain probability (HasAns below), and questions that are not answerable with a certain probability (NoAns below). For each of these sets two scores are given: one for the percent of questions that were answered exactly correct (after some normalization like casing and removing whitespace) ({}_exact below), and another that is an f1 score based on common tokens between answers ({}_f1 below). 

These two sets are combined with a weighted average based on the number of answerable (HasAns_total below) and unanswerable questions (NoAns_total below) as a fraction of the total questions (total below). This provides an exact score and f1 score for the entire dataset ("exact", "f1" below).


# Output
We can use the --out-file flag to provide an output filepath to dump the evaluation JSON.

Evaluation statistics are printed to the console.

Example evaluation output from our baseline model is shown below. Note metrics are given as percentage value.

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
