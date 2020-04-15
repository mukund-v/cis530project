# data.md

### Our data comes from The Stanford Question Answering Dataset at https://rajpurkar.github.io/SQuAD-explorer/. Since this is a shared task we are already provided with a public test and dev dataset and there is a hidden test dataset which we can test out model against through an upload portal. However for the purposes of creating our model we plan to use the provided dev set as our test data and perform k-fold cross validation on the train set to provide us with a 80:20 split of train:dev data.

## Train / Dev Data (train-v2.0.json)
The train data itself is composed of 130,319 question and answers that span across 442 articles. We plan to split it 80:20 such that 104,255 questions are used to trian on and 26,064 are used for Dev.

## Test Data (dev-v2.0.json)
We will be using the provided dev dataset of 11,873 questions and answers as our test dataset for ease of access which spans 35 different articles.

## Data Format
All data is a .json object structured like so

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

For each paragraph within each article we have a list of questions with possible answers, unique question IDs, a boolean indiciating if an answer exists, and the paragraph contents intself.
