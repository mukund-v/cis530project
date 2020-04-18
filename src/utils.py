import json
import argparse
import os




# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description='Convert json to an training ready file')
#     parser.add_argument("--file", "-f", type=str, choices=["train-v2.0", "dev-v2.0"], help="Name of the json ex: train-v2.0")
#
#     args = parser.parse_args()


def get_qa_data(file):
    with open(file, 'r') as input:
        data = json.load(input)['data']



    questions = dict()
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                info = {
                    'context' : paragraph['context'],
                    'question' : qa['question'],
                    'answer' : qa['answers'][0]['text'] if not qa['is_impossible'] else None,
                    'start_index' : qa['answers'][0]['answer_start'] if not qa['is_impossible'] else None,
                    'end_index' : qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text']) if not qa['is_impossible'] else None,
                    'is_impossible' : qa['is_impossible'],
                }
                questions[qa['id']] = info
    return questions
