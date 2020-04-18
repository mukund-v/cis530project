import json
import argparse




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert json to an training ready file')
    parser.add_argument("--file", "-f", type=str, choices=["train-v2.0", "dev-v2.0"], help="Name of the json ex: train-v2.0")

    args = parser.parse_args()

    with open("../data/{}.json".format(args.file), 'r') as input:
        data = json.load(input)['data']


    contexts_questions = [[qa['context'], qs['question']] for prgrph in data for qa in prgrph['paragraphs'] for qs in qa['qas']]
