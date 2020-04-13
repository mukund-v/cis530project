import collections
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def read_file(filename):
    datastore = None
    if filename:
        with open(filename, 'r') as f:
            return json.load(f)

def simple_baseline(filename):
    datastore = read_file(filename)

    output_questions = []
    output_answers = []

    if datastore is not None:
        for article in datastore["data"]:
            for paragraph in article["paragraphs"]:
                qas = paragraph["qas"]
                question, id, answers, is_impossible = process_question(qas)
                context = paragraph["context"]
                most_freq_word, context_dist = get_context_word_dist(context)
                output_questions.append(question)
                output_answers.append(most_freq_word)
    return output_questions, output_answers

def process_question(qas):
    return qas[0]["question"], qas[0]['id'], qas[0]['answers'], qas[0]["is_impossible"]

def get_context_word_dist(context):
    tokens = word_tokenize(context)

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if ((not w in stop_words) and (w.isalpha()))]

    wlist = []

    #Default to entire context length
    context_depth = len(tokens)

    for i in range(context_depth):
        if tokens[i] not in wlist:
            wlist.append(tokens[i])

    wordfreq = [tokens.count(w) for w in wlist]
    dist = zip(wlist, wordfreq)
    dist_sorted = sorted(dist, key=lambda x: x[1], reverse=True)
    max_token = dist_sorted[0][0]
    # print(max_token)
    return max_token, dist_sorted

if __name__ == '__main__':
    questions, answers = simple_baseline("../data/train-v2.0.json")
    for q, a in zip(questions, answers):
        print(str(q)+"\n", a)