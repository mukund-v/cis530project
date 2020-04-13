import collections
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams


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
    #Removes stopwords in preprocessing (3rd argument)
    one_grams = extract_ngrams(context, 1, True)
    #Does not remove stopwords
    two_grams = extract_ngrams(context, 2, False)

    #Change n_gram setting
    tokens = two_grams

    wlist = []

    for i in len(tokens):
        if tokens[i] not in wlist:
            wlist.append(tokens[i])

    wordfreq = [tokens.count(w) for w in wlist]
    dist = zip(wlist, wordfreq)
    dist_sorted = sorted(dist, key=lambda x: x[1], reverse=True)
    max_token = dist_sorted[0][0]
    # print(max_token)
    return max_token, dist_sorted

def extract_ngrams(data, num, remove_stopwords):
    tokens = word_tokenize(data)

    #Remove punctuation
    tokens = [w for w in tokens if w.isalpha()]

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if (w not in stop_words)]

    n_grams = ngrams(tokens, num)
    return [' '.join(grams) for grams in n_grams]

if __name__ == '__main__':
    questions, answers = simple_baseline("../data/train-v2.0.json")
    for q, a in zip(questions, answers):
        print(str(q)+"\n", a)