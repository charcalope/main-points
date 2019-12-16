import spacy
from numpy import argmax
from spacy.lang.en import English
from allennlp.predictors.predictor import Predictor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def pipeline(given_file):
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
    phrases_in_file = []
    sents_in_file = []

    def sentences(filename):
        def read_data():
            with open(filename, "r") as f:
                string = f.read()
            return string

        # setup spacy pipeline
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)

        # run nlp pipeline on given data
        doc = nlp(read_data())
        return [str(a) for a in doc.sents]

    def parse_phrases(sentence):
        result = predictor.predict(sentence)
        phrases = []
        for thing in result['verbs']:
            raw_line = thing['description']
            phrase = []

            while '[' in raw_line:
                # extract from line with brace indexes
                raw_term = raw_line[raw_line.index('['):raw_line.index(']')+1]
                # clean up braces and tags
                clean_term = raw_term[raw_term.find(':')+2:-1]
                # add term to phrase
                phrase.append(clean_term)
                # must extract unedited term
                raw_line = raw_line.replace(raw_term, '')
            phrases.append(' '.join(phrase)) # join terms into single phrase
        return phrases

    def kmeans(corpus) -> list:
        vectorizer = TfidfVectorizer(max_features=13)
        X = vectorizer.fit_transform(corpus)

        # cluster
        num_clusters = 3
        km = KMeans(n_clusters=num_clusters)
        km.fit(X)

        # order and get labels
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        clusters = []
        for i in range(num_clusters):
            cluster = []
            for ind in order_centroids[i, :4]:
                cluster.append(terms[ind])
            clusters.append(cluster)

        return clusters

    sents_in_file = sentences(given_file)
    # this for loop avoids an expected
    for sent in sents_in_file:
        phrases_in_file.extend(parse_phrases(sent))
    clusters = kmeans(phrases_in_file)

    result = dict()
    result['sentences'] = sents_in_file
    result['phrases'] = phrases_in_file
    result['clusters'] = clusters

    return result

def pipeline_2(result_dict):
    key_phrases = []

    def doc_sim(sentences, topic_terms):
        nlp = spacy.load('en_core_web_lg')
        # term list to string
        topic = nlp(' '.join(topic_terms))
        scores = [topic.similarity(nlp(sentence)) for sentence in sentences]
        # most similar document
        return sentences[argmax(scores)]

    for cluster in result_dict['clusters']:
        key_phrase = doc_sim(result_dict['sentences'], cluster)
        key_phrases.append(key_phrase)

    return key_phrases


if __name__ == '__main__':
    result = pipeline('sample.txt')
    k_p = pipeline_2(result)
    print(k_p)
