from multiprocessing.pool import Pool
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from search_hn import Hit
import json
import pickle
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("top_words", help="Cleaned data.")
parser.add_argument("saved_stories", help="The saved stories from the HN export.")
parser.add_argument("unsaved_stories", help="The stories grabbed with the training data grabber.")
parser.add_argument("-t","--test",action="store_true", help="Run the precision/recall test suite.")
parser.add_argument("-p","--print",action="store_true")
arguments = parser.parse_args()

with open(arguments.top_words, "rb") as infile:
    top_word_table = pickle.load(infile)
    twt = top_word_table

with open(arguments.saved_stories) as infile:
    saved_stories = json.load(infile)["saved_stories"]

with open(arguments.unsaved_stories, "rb") as infile:
    stories = pickle.load(infile)
    
def gen_story_features(stories):
    stemmer = PorterStemmer()
    for story_i in enumerate(stories):
        story = story_i[1]
        story_features = {}
        try:
            story_features["id"] = story.story_id
        except AttributeError:
            story_features["id"] = story["id"]
        if type(story) == dict:
            story_features["upvoted"] = 1
        else:
            story_features["upvoted"] = 0
        try:
            stemmed_title = stemmer.stem(story.title).split()
        except AttributeError:
            stemmed_title = stemmer.stem(story["title"]).split()
        title_vector = [0] * len(twt)
        for word in stemmed_title:
            if word in twt:
                title_vector[twt[word]] = 1
        story_features["title"] = title_vector
        yield story_features
    
story_features = [0] * (len(saved_stories) + len(stories))
saved_story_features = [0] * len(saved_stories)


word_frequencies = {}
for i in range(len(top_word_table)):
    word_frequencies[i] = 0
    
for feature_set in gen_story_features(saved_stories + stories):
    for feature in enumerate(feature_set["title"]):
        if feature[1]:
            word_frequencies[feature[0]] += 1

ss_word_frequencies = {}
for i in range(len(top_word_table)):
    ss_word_frequencies[i] = 0
    
for feature_set in gen_story_features(saved_stories):
    for feature in enumerate(feature_set["title"]):
        if feature[1]:
            ss_word_frequencies[feature[0]] += 1

def p_of_upvote_given_word(word_i):
    p_of_word = word_frequencies[word_i] / len(story_features)
    p_of_upvote = len(saved_story_features) / len(story_features)
    p_of_word_given_upvote = ss_word_frequencies[word_i] / len(saved_story_features)
    return (p_of_word_given_upvote * p_of_upvote) / p_of_word

p_of_upvote_given_word_vector = np.zeros(len(top_word_table))
 
for word_i in range(len(top_word_table)):
    p_of_upvote_given_word_vector[word_i] = p_of_upvote_given_word(word_i)

def p_of_upvote_given_title(title_vector, upvote_word_p_vector):
    p_updates = title_vector * upvote_word_p_vector
    p_updates = p_updates[p_updates != 0]
    return np.prod(p_updates) / (np.prod(p_updates)
                                 * np.prod(
                                     np.array(
                                         [1 - p_update for p_update in p_updates])))

def p_of_upvote_given_title2(title_vector, upvote_word_p_vector):
    p_updates = title_vector * upvote_word_p_vector
    p_updates = p_updates[p_updates != 0]
    return 1 - np.prod([1 - p_of for p_of in p_updates])     

if arguments.test:
    # Okay now lets figure out precision and recall
    hits = 0
    precision_total = 0
    recall_total = len(saved_stories)
    documents_total = len(stories) + len(saved_stories)
    p_stories = []
    for story in gen_story_features(saved_stories + stories):
        p_of_upvote = p_of_upvote_given_title(story["title"],
                                              p_of_upvote_given_word_vector)
        story["prob"] = p_of_upvote
        if p_of_upvote > 0:
            del(story["title"])
            p_stories.append(story)
    p_stories.sort(key=lambda story: story["prob"])
    from math import floor
    for story in p_stories[-floor(len(p_stories) / 10):]:
        if story["upvoted"]:
            hits += 1
        precision_total += 1

    print("Retrieved: {} documents ({}%), {} hits.".format(precision_total,
                                                           (precision_total /
                                                            documents_total) * 100,
                                                           hits))
    print("Precision: {}% of documents retrieved are relevant.".format((hits /
                                                                        precision_total) *
                                                                       100))
    print("Recall: {}% of relevant documents retrieved.".format((hits/recall_total)*100))

    if arguments.print:
        high_prob_ids = set()
        for story in p_stories[-floor(len(p_stories) / 10):]:
            high_prob_ids.add(story["id"])
        for story in stories:
            if story.story_id in high_prob_ids:
                print("{}\n{}\n".format(story.title,
                                            story.url))
