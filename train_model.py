from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from search_hn import Hit
import json
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("saved_stories", help="The saved stories from the HN export.")
parser.add_argument("unsaved_stories", help="The stories grabbed with the training data grabber.")
parser.add_argument("-t","--test",action="store_true", help="Run the precision/recall test suite.")
parser.add_argument("-d","--debug",action="store_true", help="Print debug output to console.")
arguments = parser.parse_args()

# Preprocessing stage, get saved stories and HN-search-API data in the same format

with open(arguments.saved_stories) as infile:
    saved_stories = json.load(infile)["saved_stories"]

with open(arguments.unsaved_stories, "rb") as infile:
    stories = pickle.load(infile)

# Now we have to get them into similar formats

# Lets add upvotes to the model

def class_given_upvote_tally(upvotes):
    """Given an integer number of upvotes, return an integer class corresponding to
    the range of upvotes for classification."""
    class_ranges = [2,10,25,50,100,200,500]
    for class_range in enumerate(class_ranges):
        if upvotes <= class_range[1]:
            return class_range[0]
    return 6

story_upvote_class_counts = {}
for story in stories:
    story.upvoted = False
    story.upvote_class = class_given_upvote_tally(story.points)
    try:
        story_upvote_class_counts[story.upvote_class] += 1
    except KeyError:
        story_upvote_class_counts[story.upvote_class] = 1

        
ss_objects = []
stories_length = len(stories)
ss_upvote_class_counts = {}
for saved_story in saved_stories:
    ss_object = Hit.make(saved_story)
    ss_object.upvoted = True
    ss_object.upvote_class = class_given_upvote_tally(saved_story["score"])
    try:
        ss_upvote_class_counts[ss_object.upvote_class] += 1
    except KeyError:
        ss_upvote_class_counts[ss_object.upvote_class] = 1
    ss_objects.append(ss_object)

    
# Don't forget to filter any duplicates out of the negative set

ss_id_set = set()
[ss_id_set.add(saved_story.id) for saved_story in ss_objects]
story_id_set = set()
for story in enumerate(stories):
    if story[1].story_id in ss_id_set:
        del(stories[story[0]])
    elif story[1].story_id in story_id_set:
        del(stories[story[0]])
    else:
        story_id_set.add(story[1].story_id)
        
# Calculate word frequencies for both datasets

def calculate_word_frequencies(stories):
    stemmer = PorterStemmer()
    word_frequencies = {}
    for story in stories:
        for word in stemmer.stem(story.title).split():
            if word in stopwords:
                continue
            try:
                word_frequencies[word] += 1
            except KeyError:
                word_frequencies[word] = 1
    return word_frequencies
    
word_frequencies = calculate_word_frequencies(stories + ss_objects)
ss_word_frequencies = calculate_word_frequencies(ss_objects)
word_probabilities = {}
num_titles = len(ss_objects) + len(stories)

for word in word_frequencies:
    word_probabilities[word] = word_frequencies[word] / num_titles

upvote_word_probabilities = {}
for word in ss_word_frequencies:
    upvote_word_probabilities[word] = ss_word_frequencies[word] / len(ss_objects)
        
def p_of_upvote_given_word(word):
    p_of_word = word_probabilities[word]
    p_of_upvote = len(ss_objects) / len(stories)
    try:
        p_of_word_given_upvote = upvote_word_probabilities[word]
    except:
        p_of_word_given_upvote = 1
    return (p_of_word_given_upvote * p_of_upvote) / p_of_word

def p_of_upvote_given_title(title):
    """I'm pretty sure this isn't how you do Bayes so this will probably get updated later"""
    from functools import reduce
    from operator import mul
    stemmer = PorterStemmer()
    words = [word for word in stemmer.stem(title).split() if word not in stopwords]
    p_updates = np.array(
        [p_of_upvote_given_word(word) for word in words]
        )
    return (np.prod(p_updates) /
            (np.prod(p_updates)
             + np.prod(np.array([1 - p_of for p_of in p_updates]))
            ))

def p_of_upvote_given_score(score_class):
    p_of_upvote = len(ss_objects) / len(stories)
    p_of_score = (story_upvote_class_counts[score_class]
                  + ss_upvote_class_counts[score_class]) / (len(ss_objects)
                                                            + len(stories))
    p_of_score_given_upvote = ss_upvote_class_counts[score_class] / len(ss_objects)
    return (p_of_score_given_upvote * p_of_upvote) / p_of_score

top_1000_words_list = [(word, word_frequencies[word]) for word in word_frequencies]
top_1000_words_list.sort(key=lambda key_value: key_value[1])
#TODO: what if we don't have 1000 words?
top_1000_words_list = top_1000_words_list[-1000]


if arguments.debug:
    for story in stories[0:100]:
        p_of_upvote = p_of_upvote_given_title(story.title)
        print("{}: {}%\n{}\n".format(story.title,
                                     p_of_upvote * 100,
                                     story.url))

    
if arguments.test:
    # Okay now lets figure out precision and recall
    hits = 0
    precision_total = 0
    recall_total = len(ss_objects)
    for story in stories + ss_objects:
        p_of_upvote = p_of_upvote_given_title(story.title)
        if p_of_upvote >= .15:
            if story.upvoted:
                hits += 1
                precision_total += 1
            else:
                precision_total += 1

    print("Precision: {}% of documents retrieved are relevant.".format((hits /
                                                                        precision_total) *
                                                                       100))
    print("Recall: {}% of relevant documents retrieved.".format((hits/recall_total)*100))

        
        
