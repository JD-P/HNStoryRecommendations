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
arguments = parser.parse_args()

# Preprocessing stage, get saved stories and HN-search-API data in the same format

with open(arguments.saved_stories) as infile:
    saved_stories = json.load(infile)["saved_stories"]

with open(arguments.unsaved_stories, "rb") as infile:
    stories = pickle.load(infile)

# Now we have to get them into similar formats

for story in stories:
    story.upvoted = False

ss_objects = []
stories_length = len(stories)
for saved_story in saved_stories:
    ss_object = Hit.make(saved_story)
    ss_object.upvoted = True
    ss_objects.append(ss_object)

# Don't forget to filter any duplicates out of the negative set

ss_id_set = set()
[ss_id_set.add(saved_story.id) for saved_story in ss_objects]
for story in enumerate(stories):
    if story[1].story_id in ss_id_set:
        del(stories[story[0]])
    
# Calculate word frequencies for both datasets

def calculate_word_frequencies(stories):
    stemmer = PorterStemmer()
    word_frequencies = {}
    for story in stories:
        for word in stemmer.stem(story.title).split():
            try:
                word_frequencies[word] += 1
            except KeyError:
                word_frequencies[word] = 1
    return word_frequencies
    
story_word_frequencies = calculate_word_frequencies(stories)
ss_word_frequencies = calculate_word_frequencies(ss_objects)
word_probabilities = {}
num_titles = len(ss_objects) + len(stories)

for word in ss_word_frequencies:
    try:
        word_probabilities[word] = (story_word_frequencies[word]
                                  + ss_word_frequencies[word]) / num_titles
    except KeyError:
        word_probabilities[word] = ss_word_frequencies[word] / num_titles

upvote_word_probabilities = {}
for word in ss_word_frequencies:
    upvote_word_probabilities[word] = ss_word_frequencies[word] / len(ss_objects)
        
def p_of_upvote_given_word(word):
    try:
        p_of_word = word_probabilities[word]
    except KeyError:
        return 0
    p_of_upvote = len(ss_objects) / len(stories)
    p_of_word_given_upvote = upvote_word_probabilities[word]
    return (p_of_word_given_upvote * p_of_upvote) / p_of_word

def p_of_upvote_given_title(title):
    """I'm pretty sure this isn't how you do Bayes so this will probably get updated later"""
    from functools import reduce
    from operator import mul
    stemmer = PorterStemmer()
    p_updates = [1 - p_of_upvote_given_word(word) for word in stemmer.stem(title).split()]
    try:
        return 1 - reduce(mul, p_updates)
    except:
        return 0
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

        
        
#print("{}: {}%\n{}\n".format(story.title,
#                            p_of_upvote * 100,
#                            story.url))
