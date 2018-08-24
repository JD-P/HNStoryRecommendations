from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from search_hn import Hit
import json
import pickle

# Preprocessing stage, get saved stories and HN-search-API data in the same format

with open("2018_08_21_stories_dump.json") as infile:
    saved_stories = json.load(infile)["saved_stories"]

with open("story_dataset.pickle", "rb") as infile:
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
    return 1 - reduce(mul, p_updates)
    
for story in stories:
    p_of_upvote = p_of_upvote_given_title(story.title)
    if p_of_upvote >= .15:
        print("{}: {}%\n{}\n".format(story.title,
                                     p_of_upvote * 100,
                                     story.url))
