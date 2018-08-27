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
parser.add_argument("-s", "--save_model",help="Filepath to save the trained model to.")
parser.add_argument("-u", "--use_model", help="Filepath to read a saved model in from.")
arguments = parser.parse_args()

    # Preprocessing stage, get saved stories and HN-search-API data in the same format

with open(arguments.saved_stories) as infile:
    saved_stories = json.load(infile)["saved_stories"]

with open(arguments.unsaved_stories, "rb") as infile:
    stories = pickle.load(infile)

stoplist = stopwords.words()
    
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

class FeatureVectors:
    def __init__(self, stories, ss_objects):
        ss_word_frequencies = self.calculate_word_frequencies(ss_objects)
        word_tuples = [(word,ss_word_frequencies[word]) for word in ss_word_frequencies]
        word_tuples.sort(key=lambda pair: pair[1])
        top_words = word_tuples[-4000:]
        self.top_word_table = {}
        for pair in enumerate(top_words):
            self.top_word_table[pair[1][0]] = pair[0]
        
        
    def calculate_word_frequencies(self, stories):
        stemmer = PorterStemmer()
        stoplist = stopwords.words()
        word_frequencies = {}
        for story in stories:
            for word in stemmer.stem(story.title).split():
                if word in stoplist:
                    continue
                try:
                    word_frequencies[word] += 1
                except KeyError:
                    word_frequencies[word] = 1
        return word_frequencies
    
fv = FeatureVectors(stories, ss_objects)
twt = fv.top_word_table
stemmer = PorterStemmer()

saved_story_features = []
for story_i in enumerate(ss_objects):
    story = story_i[1]
    story_features = {}
    story_features["id"] = story.id
    if story.upvoted:
        story_features["upvoted"] = 1
    else:
        story_features["upvoted"] = 0
    stemmed_title = stemmer.stem(story.title).split()
    title_vector = np.zeros(len(twt))
    for word in stemmed_title:
        if word in twt:
            title_vector[twt[word]] = 1
    story_features["title"] = title_vector
    saved_story_features.append(story_features)

#unsaved_story_features = []
#for story_i in enumerate(stories):
#    story = story_i[1]
#    story_features = {}
#    story_features["id"] = story.story_id
#    if story.upvoted:
#        story_features["upvoted"] = 1
#    else:
#        story_features["upvoted"] = 0
#    stemmed_title = stemmer.stem(story.title).split()
#    title_vector = [0] * len(twt)
#    for word in stemmed_title:
#        if word in twt:
#            title_vector[twt[word]] = 1
#    story_features["title"] = title_vector
#    unsaved_story_features.append(story_features)

with open("story_features.pickle", "wb") as outfile:
    output = twt
    pickle.dump(output, outfile)

