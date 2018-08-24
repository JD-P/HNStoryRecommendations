# Hacker News Story Recommendations #

## Introduction ##

This library uses Naive Bayes filtering to find stories a Hacker News user would
probably like to read based on stories they've already upvoted.

## Installation ##

### Linux ###

1. Before installing this tool one should go use [HackerNewsToJSON](https://github.com/JD-P/HackerNewsToJSON) to get their upvoted stories. You'll need a decent number to train the classifier. I personally have over 3000, so I'm not entirely sure how many are needed but would be surprised if it's more than 500.

Once that's done, you're ready to use this.

2. git clone this repository:

    git clone git@github.com:JD-P/HNStoryRecommendations.git

3. Copy the JSON file you got from HackerNewsToJSON into the repository.

4. You should probably set up a .gitignore so it's not possible to accidentally
upload your .json or .pickle files.

5. Set up a virtual environment for python3

    virtualenv --python=python3 recommend_env

6. Activate the virtual environment

    source recommend_env/bin/activate

7. Install the following:

    pip install nltk numpy requests

8. This part is real hack-y, and I plan to turn this library into a proper pypy
package later but you need to clone [py-search-hn](https://github.com/JD-P/py-search-hn/) and put its "get_user_comments.py" and "search_hn.py" files into this
repository.

9. You should now be ready to run the program for grabbing training data. We're
grabbing 100 stories from 3 hours before and 3 hours after each of our upvoted
stories as training data. That is, getting things the user *didn't* upvote in
the same timeframe as things they did upvote.

    python3 get_training_set.py your_stories_file.json

Buckle in because it'll take a little while to grab all the stories we want from
the search API. This tool will output the stories to a .pickle file you'll use
for the next step.

10. Finally, run the training program which will train the model and give you
an output of stories above a 15% upvote likelihood threshold:

    python3 train_model.py your_stories_file.json your_training_stories.pickle