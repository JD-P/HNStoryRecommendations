import argparse
import time
import pickle
import json
from json.decoder import JSONDecodeError
from search_hn import SearchHN

parser = argparse.ArgumentParser()
parser.add_argument("saved_stories", help="JSON file of the saved stories from HN export.")
arguments = parser.parse_args()

hn = SearchHN()
six_hours = 60 * 60 * 6

with open(arguments.saved_stories) as infile:
    dump = json.load(infile)

results = []
ss_number = 1
for saved_story in dump["saved_stories"]:
    timeline_after = saved_story["time"] - six_hours
    timeline_before = saved_story["time"] + six_hours
    try:
        results_intermediate = (hn
                                .created_between(timeline_after,timeline_before)
                                .stories()
                                .hits_per_page(100)
                                .get())
    except JSONDecodeError:
        print("Issue with story {}: {}".format(saved_story["id"],
                                               saved_story["title"]))
        time.sleep(30)
    results += results_intermediate
    print("Saved story neighbors collected ({} of {}: {} results)".format(
        ss_number,
        len(dump["saved_stories"]),
        len(results)))
    ss_number += 1
    time.sleep(0.5)
    
with open("story_dataset.pickle", 'wb') as outfile:
    pickle.dump(results, outfile)
