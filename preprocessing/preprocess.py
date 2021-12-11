from glob import glob
import pandas as pd
import jsonlines
import pprint
from glob import glob
import pandas as pd
from TweetNormalizer import normalizeTweet


with jsonlines.open("anti_blm_100k.jsonl") as f:
    with open("./blm_data/anti_blm_100k.txt", "w") as out:
        for tweet in f:
            tweet_ = normalizeTweet(tweet['full_text'].lower())
            out.write(tweet_)
            out.write("\n")


with jsonlines.open("pro_blm_200k.jsonl") as f:
    with open("./blm_data/pro_blm_200k.txt", "w") as out:
        for tweet in f:
            tweet_ = normalizeTweet(tweet['full_text'].lower())
            out.write(tweet_)
            out.write("\n")