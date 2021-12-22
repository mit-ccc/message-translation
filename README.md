# message-translation 

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

An assistive writing tool to analyze linguistic and cultural variation across communities

## Environment Setup

Please run the following:

```
conda create -n message python=3.8
pip install -r requirements.txt
```

## Dataset

You can follow the instructions from the public [BLM Twitter dataset](https://github.com/sjgiorgi/blm_twitter_corpus) to download tweets using our filtered tweetid to generate a smaller dataset which contains ~200K pro-BLM tweets and ~100K anti-BLM tweets. The preprocessing code and data are [here](https://github.com/social-machines/message-translation/tree/main/preprocessing). After that, move the dataset to `./data/blm_alm/raw/` such that you have the following two files: `pro_blm_200k.txt` and `anti_blm_100k.txt`. 


## Semantic Shift Analysis

```
cd semantic_shift
# download BERTweet to your local machine
python download_bertweet.py
sh ./bash_scripts/compute_semantic_shifts.sh
```

Check the [notebook](https://github.com/social-machines/message-translation/blob/main/semantic_shift/analysis/blm-semantic-change-analysis.ipynb) to see the analysis.

## Cultural and Ideological Analysis

```
cd ideology-alignment
sh train_script.sh
```

Check the [notebook](https://github.com/social-machines/message-translation/blob/main/ideology-alignment/src/analysis/Misalignments.ipynb) to see the analysis.

## Acknowledgement

This github is developed on the basis of [UiO-UvA at SemEval-2020 Task 1](https://github.com/akutuzov/semeval2020) and [Aligning Multidimensional Worldviews and Discovering Ideological Differences](https://github.com/jmilbauer/worldview-ideology). 
