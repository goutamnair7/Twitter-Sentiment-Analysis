# Twitter-Sentiment-Analysis

In the preprocessed data, the 
(1) URLs are replaced by ||U||
(2) Targets(like @John) are replaced by ||T||
(3) Hashtags are kept as it is and at the end of the preprocessed tweet the number of hashtags in the tweet in encoded by 'h' followed by number of hash tags Ex: h1 for 1 hash tag
(4) The emoticons are replaced by e1 or e0 oe e-1 based od their sentiment. 
(5) The acronyms are expanded using acronym dictionary
(6) Stop word removal, Tokenization, Stemming is done.

##
In Feature Extraction a dictionary is created of feature vectors and labels. Feature vector is in the form of a list of 0's and 1's where 0 means a word is not present in the tweet and 1 means it is present#
##
Labels : 
##
0 - positive
#
1 - negative
#
2 - neutral
#
