# Twitter-Sentiment-Analysis

In the preprocessed data, the 
(1) URLs are replaced by ||U||
(2) Targets(like @John) are replaced by ||T||
(3) Hashtags are kept as it is and at the end of the preprocessed tweet the number of hashtags in the tweet in encoded by 'h' followed by number of hash tags Ex: h1 for 1 hash tag
(4) The emoticons are replaced by e1 or e0 oe e-1 based od their sentiment. 
(5) The acronyms are expanded using acronym dictionary
(6) Stop word removal, Tokenization, Stemming is done.
