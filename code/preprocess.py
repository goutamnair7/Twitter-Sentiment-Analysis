# encoding=utf8  
import sys  
#sys.setdefaultencoding('utf8')
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
stemmer = PorterStemmer()

stopWords_dict = {}
emoticonSent_dict = {}
acronym_dict = {}
punctuation_list = [',','.',':','...','..','!','-',";",":","'",'"',"&","?","(",")","[","]","{","}","``","''"] 

def getDicts():
	#stop words dictionary
	f1 = open("../data/StopWords.txt",'r')
	lines1 = f1.read().splitlines()
	stopWords_dict = dict((k,1) for k in lines1)

	#emoticon sentiment dictionary
	f2 = open("../data/EmoticonSentimentLexicon.txt",'r')
	lines2 = f2.read().splitlines()
	for i in lines2:
		i = i.split('\t')
		emoticonSent_dict[i[0]] = i[1]

	#acronym dictionary
	f3 = open("../data/InternetSlangAcronyms.txt",'r')
	lines3 = f3.read().splitlines()
	for i in lines3:
		i = i.split('\t')
		acronym_dict[i[0]] = i[1]

def read_file(filename):
	f = open("../data/" + filename,'r')
	lines = f.read().splitlines()
	l = len(lines)
	for i in range(0,l):
		lines[i] = lines[i].split('\t')
	return lines

def preprocess(lines, newfile):
	f = open("../data/" + newfile, 'w+')
	l = len(lines)
	for i in range(0,l):
		if len(lines[i]) == 5:
			if lines[i][-1] != "Not Available":
				lines[i][-2] += lines[i][-1]
				lines[i][-1] = ''
			else:
				sent = lines[i][-2].split(' ')
		else: 
			sent = lines[i][-1].split(' ') 
		#sent = word_tokenize(sent)
		l_sent = len(sent)
		final_sent = []
		j = 0
		num_htags = 0
		while(1):
			if j >= l_sent:
				break
					
			#expand word if it is an acronym
			try:
				if acronym_dict[sent[j]]:
					sent[j] = acronym_dict[sent[j]].split(' ')
					sent = sent[0:j] + sent[j] + sent[j+1:]
					l_sent = len(sent)
			except:
				pass	

			#replace targets(i.e. usernames) with ||T||
			target_pattern = re.compile("@{1}.")
			if target_pattern.match(sent[j]):
				sent[j] = "||T||"
				final_sent.append(sent[j])
				j += 1
				if j == l_sent:
					break
				continue
			
			#keep hashtags intact and count them
			hashtag_pattern = re.compile("#{1}.")
			if hashtag_pattern.match(sent[j]):
				num_htags += 1
				j += 1
				if j == l_sent:
					break
				continue

			#replace URLs by ||U||
			url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
			if url_pattern.match(sent[j]):
				sent[j] = "||U||"
				final_sent.append(sent[j])
				j += 1
				if j == l_sent:
					break
				continue

			#replace emoticons with sentiment value : e0 or e1 or e-1
			try:
				if emoticonSent_dict[sent[j]]:
					sent[j] = 'e' + emoticonSent_dict[sent[j]]
					final_sent.append(sent[j])
					j += 1
					if j == l_sent:
						break
					else:
						continue
			except:
				pass

			#remove stop words
			try:
				if stopWords_dict[sent[j]]:
					j += 1
					if j == l_sent:
						break
					else:
						continue
			except:
				pass

			#tokenize the word
			sent[j] = word_tokenize(sent[j])
			sent = sent[0:j] + sent[j] + sent[j+1:]
			l_sent = len(sent)
		
			#stem the words
			if j == l_sent:
				break
			sent[j] = stemmer.stem(sent[j])

			#@ with a aspace denotes 'at' a place or time
			if sent[j] == "@":
				sent[j] = "at"

			#discard the tokens which are punctuation marks
			try:
				if sent[j] in punctuation_list: #punctuation_dict[sent[j]]: # or re.compile("\.+").match(sent[j])):
					j += 1
					if j == l_sent:
						break
					else:
						continue
				elif sent[j][-1] in punctuation_list:
					sent[j] = sent[j][:-1]
					final_sent.append(sent[j])
				else:
					final_sent.append(sent[j])		
			except:
				pass
			#final_sent.append(sent[j])

			j += 1
			if j == l_sent:
				break

		final_sent.append("h" + str(num_htags))
		lines[i][3] = ' '.join(final_sent)

	for i in range(0,l):	
		temp_string = ''
		temp_len = len(lines[i])
		for k in range(0,temp_len):
			if k!=temp_len-1:
				temp_string += lines[i][k] + '\t'
			else:
				temp_string += lines[i][k]
		temp_string + '\n'
		f.write(temp_string + "\n")

def main():
	getDicts()
	file1 = "cleanedtraining.data"
	file2 = "cleanedtesting.data"
	#preprocess(read_file(file1), "preprocessedTraining.data")
	preprocess(read_file(file2), "preprocessedTesting.data")

if __name__ == '__main__':
	main()