import csv

with open('testing.tsv','rb') as f:			# change to testing.tsv/training.tsv 
	reader=csv.reader(f, delimiter='\t')
	l=list(reader)


out=[]
for i in l:
	if i[3]=="Not Available": 
		continue
	out.append(i)	

with open("cleansedtesting.data", "wb") as f:		# change to cleanedtesting.tsv/cleanedtraining.tsv 
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(out)
