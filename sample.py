# -*- coding: utf-8 -*-
'''
Johannes Deselaers
Mateusz Buda
Olga Mikheeva
Diego A. Mancevo
'''
# DB Sampling, adding marker for train/val/test/ dataset, stemming, stopwords removal
import sqlite3
import os.path
import random
import re
from datetime import datetime as dt
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

originalDB = '/Users/Olga/Downloads/data/database.sqlite'
sampleDB = '/Users/Olga/Downloads/data/sample_db3.sqlite'

def processing(result):
    result = list(result)
    doc = result[4]
    doc = re.sub(u'[^a-zA-Z]+', ' ', doc.lower())
    result[4] = ' '.join([stemmer.stem(t) for t in word_tokenize(doc) if t not in stop])
    #adding marker for train/val/test dataser
    rand  = random.randint(1, 10)
    dataSetType = 0 if rand < 7 else (1 if rand < 9 else 2) # 0 - test, 1- val, 2 - test
    result.append(dataSetType)
    return result

def ResultIter(cursor, arraysize=1000):
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        yield results
        

start = dt.now()
print("start " + str(start))

conn = sqlite3.connect(originalDB)
cur = conn.cursor()

# get top subreddits
cur.execute("select subreddit, count(*) as num_of_comments from May2015\
        group by subreddit\
        order by num_of_comments desc")
subreddits = cur.fetchall()
top = [sr[0] for sr in subreddits[:7]]

#create sample db
f = open(sampleDB, 'w')
f.close()

conn2 = sqlite3.connect(sampleDB)
cur2 = conn2.cursor()

#create table in sample db
createTableSQL = "CREATE TABLE May2015(link_id, name, subreddit, id, body, parent_id, dataset int)"
cur2.execute(createTableSQL)

stemmer = PorterStemmer()
stop = stopwords.words('english')

selectSQL = 'select link_id, name, subreddit, id, body, parent_id from May2015 \
       where subreddit in ({seq}) and distinguished is NULL \
       and removal_reason is NULL and body is not "[deleted]"'.format(seq = ','.join(['?']*len(top)))
cur.execute(selectSQL, top)

size = 1000
c = 0
insertSQL = 'insert into May2015 values({seq})'.format(seq = ','.join(['?']*7))
for result in ResultIter(cur, size):
    result = map(processing, result)
    cur2.executemany(insertSQL, result)
    c += size
    if (c % 100000 == 0):
        conn2.commit()
        print(c)

conn2.commit()
conn2.close()
conn.close()
print("finish " + str(dt.now()))
print("total running time " + str(dt.now() - start))