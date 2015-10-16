
0. Install following packages for python (2.7):
	* numpy
	* scipy
	* nltk
	* sklearn

1. Download the database (7.9 Gb compressed) from 
https://www.kaggle.com/c/reddit-comments-may-2015/download/database.7z and unzip it.

2. Select the subset with top 7 categories:
	* Change variables 'originalDB' and 'sampleDB' in sample.py to contain pathes to original db downloaded in the 1st step and new db respectivly.
	* Run sample.py
	* Now you have preprocessed sample of data

Or alternativelty (to 1 and 2) you can download our sample from:
https://drive.google.com/file/d/0B0eFSS1XBnpMSExMeXh5Ujltc28/view?usp=sharing

3. Run classifier:
	* Change variable db_path in classirier.py to contain path to sample db.
	* Run classifier.py


nlpNet_demo.py contains simple demo of usung neural nets for feature extraction. 
To run the demo just run nlpNet_demo.py