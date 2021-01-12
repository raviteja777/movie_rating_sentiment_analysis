# movie_rating_sentiment_analysis

Given reviews from IMDB 
- Each review is given in a text file - along with the rating (
- Perform a sentiment analysis using the train data 
<p>
In the labeled (positive and negative) dataset, a negative review has a score <= 4 out of 10, and
a positive review has a score >= 7 out of 10.
Reviews are stored in text files named following the convention [[id]_[rating].txt] where [id] is a
unique id and [rating] is the star rating for that review on a 1-10 scale. For example, the file
[train/pos/205_8.txt] is the text for a positive-labeled test set example with unique id 205 and
star rating 8/10 from IMDB.
</p>

### Dependencies and versions
PYTHON 3.8

|Package   |  version  |
|-----------|-----------|
|spacy    |2.3.5 |
|sikit-learn 	 |0.23.2 |
|numpy    |1.19.2 |
|pandas   |1.2.0 |
|keras    |2.4.3 |
|keras-base   |2.4.3 |
|keras-preprocessing   |1.1.0 |
|tensorflow    |2.3.0  |
|tensorflow-base   |2.3.0  |

### Instructions for running
- Run the main.py file, this is the starting point of the file 
- Automatically the train and test phases will run 
- Please note the data folder only contains some sample files ,to run it on whole of the data please paste all the data in the data folders
- The trained keras model will be saved in saved_models folder with name model_(timestamp) 
- The metrics as well as any other execution information will be printed on console , also they will be logged in a log file in "files" folder as timestamp.log file (check for latest log)
