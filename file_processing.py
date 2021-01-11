import os
import spacy
import re
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
 Process file and extract data
 remove punctuation & stop words and lemmatize
 add processed phrase and rating to dataframe
 create dataset by converting text part to tfidf vectors
"""


class FileProcessor:
    tfidf = TfidfVectorizer(max_features=200)

    def __init__(self, files_list, is_train=True):
        self.files = files_list
        self.corpus = pd.DataFrame(columns=["doc_id","summary", "rating"])
        self.train_mode = is_train
        self.dataset = None

    def process_all_files(self):
        for filename in self.files:
            with open(filename, 'r') as file:
                self.add_to_dataframe(file)
        print(self.corpus.head())
        self.dataset = self.convert_to_vectors()

    # process all lines in a file
    # remove stop words and lemmatize
    # add processed phrase and rating to dataframe
    def add_to_dataframe(self, file):
        nlp = spacy.load("en_core_web_sm")
        text = file.read()
        lines = text.split('\n')
        for line in lines:
            line = nlp(re.sub(r"<\w+.*/>", "", line.lower().strip()))
            tokens = [word.lemma_.strip() for word in line if not word.is_punct and not word.is_stop]
            details = self.get_details(file.name)
            self.corpus = self.corpus.append({"doc_id": details[0], "summary": " ".join(tokens), "rating": details[1]},
                                             ignore_index=True)

    # get rating from filename
    def get_details(self, file_name):
        pat = re.compile(r"(\d+)_(\d+)")
        mat = pat.search(os.path.basename(file_name))
        return int(mat.group(1)), int(mat.group(2)) - 1

    # convert to tfidf
    def convert_to_vectors(self):
        if self.train_mode:
            X = self.tfidf.fit_transform(self.corpus["summary"])
        else:
            X = self.tfidf.transform(self.corpus["summary"])
        y = self.corpus["rating"]
        return {'X': X, 'y': y}

    def get_corpus(self):
        return self.corpus

    def get_dataset(self):
        return self.dataset
