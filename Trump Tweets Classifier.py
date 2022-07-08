import csv
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.decomposition import TruncatedSVD

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from datetime import datetime
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
import string
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


fnn_hidden_layer_size = 200
fnn_l_rate = 0.001
fnn_num_epochs = 40
fnn_batch_size = 256

stop_words = set(stopwords.words('english'))
NUM_CLUSTERS = 4
NUM_FEATURES_FOR_FNN = 9011
NUM_TRAIN_SPLIT = 2
SVD_REDUCTION = 400

best_model_index = 3
score_index = ['randomForest', 'SVC_sigmoid', 'SVC_rbf', 'SVC_linear', 'LogisticRegression', 'FNN']

word_embedding_file = "train_embedding_vectors_google.pkl"
train_tweet_file = "trump_train.tsv"
test_tweet_file = "trump_test.tsv"

tweet_time_key = 'tweet_time'
tweet_preprocessed_text_key = 'text_processed'
tweet_id_key = 'tweet_id'
tweet_user_handle_key = 'user_handle'
tweet_text_key = 'text'
tweet_device_key = 'device'
tweet_web_links_count_key = 'web_links_count'
tweet_target_key = 'target'


"""
   Transformers
"""


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TimeStatistics:
    def __init__(self, key):
        self.features_name = ['weekday', 'relative_time_to_election', 'hour']
        self.key = key

    def get_feature_names(self, ):
        return self.features_name

    @staticmethod
    def get_relative_time_to_election(time_list):
        election_date = datetime.strptime("8-11-2016", '%d-%m-%Y')
        relative_to_election_date = list(map(lambda x: 1 if x > election_date else 0, time_list))
        return np.array(relative_to_election_date).reshape(-1, 1)

    @staticmethod
    def get_weekday(time_list):
        week_days_for_time_list = [time.weekday() for time in time_list]
        week_days_for_time_list = [week_day/7 for week_day in week_days_for_time_list]
        return np.array(week_days_for_time_list).reshape(-1, 1)

    @staticmethod
    def get_year(time_list):
        year_for_time_list = [time.year for time in time_list]
        years = list(set(year_for_time_list))
        year_for_time_list = [years.index(year)/len(years) for year in year_for_time_list]
        return np.array(year_for_time_list).reshape(-1, 1)

    @staticmethod
    def get_hour(time_list):
        hour_for_time_list = [time.hour for time in time_list]
        hours = list(set(hour_for_time_list))
        hour_for_time_list = [hours.index(hour) / len(hours) for hour in hour_for_time_list]
        return np.array(hour_for_time_list).reshape(-1, 1)

    def fit(self, df_threads, y=None):
        return self

    def transform(self, df, y=None):
        # , self.get_year
        transforms_to_apply = [self.get_weekday, self.get_relative_time_to_election, self.get_hour]
        # print("features from time stats")
        # print(np.concatenate(([transform(df[self.key]) for transform in transforms_to_apply]), axis=1).shape)
        return np.concatenate(([transform(df[self.key]) for transform in transforms_to_apply]), axis=1)


class TextStatistics:
    def __init__(self, key):
        self.features_name = ['words_count', 'alpha_bet_dist', 'digits_dist', 'nouns_count']
        self.features_name += ["sentiment_" + i for i in ["neg", "neu", "pos", "compound"]]
        self.sid = SentimentIntensityAnalyzer()
        self.key = key

    @staticmethod
    def normalize(text):
        text = remove_web_links(text)
        # .translate(string.punctuation)
        text = text.lower()
        return text

    def get_sentiment_analysis(self, texts):
        sentiment_scores = [self.sid.polarity_scores(text) for text in texts]
        sentiment_scores_array = np.array([list(sentiment_score.values()) for sentiment_score in sentiment_scores])
        return sentiment_scores_array

    @staticmethod
    def get_words_count(texts):
        return np.array([len(text.split()) for text in texts]).reshape(-1, 1)

    @staticmethod
    def get_alpha_count(texts):
        return np.array([sum(c.isalpha() for c in text)/len(text) if len(text) > 0 else 0 for text in texts]).reshape(-1, 1)

    @staticmethod
    def get_digits_count(texts):
        return np.array([sum(c.isdigit() for c in text)/len(text) if len(text) > 0 else 0 for text in texts]).reshape(-1, 1)

    def get_feature_names(self,):
        return self.features_name

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # transforms_to_apply = [self.get_words_count, self.get_alpha_count, self.get_digits_count, self.get_sentiment_analysis]
        transforms_to_apply = [self.get_sentiment_analysis]
        texts = df[self.key].apply(self.normalize)
        # print("features from text stats")
        # print(np.concatenate(([transform(texts) for transform in transforms_to_apply]), axis=1).shape)
        return np.concatenate(([transform(texts) for transform in transforms_to_apply]), axis=1)


class WordEmbedding:

    def __init__(self, key):
        self.word_embedding = pickle.load(open(word_embedding_file, 'rb'))
        self.word_embedding_vector_size = 300
        self.features_name = ["word2vec_" + str(i) for i in range(0, self.word_embedding_vector_size)]
        self.key = key

    def get_feature_names(self, ):
        return self.features_name

    def get_word_embedding_for_text(self, text):
        if type(text) is float:
            text = ""
        text_tokens = text.split()
        tokens_vector_sum = np.array([float(0)] * self.word_embedding_vector_size)
        for token in text_tokens:
            if token in self.word_embedding:
                tokens_vector_sum = tokens_vector_sum + np.array(self.word_embedding[token])
        tokens_count_in_tweet = len(text_tokens)
        if tokens_count_in_tweet > 0:
            tokens_vector_average = tokens_vector_sum/tokens_count_in_tweet
        else:
            tokens_vector_average = tokens_vector_sum
        return tokens_vector_average

    def get_word_embedding_for_list_of_texts(self, list_of_texts):
        out = [self.get_word_embedding_for_text(text) for text in list_of_texts]
        # print("features from word_embedding")
        # print(np.array(out).shape)
        return np.array(out)

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return self.get_word_embedding_for_list_of_texts(df[self.key])


"""
   FNN network
"""


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, criterion, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 2, bias=True)
        self.output_activation = F.log_softmax
        self.criterion = criterion

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.output_activation(x, -1)
        return x


class FNNClassifier:

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def fit(self, x_train, y_train):
        y_column = np.array(y_train, dtype=int).reshape(-1, 1)
        x_train = x_train.toarray()
        data = np.concatenate([x_train, y_column],  axis=1)
        loader = DataLoader(data, batch_size=fnn_batch_size, shuffle=False)
        for epoch in range(fnn_num_epochs):
            for idx, batch_data in enumerate(loader):
                self.optimizer.zero_grad()
                batch_features = batch_data[:, :-1].float()
                batch_y = batch_data[:, -1].long()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, x_test):
        x_test = x_test.toarray()
        data = torch.from_numpy(x_test).float()
        self.optimizer.zero_grad()
        outputs = self.model(data)
        return outputs


def create_fnn_model(features_count):
    criterion = nn.CrossEntropyLoss()
    net = Net(input_size=features_count, hidden_size=fnn_hidden_layer_size, criterion=criterion, num_classes=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=fnn_l_rate)
    return FNNClassifier(model=net, optimizer=optimizer, criterion=criterion)


classifiers = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(n_jobs=2, random_state=0),
        SVC(kernel='sigmoid'),
        SVC(kernel='rbf'),
        SVC(kernel='linear'),
        create_fnn_model(NUM_FEATURES_FOR_FNN)
        ]


def create_trump_label(row):
    if row[tweet_device_key] == 'android' and row[tweet_user_handle_key] == 'realDonaldTrump' and not row[tweet_text_key].startswith('RT'):
        return 0
    else:
        return 1


def get_web_links_count(text):
    return len(text.split('http')) - 1


def load_tweets_file(file_name, to_test):
    if to_test:
        tweets = pd.read_csv(file_name, sep="\t", header=None, quoting=csv.QUOTE_NONE, names=[tweet_user_handle_key, tweet_text_key, tweet_time_key])
    else:
        tweets = pd.read_csv(file_name, sep="\t", header=None, quoting=csv.QUOTE_NONE, names=[tweet_id_key, tweet_user_handle_key, tweet_text_key, tweet_time_key, tweet_device_key])
        tweets = tweets.dropna(subset=[tweet_time_key])
        tweets[tweet_web_links_count_key] = tweets[tweet_text_key].apply(get_web_links_count)
        tweets[tweet_target_key] = tweets.apply(create_trump_label, axis=1)

    tweets[tweet_time_key] = tweets[tweet_time_key].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return tweets


def print_evaluation_results(y_test, predictions):
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
    print("auc: ")
    auc = metrics.auc(fpr, tpr)
    print(auc)
    return auc


def remove_web_links(token):
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    return token


def normalize_text(text):
    """
    This function takes as input a text on which several
    NLTK algorithms will be applied in order to preprocess it
    """
    # Remove web links
    pattern = r'''(?x)          # set flag to allow verbose regexps
           (?:[A-Z]\.)+          # abbreviations, e.g. U.S.A.
           | \w+(?:-\w+)*        # words with optional internal hyphens
           | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
           | \.\.\.              # ellipsis
           | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
           '''
    text = text.lower().translate(string.punctuation)
    regexp = re.compile(pattern)
    tokens = regexp.findall(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def normalize_tweet(tweet_df):
    tweets_list = tweet_df[tweet_text_key].to_list()
    return [tweet.lower().translate(string.punctuation) for tweet in tweets_list]


def get_raw_data():
    tweets_df = load_tweets_file(train_tweet_file, to_test=False)
    tweets_df[tweet_preprocessed_text_key] = tweets_df[tweet_text_key].apply(normalize_text)
    tweets_df = tweets_df.sort_values([tweet_time_key]).reset_index()
    target = tweets_df[tweet_target_key]
    # tweets_df.to_csv("tweets_df_with preprocessed_text.csv", encoding='utf-8', index=False)
    tweets_df = tweets_df.drop([tweet_target_key], axis=1)
    return tweets_df, target


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if item not in stop_words:
            stems.append(item)
    return stems


def get_pipeline_for_clf(clf_index, clf):
    if score_index[clf_index] != 'SVC_rbf' and score_index[clf_index] != 'SVC_linear':
        pipeline = Pipeline([('features', FeatureUnion(transformer_list=[
            ('tfidf', Pipeline([
                ('selector', ItemSelector(key=tweet_preprocessed_text_key)),
                ('TfidfVectorizer', TfidfVectorizer(lowercase=False, min_df=1, tokenizer=tokenize, max_features=9000, ngram_range=(1, 3)))
            ])),
            ('time', TimeStatistics(tweet_time_key)),
            ('text', TextStatistics(tweet_text_key)),
            ('kmeans_clustering', Pipeline([
                ('word_embedding', WordEmbedding(tweet_preprocessed_text_key)),
                ('kmeans', KMeans(n_clusters=NUM_CLUSTERS))
            ]))
        ])),
            ('clf', clf)
        ])
    else:
        pipeline = Pipeline([('features', FeatureUnion(transformer_list=[
            ('tfidf', Pipeline([
                ('selector', ItemSelector(key=tweet_preprocessed_text_key)),
                ('TfidfVectorizer',
                 TfidfVectorizer(lowercase=False, min_df=1, tokenizer=tokenize, max_features=9000, ngram_range=(1, 3)))
            ])),
            ('time', TimeStatistics(tweet_time_key)),
            ('text', TextStatistics(tweet_text_key)),
            ('kmeans_clustering', Pipeline([
                ('word_embedding', WordEmbedding(tweet_preprocessed_text_key)),
                ('kmeans', KMeans(n_clusters=NUM_CLUSTERS))
            ]))
        ])),
             ('svd', TruncatedSVD(algorithm='randomized', n_components=300)),
             ('clf', clf)
             ])

    return pipeline


def apply_trian_and_test(X_train, y_train, X_test, y_test, score):
    for clf_index, clf in enumerate(classifiers):
        print(clf)
        pipeline = get_pipeline_for_clf(clf_index, clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if score_index[clf_index] == 'FNN':
            y_pred = list(map(lambda x: 0 if x[0] > x[1] else 1, y_pred))
        score[score_index[clf_index]].append(print_evaluation_results(y_test, y_pred))
    return score


def train_and_test_models():
    score = {'randomForest': [], 'SVC_sigmoid': [], 'SVC_rbf': [], 'SVC_linear': [], 'LogisticRegression': [], 'FNN':[]}
    tweets_df, target = get_raw_data()
    tscv = TimeSeriesSplit(n_splits=NUM_TRAIN_SPLIT)
    split_counter = 1
    for train_index, test_index in tscv.split(tweets_df):
        print("split counter :"+str(split_counter) +"from "+str(NUM_TRAIN_SPLIT))
        X_train, X_test = tweets_df.iloc[train_index, :], tweets_df.iloc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        apply_trian_and_test(X_train, y_train, X_test, y_test, score)
        split_counter += 1
    print(score_index)
    print(score)


def load_best_model():
    """

    :return: returning your best performing model that was saved as part of the submission bundle
    """
    best_model_pipeline = pickle.load(open("best_model_pipeline.pkl", 'rb'))
    return best_model_pipeline


def train_best_model():
    """
        training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
        Of course, the final model could be slightly different than the one returned by  load_best_model(),
        due to randomization issues. This function call training on the data file you recieved. You could assume it
        is in the current directory. It should trigger the preprocessing and the whole pipeline.
    :return:
    """
    tweets_df, target = get_raw_data()
    best_model = SVC(kernel='linear')
    pipeline = get_pipeline_for_clf(best_model_index, best_model)
    pipeline.fit(tweets_df, target)
    return pipeline


def predict(m, fn):
    """
        returns the predictions of model m on test set from fn
    :param m: the trained model
    :param fn: the full path to a file in the same format as the test set
    :return: returns a list of 0s and 1s, corresponding to the lines in the specified file.
    """
    test_df = load_tweets_file(fn, True)
    test_df[tweet_preprocessed_text_key] = test_df[tweet_text_key].apply(normalize_text)
    y_pred = m.predict(test_df)
    return y_pred


def predict_and_write_to_file():
    pipeline = load_best_model()
    y_pred = predict(pipeline, 'trump_test.tsv')
    result_to_file = " ".join([str(i) for i in y_pred])
    with open("test_trump_output_load_model.txt", "w") as text_file:
        text_file.write(result_to_file)

def main():
    # score = train_and_test_models()
    # print(score_index)
    # print(score)
    pipeline = train_best_model()
    y_pred = predict(pipeline, 'trump_test.tsv')
    result_to_file = " ".join([str(i) for i in y_pred])
    with open("test_trump_output_load_model_new.txt", "w") as text_file:
        text_file.write(result_to_file)
    # pickle.dump(pipeline, open("best_model_pipeline.pkl", "wb"))
    # predict_and_write_to_file()




if __name__ == "__main__":
    main()



