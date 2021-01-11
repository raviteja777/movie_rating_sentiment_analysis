from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np


class Modeller:

    def __init__(self, data):
        self.data = data
        self.model = None

    def define_model(self, output_dim=10):
        dim = self.data['X'][0].shape[1]
        print(dim)
        model = Sequential()
        model.add(Dense(dim / 2, input_shape=(dim,)))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(dim / 5))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        self.model = model

    # def construct_network
    def train_model(self, num_classes=10, epochs=10):
        self.model.fit(self.convert_to_tensor(self.data['X']),
                       to_categorical(self.data['y'], num_classes, dtype='float32'),
                       batch_size=32,
                       epochs=epochs,
                       verbose=1)

    def get_model(self):
        if self.model is not None:
            return self.model
        else:
            raise Exception("Model is not created ")

    # converts tfidf scipy sparse matrix to sparse tensor
    # sparse matrix is not directly handled by keras
    def convert_to_tensor(self, dataset):
        print(type(dataset))
        coo = dataset.tocoo()
        indices = np.mat([coo.row,coo.col]).transpose()
        sparse_tensor = tf.SparseTensor(indices,coo.data,coo.shape)
        return tf.sparse.reorder(sparse_tensor)

    # output predtions for test data
    def predict_output(self,data):
        model = self.get_model()
        input_data = self.convert_to_tensor(data['X'])
        actual = [int(x) for x in data['y']]
        predicted = [int(x) for x in np.argmax(model.predict(input_data),axis=1)]
        return pd.DataFrame({"actual":actual, "predicted":predicted})
