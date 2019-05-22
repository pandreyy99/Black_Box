import os
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Data_Loader:
    random_distribution = 0
    def __init__(self):
        # self.dataset_path = "ml-unibuc-2019-24/"
        self.dataset_path = os.getcwd() + '/ml-unibuc-2019-24/'
        self.data = []
        self.labels = []
        self.eval = []

    def getStationMetadata(self):
        '''
        Retrieve metadata on preprocessed_data

        @return pandas dataframe with preprocessed_data information
        '''

        data_file = os.path.exists('processed_data.h5')
        # print(data_file)
        if data_file is False:
            print('Dataset not available')
            return (False,None,0)

        store = pd.HDFStore('processed_data.h5', 'r')
        meta_data = store['preprocessed_data']
        random_file = open('random.txt', 'r')
        read = random_file.read()
        random = float(read)
        store.close()
        random_file.close()

        return (True,meta_data,random)

        # functie care primeste path-ul catre un folder si
    # citeste toate fisierele din el, salvand recenziile intr-o lista

    def build_dataset(self):
        # random_distribution = random.random()
        ok, data, random_distribution = self.getStationMetadata()
        if ok == False:
            random_distribution = random.uniform(0.75,1)
            if (random_distribution < 0.5):
                random_distribution = 1 - random_distribution
            data = pd.read_csv(self.dataset_path + 'train_samples.csv', header=None)

            # Create storage object with filename `processed_data`
            data_store = pd.HDFStore('processed_data.h5')

            # Put DataFrame into the object setting the key as 'preprocessed_data'
            data_store['preprocessed_data'] = data
            random_file = open('random.txt', 'w')
            random_file.write(str(random_distribution) + '\n')
            data_store.close()
            random_file.close()
        eval = pd.read_csv(self.dataset_path + 'test_samples.csv', header=None)
        for rows in data:
            self.data = data.to_numpy(dtype='float')
        for rows in eval:
            self.eval = eval.to_numpy(dtype='float')
        labels = pd.read_csv(self.dataset_path + 'train_labels.csv', header=None, dtype='int')
        for rows in labels:
            self.labels = labels.to_numpy(dtype = 'int')
        self.labels = self.labels.ravel()

        # impartim setul de date in random% pentru antrenare si 1-random% pentru test
        num_training_samples_per_class = int(random_distribution * len(data))
        num_test_samples_per_class = len(data) - num_training_samples_per_class

        # in setul de antrenare bagam datele cu indecsii 0:random*15000
        train_data = self.data[0:num_training_samples_per_class]

        # la fel si in etichetele  datelor de intrare
        train_labels = self.labels[0:num_training_samples_per_class]

        # in setul de test salvam restul datelelor din cel de antrenare
        test_data = self.data[num_training_samples_per_class:len(data)]
        test_labels = self.labels[num_training_samples_per_class:len(labels)]

        # self.train_labels, self.train_data = train_labels, train_data
        # self.test_labels, self.test_data = test_labels, test_data

        # amestecam datele
        self.train_data, self.train_labels = shuffle(train_data, train_labels)
        self.test_data, self.test_labels = shuffle(test_data, test_labels)

        self.num_classes = max(np.max(self.train_labels), np.max(self.test_labels))

