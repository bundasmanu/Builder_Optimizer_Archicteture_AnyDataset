from . import Strategy
from imblearn.under_sampling import RandomUnderSampler
from exceptions import CustomError
import config
import numpy as np
import keras
from sklearn.utils import shuffle
import Data
import copy

#REF: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

class UnderSampling(Strategy.Strategy):

    def __init__(self):
        super(UnderSampling, self).__init__()

    def applyStrategy(self, data : Data.Data, **kwargs):

        '''
        THIS FUNCTION APPLIES UNDERSAMPLING TECHNIQUE ON TRAINING DATA
        :param X_train: numpy array --> training data
        :param y_train: numpy array --> training targets
        :return X_train: numpy array --> under sampler training data
        :return y_train: numpy array --> under sampler training targets
        '''

        try:

            if not bool(kwargs) == False: #CHECK IF DICT IS EMPTY
                raise CustomError.ErrorCreationModel(config.ERROR_NO_ARGS_ACCEPTED)

            underSampler = RandomUnderSampler(random_state=0) #ALLOWS REPRODUCIBILITY

            #I NEED TO RESHAPE TRAINING DATA TO 2D ARRAY (SAMPLES, FEATURES)
            X_train = data.reshape4D_to_2D()

            #APLY DECODE OF TARGET DATA NEEDED TO APPLY RESAMPLE
            decoded_ytrain = data.decodeYData()

            #APPLY RESAMPLE OF DATA
            args = (None, None, None, None, None) # REDUCE MEMORY USAGE
            deepData = Data.Data(data.X_train, *args)
            deepData.X_train, decoded_ytrain = underSampler.fit_resample(X_train, decoded_ytrain)

            #I NEED TO RESHAPE DATA AGAIN FROM 2D TO 4D
            X_train = deepData.reshape2D_to_4D()
            del deepData

            #TRANSFORM Y_DECODED TO CATEGORICAL AGAIN
            decoded_ytrain = keras.utils.to_categorical(decoded_ytrain, config.NUMBER_CLASSES)

            #SHUFFLE DATA
            X_train, decoded_ytrain = shuffle(X_train, decoded_ytrain)

            return X_train, decoded_ytrain

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_UNDERSAMPLING)