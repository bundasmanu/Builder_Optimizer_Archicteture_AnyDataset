from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten
import config
from exceptions import CustomError
from .Strategies_Train import Strategy, DataAugmentation
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
from typing import Tuple
import Data
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import config_func
from sklearn.utils import class_weight
import numpy

class AlexNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(AlexNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(AlexNet, self).addStrategy(strategy)

    def add_conv(self, input, numberFilters, dropoutRate, input_shape=None):

        '''
        This function represents the use of individual convolutional layers (initial layers on AlexNet)
        Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :param input_shape: tuple (height, width, channels) with shape of first cnn layer --> default None (not initial layers)
        :return: tensor of updated model
        '''

        try:

            if input_shape != None:
                input = Conv2D(filters=numberFilters, input_shape=input_shape, kernel_size=(5,5), strides=2, kernel_initializer='he_uniform',
                               padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            else:
                input = Conv2D(filters=numberFilters, kernel_size=(3, 3), strides=1, kernel_initializer='he_uniform',
                               padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY))(input)

            input = Activation(config.RELU_FUNCTION) (input)
            input = BatchNormalization() (input)
            input = MaxPooling2D(pool_size=(3,3), strides=2, padding='same') (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

    def add_stack(self, input, numberFilters, dropoutRate):

        '''
        This function represents the implementation of stack cnn layers (in this case using only 2 cnn layers compacted)
        Conv --> Activation --> Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :return: tensor of updated model
        '''

        try:

            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_initializer='he_uniform',
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_initializer='he_uniform',
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = BatchNormalization() (input)
            input = MaxPooling2D(pool_size=(3,3), strides=2, padding='same') (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

    def build(self, *args, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        :param args: list integers, in logical order --> to populate cnn (filters) and dense (neurons)
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            if len(args) < (self.nDenseLayers+self.nCNNLayers):
                raise CustomError.ErrorCreationModel(config.ERROR_INVALID_NUMBER_ARGS)

            model = Sequential()
            ##COMPLETE MODEL
            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model : Sequential, *args) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :return: Sequential model --> trained model
        :return: History.history --> train and validation loss and metrics variation along epochs
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            # OPTIMIZER
            opt = Adam(learning_rate=config.LEARNING_RATE, decay=config.DECAY)

            # COMPILE
            model.compile(optimizer=opt, loss=config.LOSS_BINARY, metrics=[config.ACCURACY_METRIC])

            #GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            if self.StrategyList: # if strategylist is not empty
                for i, j in zip(self.StrategyList, range(len(self.StrategyList))):
                    if isinstance(i, DataAugmentation.DataAugmentation):
                        train_generator = self.StrategyList[j].applyStrategy(self.data)
                    else:
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)

            ## CALLBACKS
            ## OPTIMIZER
            ## COMPILE

            if train_generator is None: #NO DATA AUGMENTATION

                history = model.fit(
                    #NEED TO BE COMPLETED
                )

                return history, model

            #ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                # NEED TO BE COMPLETED
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass