from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Dropout, BatchNormalization, Dense, concatenate, Input, ZeroPadding2D, AveragePooling2D, MaxPooling2D
import config
from exceptions import CustomError
from .Strategies_Train import Strategy
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
from typing import Tuple
import Data
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import config_func
from sklearn.utils import class_weight
import numpy as np
from keras.models import Model as mp
from keras_applications.densenet import DenseNet as dnet
from keras.initializers import he_uniform
from models.Strategies_Train import DataAugmentation, UnderSampling, OverSampling

class DenseNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(DenseNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(DenseNet, self).addStrategy(strategy)

    def H(self, inputs, num_filters):

        '''
        THIS FUNCTION SIMULES THE BEHAVIOR OF EXPRESSION PRESENT IN PAPER:
            - xl=H(xl−1)+xl−1
                * H:  represents a composite function which takes in an image/feature map ( x ) and performs some operations on it.
                * x → Batch Normalization → ReLU → Zero Padding → 3×3 Convolution → Dropout
        :param inputs: previous layer
        :param num_filters: integer: number of filters of CNN layer
        :return: Convolution Layer output
        '''

        conv_out = BatchNormalization(axis=3)(inputs)
        conv_out = Activation(config.RELU_FUNCTION)(conv_out)
        bootleneck_filters = num_filters * 4 ## paper
        conv_out = Conv2D(bootleneck_filters, kernel_size=(1, 1), use_bias=False, padding=config.SAME_PADDING, kernel_initializer=he_uniform(config.HE_SEED))(conv_out)
        #conv_out = Dropout(0.2)(conv_out)

        conv_out = BatchNormalization(axis=3)(conv_out)
        conv_out = Activation(config.RELU_FUNCTION)(conv_out)
        #conv_out = ZeroPadding2D((1, 1))(conv_out)
        conv_out = Conv2D(num_filters, kernel_size=(3, 3), use_bias=False, padding=config.SAME_PADDING, kernel_initializer=he_uniform(config.HE_SEED))(conv_out)
        #conv_out = Dropout(0.2)(conv_out)
        return conv_out

    def transition(self, inputs, compresion_rate):

        '''
        The Transition layers perform the downsampling of the feature maps. The feature maps come from the previous block.
        :param inputs: previous layer
        :return: Convolution Layer output
        '''

        x = BatchNormalization(axis=3)(inputs)
        x = Activation(config.RELU_FUNCTION)(x)
        num_feature_maps = inputs.shape[3]

        x = Conv2D(filters=np.floor( compresion_rate * num_feature_maps ).astype( np.int ),
                                   kernel_size=(1, 1), use_bias=False, padding=config.SAME_PADDING, kernel_initializer=he_uniform(config.HE_SEED),
                                   kernel_regularizer=regularizers.l2(1e-4))(x)
        #x = Dropout(rate=0.2)(x)

        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        return x

    def dense_block(self, inputs, num_layers, initFilter, growth_rate):

        '''
        This function represents the logic of Dense block
        :param inputs: input from previous Transition layer
        :param num_layers: integer : number of Conv layer
        :param initFilter: integer: initial number of filters of Bootleneck layer (H function)
        :param growth_rate: integer: increase of filters on bootleneck layers
        :return: Concatenation of Bootleneck layers
        '''

        try:

            for i in range(num_layers):
                conv_outputs = self.H(inputs, initFilter)
                inputs = concatenate([conv_outputs, inputs])
                #initFilter += growth_rate --> sometimes DenseNet can increase number of filters between composite blocks (*2) --> but i don't use this approach
            return inputs

        except:
            raise

    def build(self, *args, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        Reference: https://arxiv.org/pdf/1608.06993.pdf --> Original Paper
        Reference: https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua --> Original Author of DenseNet Paper
        :param args: list integers, in logical order --> to populate cnn (filters) and dense (neurons)
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            model = None
            ## NEED TO BE COMPLETED

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
            model.compile(optimizer=opt, loss=config.LOSS_CATEGORICAL, metrics=[config.ACCURACY_METRIC])

            #GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            # get data
            X_train = self.data.X_train
            y_train = self.data.y_train

            if self.StrategyList: # if strategylist is not empty
                for i, j in zip(self.StrategyList, range(len(self.StrategyList))):
                    if isinstance(i, DataAugmentation.DataAugmentation):
                        train_generator = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, OverSampling.OverSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, UnderSampling.UnderSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)

            ## CALLBACKS
            ## OPTIMIZER
            ## COMPILE

            if train_generator is None:  # NO DATA AUGMENTATION

                history = model.fit(
                    # NEED TO BE COMPLETED
                )

                return history, model

            # ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                # NEED TO BE COMPLETED
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass