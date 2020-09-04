from . import Model
import Data
from .Strategies_Train import Strategy, DataAugmentation
from exceptions import CustomError
import config
import config_func
import numpy
from keras.models import Model as mp, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Input, BatchNormalization, Dense, Flatten, Add, \
    ZeroPadding2D, AveragePooling2D
from keras.callbacks.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.regularizers import l2
from keras.utils import plot_model
from sklearn.utils import class_weight
from typing import Tuple


class ResNet(Model.Model):

    def __init__(self, data: Data.Data, *args):
        super(ResNet, self).__init__(data, *args)

    def addStrategy(self, strategy: Strategy.Strategy) -> bool:
        super(ResNet, self).addStrategy(strategy=strategy)

    def identity_block(self, tensor_input, *args):

        '''
        THIS FUNCTION SIMULES THE CONCEPT OF A IDENTITY BLOCK
            paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        :param tensor_input: input_tensor result of previous block application on cnn architecture (conv_block or identity_block)
        :param args: number of filters to populate conv2d layers
        :return: tensor merge of input and identity conv blocks
        '''

        try:

            ## save copy input, because i need to apply alteration on tensor_input parameter, and in final i need to merge this two tensors
            input = tensor_input

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3, 3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED),
                                  kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3)(
                tensor_input)  ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3, 3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED),
                                  kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3)(
                tensor_input)  ## perform batch normalization alongside channels axis [samples, width, height, channels]
            #tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            ## now i need to merge initial input and identity block created, this is passed to activation function
            tensor_input = Add()([tensor_input, input])
            tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            return tensor_input

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_IDENTITY_BLOCK)

    def convolution_block(self, tensor_input, *args):

        '''
        THIS FUNCTIONS REPRESENTS THE CONCEPT OF CONVOLUTION BLOCK ON RESNET, COMBINING MAIN PATH AND SHORTCUT
            paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        :param tensor_input: input_tensor result of previous block application on cnn architecture (conv_block or identity_block)
        :param args: number of filters to populate conv2d layers
        :return: tensor merge of path created using convs and final shortcut
        '''

        try:

            ## save copy input, because i need to apply alteration on tensor_input parameter, and in final i need to merge this two tensors
            shortcut_path = tensor_input

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3, 3), strides=args[1],
                                  # in paper 1 conv layer in 1 conv_block have stride=1, i continue with stride=2, in order to reduce computacional costs)
                                  kernel_initializer=he_uniform(config.HE_SEED),
                                  kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3)(
                tensor_input)  ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3, 3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED),
                                  kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3)(tensor_input)  ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            ## definition of shortcut path
            shortcut_path = Conv2D(filters=args[0], kernel_size=(1, 1), strides=args[1], padding=config.SAME_PADDING,
                                   kernel_initializer=he_uniform(config.HE_SEED),
                                   kernel_regularizer=l2(config.DECAY))(shortcut_path)
            shortcut_path = BatchNormalization(axis=3)(shortcut_path)

            ## now i need to merge conv path and shortcut path, this is passed to activation function
            tensor_input = Add()([tensor_input, shortcut_path])
            tensor_input = Activation(config.RELU_FUNCTION)(tensor_input)

            return tensor_input

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_CONV_BLOCK)

    def build(self, *args, trainedModel=None) -> Sequential:

        ## model based on resnet-18 approach and described in paper cited in identity_block and convolution_block functions

        try:

            # IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            model = None
            ## MODEL NEEED TO BE COMPLETED

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model: Sequential, *args) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :param args: only one value batch size
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

            # GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
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
        super(ResNet, self).__str__()