import numpy as np
import itertools

# global counter
counter_iterations = itertools.count(start=0, step=1)

# change
NUMBER_CLASSES = 2

# change
WIDTH = 50
HEIGHT = 50
CHANNELS = 3

STANDARDIZE_AXIS_CHANNELS = (0,1,2)

MULTIPROCESSING = True

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"
RES_NET = "RESNET"
DENSE_NET = "DENSENET"

#EXCEPTIONS MESSAGES
ERROR_MODEL_EXECUTION = "\nError on model execution"
ERROR_NO_ARGS = "\nPlease provide args: ",X_VAL_ARGS," and ", Y_VAL_ARGS
ERROR_NO_ARGS_ACCEPTED = "\nThis Strategy doesn't accept more arguments"
ERROR_NO_MODEL = "\nPlease pass a initialized model"
ERROR_INVALID_OPTIMIZER = "\nPlease define a valid optimizer: ", PSO_OPTIMIZER," or ", GA_OPTIMIZER
ERROR_INCOHERENT_STRATEGY = "\nYou cannot choose the oversampling and undersampling strategies at the same time"
ERROR_ON_UNDERSAMPLING = "\nError on undersampling definition"
ERROR_ON_OVERSAMPLING = "\nError on oversampling definition"
ERROR_ON_DATA_AUG = "\nError on data augmentation definition"
ERROR_ON_TRAINING = "\nError on training"
ERROR_ON_OPTIMIZATION = "\nError on optimization"
ERROR_INVALID_NUMBER_ARGS = "\nPlease provide correct number of args"
ERROR_ON_BUILD = "\nError on building model"
ERROR_APPEND_STRATEGY = "\nError on appending strategy"
ERROR_ON_PLOTTING = "\nError on plotting"
ERROR_ON_GET_DATA = "\nError on retain X and Y data"
ERROR_ON_IDENTITY_BLOCK ="\nError on modelling identity block, please check the problem"
ERROR_ON_CONV_BLOCK ="\nError on modelling convolutional block, please check the problem"
ERROR_ON_SUBSAMPLING = "\n Error on subsampling, percentage invalid"
WARNING_SUBSAMPLING = "\nIf you want to subsampling data, please pass a value >0 and <1"

#PSO OPTIONS
PARTICLES = 1
ITERATIONS = 2
PSO_DIMENSIONS = 5
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4}
lbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4, 'k' : 2, 'p' : 2} # p =2, means euclidean distance

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2

# activation functions
RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"
SIGMOID_FUNCTION = "sigmoid"

# padding types
VALID_PADDING = "valid"
SAME_PADDING = "same"

# regularization and train optimizer parameters
LEARNING_RATE = 0.001
DECAY = 1e-6

# train function's of loss
LOSS_BINARY = "binary_crossentropy"
LOSS_CATEGORICAL = "categorical_crossentropy"

# train metrics
ACCURACY_METRIC = "accuracy"
VALIDATION_ACCURACY = "val_accuracy"
LOSS = "loss"
VALIDATION_LOSS = "val_loss"

# train parameters
BATCH_SIZE_ALEX_NO_AUG = 16
BATCH_SIZE_ALEX_AUG = 16
EPOCHS = 30
SHUFFLE = True
GLOROT_SEED = 0
HE_SEED = 0

# data augmentation options
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ROTATION_RANGE = 10
ZOOM_RANGE = 0.25
BRITNESS_RANGE= 0.3

## complete with class names, in respective category order of classes
DICT_TARGETS = (

)