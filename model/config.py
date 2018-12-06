import numpy as np

class Config(object):
    # Model name
    NAME = None

    # Res net type
    FEATURE_EXTRACTOR = 'res50'


    # Layer to extract from residual net
    # for res-net50, only have index(4, 5, 6, 7) to extract.
    # the corresponding output channel is (256, 512, 1024, 2048)
    EXTRACT_LAYER = [4, 5, 6, 7]

    # output channel corresonding residual network output.
    # ToDo: Come up a new method to infer this rather than make it as configuration
    EXTRACTOR_OUTPUT_CHANNELS= [256, 512, 1024, 2048]
    #

    # FPN output channels
    FPN_OUTPUT_CHANNELS = 512

    BATCH_SIZE=2
    # Currently only random crop supported
    CROP_SIZE=512
    EPOCH = 20
    LR = 10e-4
    # The penality parameters to tune the top layers loss with other layers.
    ALPHA = 0.5
    BETA = 0.5

    # Currently loss function only support bce loss
    LOSS= "BCELoss"

    # Saving Setting:
    LOG_STEP = 1000
