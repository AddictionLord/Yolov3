import torch
from torch.optim import Adam, Adagrad, Adadelta, RMSprop, SGD

from yolo_trainer import YoloTrainer
from yolo import Yolov3
import config
from utils import getLoaders, getValLoader


if torch.cuda.is_available():
    torch.cuda.empty_cache()



# ------------------------------------------------------
def run(model: Yolov3, optimizer: torch.optim):

    t = YoloTrainer()
    # container = {'architecture': config.yolo_config}
    container = YoloTrainer.loadModel('optim_Adam')
    # t.trainNet(model, optimizer, load=container)

    try:
        # t.trainNet(model, optimizer=algorithm)
        t.trainNet(model, optimizer, load=container)

    except KeyboardInterrupt as e:
        print('[YOLO TRAINER]: KeyboardInterrupt', e)

    except Exception as e:
        print(e)

    finally:
        savestr = f'optim_{type(optimizer).__name__}'
        YoloTrainer.saveModel(savestr, t.model, t.optimizer, t.supervisor)


# ------------------------------------------------------
def createModel():

    return  Yolov3(config.yolo_config)


# # ------------------------------------------------------
# def createSamples():

#     samples = dict()

#     adam_model = Yolov3(config.yolo_config)
#     adam = Adam(
#         [
#             {'params': adam_model.yolo.parameters()},
#             {'params': adam_model.darknet.parameters()}
#         ], 
#         lr=config.LEARNING_RATE
#     )

#     adagrad_model = Yolov3(config.yolo_config)
#     adagrad = Adagrad(adagrad_model.parameters(), lr=config.LEARNING_RATE)

#     adadelta_model = Yolov3(config.yolo_config)
#     adadelta = Adadelta(adadelta_model.parameters(), lr=config.LEARNING_RATE)

#     rmsprop_model = Yolov3(config.yolo_config)
#     rmsprop = RMSprop(rmsprop_model.parameters(), lr=config.LEARNING_RATE)

#     sgd_model = Yolov3(config.yolo_config)
#     sgd = SGD(sgd_model.parameters(), lr=config.LEARNING_RATE)

#     samples = {
#         'adam': [adam_model, adam],
#         'adagrad': [adagrad_model, adagrad],
#         'adadelta': [adadelta_model, adadelta],
#         'rmsprop': [rmsprop_model, rmsprop],
#         'sgd': [sgd_model, sgd]
#     }

#     return samples


# ------------------------------------------------------
if __name__ == "__main__":

    adam_model = Yolov3(config.yolo_config)
    adam = Adam(
        [
            {'params': adam_model.yolo.parameters()},
            {'params': adam_model.darknet.parameters()}
        ], 
        lr=config.LEARNING_RATE
    )

    run(adam_model, adam)




    # rmsprop_model = Yolov3(config.yolo_config)
    # rmsprop = RMSprop(rmsprop_model.yolo.parameters(), lr=config.LEARNING_RATE)
    # rmsprop = Adam(rmsprop_model.yolo.parameters(), lr=config.LEARNING_RATE)


    # t = YoloTrainer((getValLoader([25, 30, 35, 40, 45]), getValLoader([25, 30, 35, 40, 45])))
    # t = YoloTrainer()

    # container = {'architecture': config.yolo_config}
    # container = YoloTrainer.loadModel('ultralytics_focal_loss')

    # # t.trainNet(rmsprop_model, rmsprop)
    # # t.trainNet(rmsprop_model, rmsprop, load=container)

    # try:
    #     t.trainNet(rmsprop_model, rmsprop)
    #     # t.trainNet(rmsprop_model, rmsprop, load=container)

    # except KeyboardInterrupt as e:
    #     print('[YOLO TRAINER]: KeyboardInterrupt', e)

    # except Exception as e:
    #     print(e)

    # finally:
    #     savestr = f'optim_{type(t.optimizer).__name__}'
    #     YoloTrainer.saveModel(savestr, t.model, t.optimizer, t.supervisor)




