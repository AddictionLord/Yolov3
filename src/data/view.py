import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




class Visualizer:

    def __init__(self, filename):

        self.path = f'./models/train_data/{filename}.pkl'
        self.data = pd.read_pickle(self.path)

        # self.data = self.data.iloc[:700]
        # self.data = self.data.iloc[680:740]
        print(f'[VISUALIZER]: Opening file ./models/train_data/{filename}.pkl')
        # print(data)


    # ------------------------------------------------------
    def viewLoss(self, plot=True):

        loss = self.data.get(['loss', 'val_loss'])
        
        print(loss)

        if plot:
                
            loss.plot()
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('loss')

            plt.show()

        return loss


    # ------------------------------------------------------
    def viewAccuracy(self, plot=True):

        mAP = self.data.drop('learning_rate', axis=1)
        mAP = mAP.drop('val_loss', axis=1)
        mAP = mAP.drop('loss', axis=1)
        mAP = mAP.drop('epoch', axis=1)

        print(mAP)

        if plot:

            mAP.plot()
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('accuracy')

            plt.show()

        return mAP


    
    # ------------------------------------------------------
    def viewPrecision(self, plot=True):

        mAP = self.data.drop('learning_rate', axis=1)
        mAP = mAP.drop('val_loss', axis=1)
        mAP = mAP.drop('loss', axis=1)
        mAP = mAP.drop('epoch', axis=1)
        mAP = mAP.drop('mar_1', axis=1)
        mAP = mAP.drop('mar_10', axis=1)
        mAP = mAP.drop('mar_100', axis=1)
        mAP = mAP.drop('mar_large', axis=1)

        print(mAP)

        if plot:

            mAP.plot()
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('accuracy')

            plt.show()

        return mAP


    # ------------------------------------------------------
    def viewRecall(self, plot=True):

        mAP = self.data.drop('learning_rate', axis=1)
        mAP = mAP.drop('val_loss', axis=1)
        mAP = mAP.drop('loss', axis=1)
        mAP = mAP.drop('epoch', axis=1)
        mAP = mAP.drop('map', axis=1)
        mAP = mAP.drop('map_50', axis=1)
        mAP = mAP.drop('map_75', axis=1)
        mAP = mAP.drop('map_large', axis=1)

        print(mAP)

        if plot:

            mAP.plot()
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('accuracy')

            plt.show()

        return mAP



    # ------------------------------------------------------
    def overview(self):

        loss = self.viewLoss(plot=False)
        mAP = self.viewPrecision(plot=False)
        mAR = self.viewRecall(plot=False)

        fig, axes = plt.subplots(nrows=3, ncols=1)

        loss.plot(ax=axes[0]).set_ylim(top=500 if loss['loss'].max() > 500 else loss['loss'].max() + 50)
        # axes[0].ylim(top=500)
        mAP.plot(ax=axes[1])
        mAR.plot(ax=axes[2])

        for ax in axes:
            
            ax.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)

        plt.show()




# ------------------------------------------------------
if __name__ == "__main__":

    # # # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # # # v = Visualizer('transfer_learning_darknet_eval')
    # v = Visualizer('ultralytics_focal_loss')
    # # # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # # # v = Visualizer('overfit_checkpoint')

    # v = Visualizer('optim_Adam')
    # # # v = Visualizer('optim_RMS_continue_checkpoint')
    v = Visualizer('yolo_model')
    
    
    # # # v.viewLoss()
    # # # v.viewAccuracy()
    v.overview()

    # optims = [
    #     'optim_Adam_better',
    #     'optim_Adagrad',
    #     'optim_Adadelta',
    #     'optim_SGD',
    #     'optim_RMSprop'
    # ]

    # fig, ax = plt.subplots()

    # for opt in optims:

    #     v = Visualizer(opt)
    #     mAP = v.viewPrecision(False)
    #     mAP = mAP.drop('map_50', axis=1)
    #     mAP = mAP.drop('map_75', axis=1)
    #     mAP = mAP.drop('map_large', axis=1)
    #     mAP.rename(columns={'map': f'mAP_{opt}'}, inplace=True)
    #     mAP.plot(ax=ax)

    # plt.show()


