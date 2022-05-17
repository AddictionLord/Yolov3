import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




class Visualizer:

    def __init__(self, filename):

        self.path = f'./models/train_data/{filename}.pkl'
        self.data = pd.read_pickle(self.path)

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

        loss.plot(ax=axes[0])
        mAP.plot(ax=axes[1])
        mAR.plot(ax=axes[2])

        plt.show()




# ------------------------------------------------------
if __name__ == "__main__":

    # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # v = Visualizer('transfer_learning_darknet_eval')
    # v = Visualizer('ultralytics_focal_loss')
    v = Visualizer('ultralytics_focal_loss_checkpoint')
    # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # v = Visualizer('overfit_checkpoint')
    
    
    # v.viewLoss()
    # v.viewAccuracy()
    v.overview()
