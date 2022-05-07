import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




class Visualizer:

    def __init__(self, filename):

        self.path = f'./models/train_data/{filename}.pkl'
        self.data = pd.read_pickle(self.path)

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
    def overview(self):

        loss = self.viewLoss(plot=False)
        mAP = self.viewAccuracy(plot=False)

        fig, axes = plt.subplots(nrows=2, ncols=1)

        loss.plot(ax=axes[0])
        mAP.plot(ax=axes[1])

        plt.show()




# ------------------------------------------------------
if __name__ == "__main__":

    v = Visualizer('supervisor_checkpoint')
    
    # v.viewLoss()
    # v.viewAccuracy()
    v.overview()