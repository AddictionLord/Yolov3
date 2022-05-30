import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



class Visualizer:

    def __init__(self, filename):

        self.path = f'./models/train_data/{filename}.pkl'
        # self.path = f'./models/optim/train_data/{filename}.pkl'
        self.data = pd.read_pickle(self.path)

        # self.data = self.data.iloc[:700]
        # self.data = self.data.iloc[700:]
        # print(f'[VISUALIZER]: Opening file ./models/train_data/{filename}.pkl')
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
    def viewRecall(self, plot=True):

        mAP = self.data.drop('learning_rate', axis=1)
        mAP = mAP.drop('val_loss', axis=1)
        mAP = mAP.drop('loss', axis=1)
        mAP = mAP.drop('epoch', axis=1)
        mAP = mAP.drop('map', axis=1)
        mAP = mAP.drop('map_50', axis=1)
        mAP = mAP.drop('map_75', axis=1)
        mAP = mAP.drop('map_large', axis=1)
        mAP = mAP.drop('mar_large', axis=1)

        print(mAP)

        if plot:

            mAP.plot()
            plt.grid()
            plt.xlabel('epochs')
            plt.ylabel('accuracy')

            plt.show()

        return mAP


    def view_lr(self, plot=True):

        mAP = self.data.drop('val_loss', axis=1)
        mAP = mAP.drop('loss', axis=1)
        mAP = mAP.drop('epoch', axis=1)
        mAP = mAP.drop('map', axis=1)
        mAP = mAP.drop('map_50', axis=1)
        mAP = mAP.drop('map_75', axis=1)
        mAP = mAP.drop('mar_large', axis=1)
        mAP = mAP.drop('mar_1', axis=1)
        mAP = mAP.drop('mar_10', axis=1)
        mAP = mAP.drop('mar_100', axis=1)
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
        # fig, axes = plt.subplots(nrows=2, ncols=1)

        loss.plot(ax=axes[0], color=['blue', 'orange']).set_ylim(top=500 if loss['loss'].max() > 500 else loss['loss'].max() + 50)
        axes[0].set_ylabel('loss')
        mAP.plot(ax=axes[1], color=['red', 'orange', 'blue'])
        axes[1].set_ylabel('mAP')
        mAR.plot(ax=axes[2], color=['red', 'orange', 'blue'])
        axes[2].set_ylabel('mAR')

        for ax in axes:
            
            ax.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('number of epochs')

        plt.show()



# ------------------------------------------------------
def loss_comparison():

    optims = [ 
        'rms_normal_loss',
        'optim_RMSprop'
    ]

    fig, ax = plt.subplots()

    first = True
    for opt in optims:

        v = Visualizer(opt)
        mAP = v.viewPrecision(False)
        mAP = mAP.drop('map_50', axis=1)
        mAP = mAP.drop('map_75', axis=1)
        mAP = mAP.drop('map_large', axis=1)
        name = f'Yolov3 original loss' if first == True else 'Focal loss' 
        mAP.rename(columns={'map': name}, inplace=True)
        mAP.plot(ax=ax)
        first = False

    ax.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
    plt.xlabel('number of epochs')
    plt.ylabel('Accucracy (mAP)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()



# ------------------------------------------------------
if __name__ == "__main__":

    # # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # # v = Visualizer('transfer_learning_darknet_eval')
    # # v = Visualizer('ultralytics_focal_loss')
    # # v = Visualizer('transfer_learning_darknet_eval_checkpoint')
    # # v = Visualizer('overfit_checkpoint')

    # # v = Visualizer('ultralytics_focal_loss_checkpoint')
    # # v = Visualizer('optim_RMS_continue_checkpoint')
    v = Visualizer('bleee272')
    # print(v.data)
    
    # v.viewLoss()
    # v.viewAccuracy()
    # v.view_lr()
    v.overview()        

    # loss_comparison()



    # optims = [
    #     'optim_Adam_better',
    #     'optim_Adagrad',
    #     'optim_Adadelta',
    #     'optim_SGD',
    #     'optim_RMSprop'
    # ]

    # names = [
    #     'Adam',
    #     'Adagrad',
    #     'Adadelta',
    #     'SGD',
    #     'RMSprop'
    # ]

    
    # fig, axes = plt.subplots(2, 1)

            
    # # ------------------------------------------------------
    # for num, opt in enumerate(optims):

    #     v = Visualizer(opt)
    #     mAP = v.viewPrecision(False)
    #     # mAP = mAP.drop('map', axis=1)
    #     mAP = mAP.drop('map_50', axis=1)
    #     mAP = mAP.drop('map_75', axis=1)
    #     # mAP = mAP.drop('map_large', axis=1)
    #     # mAP.rename(columns={'map': f'{opt}'}, inplace=True)
    #     mAP.rename(columns={'map': f'{names[num]}'}, inplace=True)
    #     mAP.plot(ax=axes[0])

    # axes[0].grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
    # axes[0].set_xlabel('number of epochs')
    # axes[0].set_ylabel('mAP')
    # axes[0].spines['top'].set_visible(False)
    # axes[0].spines['right'].set_visible(False)
    # axes[0].spines['left'].set_visible(False)


    # # ------------------------------------------------------
    # for num, opt in enumerate(optims):

    #     v = Visualizer(opt)
    #     mAP = v.viewPrecision(False)
    #     mAP = mAP.drop('map', axis=1)
    #     # mAP = mAP.drop('map_50', axis=1)
    #     mAP = mAP.drop('map_75', axis=1)
    #     # mAP = mAP.drop('map_large', axis=1)
    #     # mAP.rename(columns={'map': f'{opt}'}, inplace=True)
    #     mAP.rename(columns={'map_50': f'{names[num]}'}, inplace=True)
    #     mAP.plot(ax=axes[1])

    # axes[1].grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
    # axes[1].set_xlabel('number of epochs')
    # axes[1].set_ylabel('mAP50')
    # axes[1].spines['top'].set_visible(False)
    # axes[1].spines['right'].set_visible(False)
    # axes[1].spines['left'].set_visible(False)


    # # ------------------------------------------------------
    # for num, opt in enumerate(optims):

    #     v = Visualizer(opt)
    #     mAP = v.viewPrecision(False)
    #     mAP = mAP.drop('map', axis=1)
    #     mAP = mAP.drop('map_50', axis=1)
    #     # mAP = mAP.drop('map_75', axis=1)
    #     # mAP = mAP.drop('map_large', axis=1)
    #     # mAP.rename(columns={'map': f'{opt}'}, inplace=True)
    #     mAP.rename(columns={'map_75': f'{names[num]}'}, inplace=True)
    #     mAP.plot(ax=axes[2])

    # axes[2].grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
    # plt.xlabel('number of epochs')
    # plt.ylabel('Accucracy (mAP)')
    # axes[2].spines['top'].set_visible(False)
    # axes[2].spines['right'].set_visible(False)
    # axes[2].spines['left'].set_visible(False)
    


    plt.show()

    
