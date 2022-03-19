from keras.datasets import cifar10
from matplotlib import pyplot

def load_dataset():
    """
    This function gets the data from keras datasets 
    and loads cifar10 dataset.
    """
    (trainX,trainY),(testX,testY) = cifar10.load_data()
    return trainX, trainY, testX, testY

def plot_datasets(trainX,trainY,testX,testY):
    """
    Prints out the summary of the train and test dataset.
    Also gives a sumrray plot of the dataset
    """
    print("Train: X={} y={}".format(trainX.shape, trainY.shape))
    print("Test: X={} y={}".format(testX.shape, testY.shape))
    for i in range(9):
        pyplot.subplot(330+1+i)
        pyplot.imshow(trainX[i])
    pyplot.show()

if __name__ == "__main__":
    trainX,trainY,testX,testY = load_dataset()
    plot_datasets(trainX,trainY,testX,testY)
