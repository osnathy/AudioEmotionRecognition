import matplotlib.pyplot as plt


def show_loss(history):
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()


def show_accuracy(history):
    plt.title('Keras model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()
