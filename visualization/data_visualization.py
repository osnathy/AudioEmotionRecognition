import matplotlib.pyplot as plt


def show_loss(records):
    plt.title('Loss')
    plt.plot(records.history['loss'], label='train')
    plt.plot(records.history['val_loss'], label='validation')
    plt.legend()
    plt.savefig("loss")

    plt.show()


def show_accuracy(records):
    plt.title('Accuracy')
    plt.plot(records.history['accuracy'], label='train')
    plt.plot(records.history['val_accuracy'], label='validation')
    plt.legend()
    plt.savefig("accuracy")
    plt.show()



