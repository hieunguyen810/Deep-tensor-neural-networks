import matplotlib.pyplot as plt
def show_result_tf2(loss, val_loss, num_epochs):
    epochs_range = range(num_epochs)
    plt.plot(epochs_range, loss, label='Test Loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()

    
