import warnings

from matplotlib import pyplot as plt
from skimage.filters import prewitt_v

import mnist_reader
from content import *

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    # load data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # divide for validation set
    X_train = np.reshape(X_train, (60000, 28, 28))
    X_val = X_train[0:10000]
    X_train = X_train[10000::]
    y_val = y_train[0:10000]
    y_train = y_train[10000::]
    X_test = np.reshape(X_test, (10000, 28, 28))

    # prepare for better display
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_error(error_1, error_2, error_3):
    plt.figure()
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    labels = ["No changes", "Round", "Edges"]
    data = [error_1, error_2, error_3]

    x_locations = np.array(range(len(data))) + 0.5
    width = 0.5
    plt.bar(x_locations, data, width=width, color='#FFCC55')
    plt.xticks(x_locations, labels)
    plt.xlim(0, x_locations[-1] + width * 2 - .5)
    plt.title("Models comparison - classification error")
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.draw()
    plt.show()


def classification_for_k(xs, ys):
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'none'
    plt.style.use(['dark_background'])
    plt.xlabel('k value')
    plt.ylabel('classification error')
    plt.title("Model selection")

    plt.plot(xs, ys, 'r-', color='#FFCC55')
    plt.draw()


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)
    plt.show()


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


def run_all():
    error_1 = run_training(load_data(), "unchanged data")
    error_2 = run_for_round(load_data())
    new_data = get_edges(load_data())
    error_3 = run_training(new_data, "edges")
    plot_error(error_1, error_2, error_3)


def run_training(data, data_print):
    k_values = range(1, 101, 2)
    print(f'\n------------- Model selection: k-NN, {data_print} -------------')
    print('-------------------- k values: 1, 3, ..., 100 -----------------------')
    error_best, best_k, errors = model_selection(data[0], data[2], data[3], data[1], k_values)
    print('Best k: {num1} and best classification error: {num2:.4f}'.format(num1=best_k, num2=error_best))
    classification_for_k(k_values, errors)

    predictions, error = probabilities_for_k(data[0], data[4], data[1], data[5], best_k)
    print('Error: {num1:.4f} for k: {num2}'.format(num1=error, num2=best_k))
    show_examples(predictions, data)
    return error


def run_for_round(data):
    k_values = range(1, 101, 2)
    print('\n------------- Model selection: k-NN, rounded data -------------')
    print('-------------------- k values: 1, 3, ..., 100 -----------------------')
    mn = [3, 2, 1, 0]
    err_mn = []
    best_k_t = []
    for mni in mn:
        X_test, X_val = round_values(data[0], data[2], mni)
        error_best, best_k, errors = model_selection(X_test, X_val, data[3], data[1], k_values)
        err_mn.append(error_best)
        best_k_t.append(best_k)
        print('Best k: {num1} and best classification error: {num2:.4f}, Decimal places: {num3}'.format(num1=best_k,
                                                                                                        num2=error_best,
                                                                                                        num3=mni))
        classification_for_k(k_values, errors)

    X_test, X_val = round_values(data[0], data[4], mn[np.argmin(err_mn)])
    predictions, error = probabilities_for_k(X_test, X_val, data[1], data[5], best_k_t[np.argmin(err_mn)])
    print('Error: {num1:.4f} for k: {num2}'.format(num1=error, num2=best_k_t[np.argmin(err_mn)]))
    show_examples(predictions, data)
    return error


def show_examples(predictions, data):
    num_rows = 5
    num_cols = 5
    num_images = num_rows * num_cols
    plt.title("Example classification")
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], data[5], data[4])
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], data[5])
    plt.tight_layout()
    plt.show()


def get_edges(data):
    return list(map(prewitt_v, data[0])), data[1], list(map(prewitt_v, data[2])), data[3], \
           list(map(prewitt_v, data[4])), data[5]


def round_values(x_test, x_val, mn):
    x_val = np.around(x_val, mn)
    x_test = np.around(x_test, mn)
    return x_test, x_val


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_all()
