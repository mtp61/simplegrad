from simplegrad.engine import Num
from simplegrad.nn import FeedForward, crossentropy_loss
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np


# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

# get dataset
digits = datasets.load_digits()

# visualize
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.show()
'''

# pre-process dataset
X = digits.images.reshape(len(digits.images), -1)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

scaler = preprocessing.StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

X_train_nums = [[Num(n) for n in example] for example in X_train_std]
X_test_nums = [[Num(n) for n in example] for example in X_test_std]

# create model
input_size = 64
output_size = 10

hidden_size = [10]
model = FeedForward(input_size, hidden_size, output_size)

# train (stochastic gradient descent)
epochs = 1
learning_rate = 0.05

def main():
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y_) in enumerate(zip(X_train_nums, y_train)):
            # forward
            y_pred = model.forward(x)
            loss = crossentropy_loss([y_pred], [y_])
            total_loss += loss.value

            # backward
            model.zero_grad()
            loss.backward()
            model.update_weights(learning_rate)

            if i % 10 == 0:
                print(f'epoch {epoch}, {i+1} / {len(X_train_nums)}: epoch mean loss {total_loss/(i+1):.4f}')
                if i % 100 == 0:
                    acc_train, loss_train = get_accuracy_loss(X_train_nums, y_train)
                    acc_test, loss_test = get_accuracy_loss(X_test_nums, y_test)
                    print(f'\ttrain accuracy={acc_train:.4f}, test accuracy={acc_test:.4f}, train loss={loss_train:.4f}, test loss={loss_test:.4f}')


def get_accuracy_loss(X, y):
    preds = []
    loss = 0
    for x, y_ in zip(X, y):
        out = model.forward(x)
        preds.append(np.argmax(np.array([n.value for n in out])))
        loss += crossentropy_loss([out], [y_]).value

    preds = np.array(preds)
    return (y == preds).sum() / len(y), loss / len(y)


if __name__ == '__main__':
    main()

