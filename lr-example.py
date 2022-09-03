from simplegrad.engine import Num
from simplegrad.nn import FeedForward, mse_loss
import random
import matplotlib.pyplot as plt


# create dataset
input_size = 1
output_size = 1
X = []
Y = []
for _ in range(100):
    x = random.uniform(-10, 10)
    y = 2 * x + 0.3 + random.uniform(-3, 3)
    X.append([Num(x)])
    Y.append(Num(y))

# create model
hidden_size = []
nn = FeedForward(input_size, hidden_size, output_size)

# train
epochs = 100
learning_rate = 0.1
for epoch in range(epochs):
    '''
    # stochastic gradient descent
    total_loss = 0
    for x, y in zip(X, Y):
        # forward
        y_pred = nn.forward(x)[0]
        loss = mse_loss([y_pred], [y])
        total_loss += loss.value

        # backward
        nn.zero_grad()
        loss.backward()
        nn.update_weights(learning_rate)
    print(f'epoch {epoch}: loss={total_loss/len(X):.8f}')
    '''

    # batch gradient descent
    # forward
    y_preds = []
    for x, y in zip(X, Y):
        y_preds.append(nn.forward(x)[0])
    loss = mse_loss(y_preds, Y)
    print(f'epoch {epoch}: loss={loss.value:.8f}')

    # update weights
    nn.zero_grad()
    loss.backward()
    nn.update_weights(learning_rate / len(X))

# plot
plt.scatter([x[0].value for x in X], [y.value for y in Y], color='red')
xs = [[Num(i / 10)] for i in range(-100, 100)]
ys = [nn.forward(x)[0] for x in xs]
plt.plot([x[0].value for x in xs], [y.value for y in ys], color='blue')
plt.show()

