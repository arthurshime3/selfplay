# Communication Model - Bidirectional LSTM

import numpy as np
from lstm_cell import LSTM_cell


class CM:
    def __init__(self, data):
        self.window_size = 5
        self.batch_size = 5
        self.hidden_layer = 256
        self.learning_rate = 0.001
        self.epochs = 200
        self.train_data = data[0:int(len(data) * 0.9)]
        self.test_data = data[int(len(data) * 0.9) + 1:len(data)]
        self.X_train = self.processXVals([x[0] for x in self.train_data])
        self.y_train = self.processYVals([x[1] for x in self.train_data])
        self.X_test = self.processXVals([x[0] for x in self.test_data])
        self.y_test = self.processYVals([x[1] for x in self.test_data])
        self.LSTM_cells = []
        for i in range(len(X_train)):
            self.LSTM_cells.append(LSTM_cell())

    def processXVals(self, x_vals):
        processed_x = []
        for val in x_vals:
            processed_x.append(val + [0, 0])
        return processed_x

    def processYVals(self, y_vals):
        processed_y = []
        for val in y_vals:
            if val == "UP":
                processed_y.append([1, 0, 0, 0])
            elif val == "RIGHT":
                processed_y.append([0, 1, 0, 0])
            elif val == "DOWN":
                processed_y.append([0, 0, 1, 0])
            else:
                processed_y.append([0, 0, 0, 1])
        return processed_y

    def feed_forward(self, inputs):
        # Network loop
        outputs = []
        for i in range(self.batch_size):
            batch_state = np.zeros((1, self.hidden_layer))
            batch_output = np.zeros((1, self.hidden_layer))

            for ii in range(self.window_size):
                batch_state, batch_output = self.LSTM_cell(np.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)

            outputs.append(np.matmul(batch_output, self.weights_output) + self.bias_output_layer)
        return outputs

    def get_loss(self, outputs, targets):
        losses = []
        for i in range(len(outputs)):
            losses.append((targets[i] - outputs[i]) ** 2)
        loss = np.mean(losses)
        return loss

    def run(self):
        self.initialize_weights()
        inputs = []
        targets = []
        for i in range(self.batch_size):
            inputs.append(self.X_train[i:i + self.window_size])
            targets.append(self.y_train[i:i + self.window_size])
        inputs = np.array(inputs)
        targets = np.array(targets)

        outputs = self.feed_forward(inputs)
        print(outputs)

        self.backpropagate()

    # def run_actual(self):
    #     for i in range(epochs):
    #         trained_scores = []
    #         ii = 0
    #         epoch_loss = []
    #         while (ii + batch_size) <= len(self.X_train):
    #             X_batch = self.X_train[ii:ii + batch_size]
    #             y_batch = self.y_train[ii:ii + batch_size]
    #
    #
    #             ii += batch_size
