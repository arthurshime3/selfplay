import numpy as np

class LSTM_cell:
    def __init__(self):
        stddev = 0.05
        self.weights_input_gate = np.random.normal(0, stddev, (1, self.hidden_layer))
        self.weights_input_hidden = np.random.normal(0, stddev, (self.hidden_layer, self.hidden_layer))
        self.bias_input = np.zeros(self.hidden_layer)

        self.weights_forget_gate = np.random.normal(0, stddev, (1, self.hidden_layer))
        self.weights_forget_hidden = np.random.normal(0, stddev, (self.hidden_layer, self.hidden_layer))
        self.bias_forget = np.zeros(self.hidden_layer)

        self.weights_output_gate = np.random.normal(0, stddev, (1, self.hidden_layer))
        self.weights_output_hidden = np.random.normal(0, stddev, (self.hidden_layer, self.hidden_layer))
        self.bias_output = np.zeros(self.hidden_layer)

        self.weights_memory_cell = np.random.normal(0, stddev, (1, self.hidden_layer))
        self.weights_memory_cell_hidden = np.random.normal(0, stddev, (self.hidden_layer, self.hidden_layer))
        self.bias_memory_cell = np.zeros(self.hidden_layer)

        self.weights_output = np.random.normal(0, stddev, (self.hidden_layer, 1))
        self.bias_output_layer = np.zeros(1)

    def sigmoid(x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def tanh_deriv(x):
        return 1.0 - np.tanh(x) ** 2

    def backpropagate(self, input, output, target, batch_output):
        d_weights_output = np.matmul(2 * (output - target), batch_output)
        d_bias_output_layer = 2 * (output - target)

        deriv_loss = 2 * (output - target)
        deriv_output = self.weights_output
        deriv_batch_output = np.matmul(output_gate, self.tanh_deriv(batch_output))

        deriv_forget_gate = np.matmul(self.sigmoid(forget_gate, True), input)
        deriv_forget_gate_hidden = np.matmul(self.sigmoid(forget_gate, True), output)

        d_weights_forget_gate = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), prev_state), deriv_forget_gate)
        d_weights_forget_hidden = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), prev_state), deriv_forget_gate_hidden)
        d_bias_forget = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), prev_state), self.sigmoid(forget_gate, True))

        deriv_input_gate = np.matmul(self.sigmoid(input_gate, True), input)
        deriv_input_gate_hidden = np.matmul(self.sigmoid(input_gate, True), output)

        d_weights_input_gate = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), memory_cell), deriv_input_gate)
        d_weights_input_hidden = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), memory_cell), deriv_input_gate_hidden)
        d_bias_input = np.matmul(np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), deriv_batch_output), prev_state), self.sigmoid(input_gate, True))

        deriv_output_gate = np.matmul(self.sigmoid(output_gate, True), input)
        deriv_output_gate_hidden = np.matmul(self.sigmoid(output_gate, True), output)

        d_weights_output_gate = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), np.tanh(output)), deriv_output_gate)
        d_weights_output_hidden = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), np.tanh(output)), deriv_output_gate_hidden)
        d_bias_output = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), np.tanh(output)), self.sigmoid(output_gate, True))

        deriv_memory_cell = np.matmul(self.tanh_deriv(memory_cell), input)
        deriv_memory_cell_hidden = np.matmul(self.tanh_deriv(memory_cell), output)

        d_weights_memory_cell = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), output_gate), deriv_memory_cell)
        d_weights_memory_cell_hidden = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), output_gate), deriv_memory_cell_hidden)
        d_bias_memory_cell = np.matmul(np.matmul(np.matmul(deriv_loss, deriv_output), output_gate), self.tanh_deriv(memory_cell))

        self.weights_input_gate += d_weights_input_gate
        self.weights_input_hidden += d_weights_input_hidden
        self.bias_input += d_bias_input

        self.weights_forget_gate += d_weights_forget_gate
        self.weights_forget_hidden += d_weights_forget_hidden
        self.bias_forget += d_bias_forget

        self.weights_output_gate += d_weights_output_gate
        self.weights_output_hidden += d_weights_output_hidden
        self.bias_output += d_bias_output

        self.weights_memory_cell += d_weights_memory_cell
        self.weights_memory_cell_hidden += d_weights_memory_cell_hidden
        self.bias_memory_cell += d_bias_memory_cell

        self.weights_output += d_weights_output
        self.bias_output_layer += d_bias_output_layer

    def cell(self, input, state, output):
        # i_t
        self.input_gate = self.sigmoid(np.matmul(input, self.weights_input_gate) + np.matmul(output, self.weights_input_hidden) + self.bias_input)
        # f_t
        self.forget_gate = self.sigmoid(np.matmul(input, self.weights_forget_gate) + np.matmul(output, self.weights_forget_hidden) + self.bias_forget)
        # o_t
        self.output_gate = self.sigmoid(np.matmul(input, self.weights_output_gate) + np.matmul(output, self.weights_output_hidden) + self.bias_output)
        # C~_t
        self.memory_cell = np.tanh(np.matmul(input, self.weights_memory_cell) + np.matmul(output, self.weights_memory_cell_hidden) + self.bias_memory_cell)
        # C_t
        self.state = state * self.forget_gate + self.input_gate * self.memory_cell
        self.output = output_gate * np.tanh(self.state)
        return self.state, self.output
