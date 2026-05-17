import numpy as np

class Tanh:
    """[Given] The Tanh activation function, similar to torch.nn.Tanh."""
    def forward(self, x):
        return np.tanh(x)

    def backward(self, state):
        return 1 - (state**2)


class RNNCell:
    """RNNCell, similar to torch.nn.RNNCell.
    
    Args:
        input_size (int): Integer representing # input features expected by layer
        hidden_size (int): Integer representing # features in the outputted hidden state
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize activation function
        self.activation = Tanh()

        # Randomly initialize weights and biases ("Kaiming Uniform" init)
        bound = np.sqrt(1 / hidden_size)
        self.weight_ih = np.random.uniform(low=-bound, high=bound, size=(hidden_size, input_size))
        self.weight_hh = np.random.uniform(low=-bound, high=bound, size=(hidden_size, hidden_size))
        self.bias_ih = np.random.uniform(low=-bound, high=bound, size=(hidden_size,))
        self.bias_hh = np.random.uniform(low=-bound, high=bound, size=(hidden_size,))

        # Gradients
        self.grad_weight_ih = np.zeros((hidden_size, input_size))
        self.grad_weight_hh = np.zeros((hidden_size, hidden_size))

        self.grad_bias_ih = np.zeros(hidden_size)
        self.grad_bias_hh = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """RNNCell forward (single timestep)

        Args:
            x_t (np.array): (batch_size, input_size) current timestep's input
            h_prev (np.array): (batch_size, hidden_size) the previous timestep's outputted hidden state

        Returns:
            np.array: (batch_size, hidden_size)
        """
        # Calculate y_t
        y_t = np.dot(x_t, self.weight_ih.T) + self.bias_ih + np.dot(h_prev, self.weight_hh.T) + self.bias_hh
        
        # Calculate h_t
        h_t = self.activation.forward(y_t)

        return h_t

    def backward(self, grad, h_t, h_prev_l, h_prev_t):
        """RNNCell backward (single timestep)

        Args:
            grad (np.array): (batch_size, hidden_size) the gradient w.r.t. the output of the current hidden layer (dL/dh_(t,l))
            h_t (np.array): (batch_size, hidden_size) the output of the forward pass at this timestep
            h_prev_l (np.array): (batch_size, input_size) the hidden state given from the previous layer at this timestep
            h_prev_t (np.array): (batch_size, hidden_size) the hidden state from this same layer, just at the previous timestep

        Returns:
            np.array, np.array: shaped (batch_size, input_size) and (batch_size, hidden_size)
        """
        # Backprop through the activation function
        # Using h_t = The state is the output of RNNCell.forward() at the current timestep.
        grad_activation = grad * self.activation.backward(h_t)

        # Accumulate the gradients for the weights and biases
        self.grad_bias_ih += np.sum(grad_activation, axis=0)
        self.grad_bias_hh += np.sum(grad_activation, axis=0)
        self.grad_weight_ih += np.dot(grad_activation.T, h_prev_l)
        self.grad_weight_hh += np.dot(grad_activation.T, h_prev_t)

        # Calculate gradients for the input and the previous hidden state
        dx = np.dot(grad_activation, self.weight_ih)
        dh = np.dot(grad_activation, self.weight_hh)

        return dx, dh


class RNN:
    """RNN layer, similar to torch.nn.RNN. 
    
    Assume our version has `batch_first=True` and `bidirectional=False`.
    
    Args:
        input_size (int): Number of input features expected by layer
        hidden_size (int): Number of features in the outputted hidden state
        num_layers (int): Number of RNNCells to stack, s.t. each layer feeds its output into the next layer.
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize first RNNCell, then add on more if num_layers > 1
        self.layers = [RNNCell(input_size, hidden_size)]
        for l in range(num_layers-1):
            self.layers.append(RNNCell(hidden_size, hidden_size))

    def forward(self, x, h_0=None):
        """Forward propagation over multiple RNNCells, multiple timesteps.

        Args:
            x (np.array): (batch_size, seq_len, input_size)
                          Input. A batch of sequences that each have `input_size` features at each timestep
            h_0 (np.array): (num_layers, batch_size, hidden_size)
                            Initial hidden state (useful if you have prior context to give to this layer). If not given, creates a zero array.

        Returns:
            output (np.array): (batch_size, seq_len, hidden_size)
                               Output of the last cell for each timestep
            h_n (np.array): (num_layers, batch_size, hidden_size)
                            The hidden state at the last timestep for each layer in the batch
        """
        batch_size, seq_len, _ = x.shape
        
        # [Given] Intialize tensor to store every hidden state at each time step and cell of the rnn 
        hiddens = np.zeros((seq_len+1, self.num_layers, batch_size, self.hidden_size))
        if h_0 is not None: # If given, store the initial hidden state 
            hiddens[0,:,:,:] = h_0

        # Process input, timestep by timestep, layer by layer.
        for t in range(seq_len):
            for l in range(self.num_layers):
                # Get input to this layer at this timestep
                # if l == 0, else the output of the previous layer at this timestep
                if l == 0:
                    input_l = x[:, t, :]  # (batch_size, input_size) — from raw input
                else:
                    input_l = hiddens[
                        t + 1, l - 1, :, :]  # (batch_size, hidden_size) — from previous layer at same timestep

                # Get the previous hidden state (same layer, previous timestep)
                h_prev_t = hiddens[t, l, :, :]  # (batch_size, hidden_size)

                # Forward through this layer's RNNCell
                h_out = self.layers[l].forward(input_l, h_prev_t)  # (batch_size, hidden_size)

                # Store the output hidden state for this layer and timestep
                hiddens[t + 1, l, :, :] = h_out

        # [Given] Save the original input and hidden vectors we used, for backprop later.
        self.x = x
        self.hiddens = hiddens
        
        # [Given] Return the output and final hidden states (we transpose to make the batch_size dim first)
        return hiddens[1:,-1,:,:].transpose(1,0,2), hiddens[-1,:,:,:]

    def backward(self, grad):
        """Back Propagation Through Time (BPTT) through multiple RNNCells.

        Args:
            grad (np.array): (batch_size, hidden_size)
                             Gradient of the loss w.r.t. output of last RNN cell

        Returns:
            dx (np.array): (batch_size, seq_len, input_size)
                           Gradient of loss w.r.t. input
            dh_0 (np.array) : (num_layers, batch_size, hidden_size)
                              Gradient of loss w.r.t. the initial hidden state
        """
        batch_size, seq_len, input_size = self.x.shape

        # [Given] Initialize gradients of the input and initial hidden state
        dx = np.zeros((batch_size, seq_len, input_size))
        dh_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        dh_0[-1,:,:] = grad

        # TODO: For each timestep in reverse order, iterate backwards through the layers and run backprop for each one.
        
        return dx, dh_0