import numpy as np

class NeuralNetwork:
    """
    Lightweight Feed-Forward Neural Network brain for the agents.
    Uses basic matrix math with NumPy.
    Dimensions: 5 inputs -> 8 hidden neurons -> 2 outputs.
    All layers use tanh activations to squash values to [-1.0, 1.0].
    """
    def __init__(self, input_size=5, hidden_size=8, output_size=2):
        """
        Initializes neural weights and biases randomly in range [-1.0, 1.0].
        
        Inputs:
            input_size: Number of input nodes (integer, default 5)
            hidden_size: Number of hidden layer nodes (integer, default 8)
            output_size: Number of output nodes (integer, default 2)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # w1: input -> hidden weights matrix (input_size x hidden_size)
        # b1: hidden biases vector (1 x hidden_size)
        self.w1 = np.random.uniform(-1.0, 1.0, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1.0, 1.0, (1, hidden_size))
        
        # w2: hidden -> output weights matrix (hidden_size x output_size)
        # b2: output biases vector (1 x output_size)
        self.w2 = np.random.uniform(-1.0, 1.0, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1.0, 1.0, (1, output_size))

    def forward(self, inputs):
        """
        Runs inputs through the network.
        
        Inputs:
            inputs: List or 1D array of size input_size (floats)
            
        Outputs:
            outputs: Flat 1D array of size output_size (floats squashed to [-1.0, 1.0])
        """
        # Format inputs into shape (1, input_size) for matrix multiplication
        x = np.array(inputs).reshape(1, -1)
        
        # Layer 1: Input to Hidden
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        
        # Layer 2: Hidden to Output
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        
        return out.flatten()

    def get_weights(self):
        """
        Flattens all weights and biases into a single 1D numpy array.
        This provides direct access for mutation and crossover in evolution operators.
        
        Outputs:
            flat_weights: 1D NumPy float array
        """
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, flat_weights):
        """
        Reconstructs matrices and bias vectors from a flat 1D array.
        Used for updating the brain with evolved weights.
        
        Inputs:
            flat_weights: 1D NumPy float array
        """
        idx = 0
        w1_size = self.input_size * self.hidden_size
        self.w1 = flat_weights[idx : idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        b1_size = self.hidden_size
        self.b1 = flat_weights[idx : idx + b1_size].reshape(1, self.hidden_size)
        idx += b1_size
        
        w2_size = self.hidden_size * self.output_size
        self.w2 = flat_weights[idx : idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        b2_size = self.output_size
        self.b2 = flat_weights[idx : idx + b2_size].reshape(1, self.output_size)
