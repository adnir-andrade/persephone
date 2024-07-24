from torch import nn

class Classifier(nn.Module):
    '''
    Defines a neural network classifier with one hidden layer.
    
    Attributes:
        hidden1 (nn.Linear): Fully connected layer from input size to hidden layer size.
        output (nn.Linear): Fully connected layer from hidden layer size to output size.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer with a probability of 0.2 to prevent overfitting.
        log_softmax (nn.LogSoftmax): Log softmax function applied to the output.
    '''

    def __init__(self, input_size, h1_size, output_size):
        super(Classifier, self).__init__()
        self.hidden1 = nn.Linear(input_size, h1_size)
        self.output = nn.Linear(h1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.log_softmax(x)
        return x