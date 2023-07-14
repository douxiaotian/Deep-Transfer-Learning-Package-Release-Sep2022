# used in baseline models
class CNN_feedforward(nn.Module):
    def __init__(self, symptom_size, embedding_dim, number_of_classes):
        super().__init__()
        number_of_channel = 1
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride for the convolution
        padding = 1  # Padding for the input

        self.embedding = nn.Embedding(symptom_size, embedding_dim)
        self.conv1 = nn.Conv2d(number_of_channel, number_of_classes, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)


    def forward(self, x):
        x = self.embedding(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc3(x)
        return x
