import torch
import torch.nn as nn

class GenNet(nn.Module):

    def __init__(self, layers=4, batch_size=4, input_length=576):
        super(GenNet, self).__init__()
    

        self.input = nn.Sequential(
            nn.Conv1d(1, batch_size, 5),
            nn.BatchNorm1d(batch_size, momentum=0.9),
            nn.LeakyReLU(0.2),

            nn.Conv1d(batch_size, 32, 5),
            nn.BatchNorm1d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            )
        
        self.layers = nn.ModuleList()

        for _ in range(layers):
            self.layers.append(nn.Conv1d(32, 32, 4))
            self.layers.append(nn.BatchNorm1d(32, momentum=0.9))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.Dropout(0.3))
        
        conv_output_size = self._get_conv_output(input_length)

        self.out1 = nn.Linear(conv_output_size, 256)
        self.out2 = nn.Linear(256, 1)
    
    def _get_conv_output(self, length):
        x = torch.zeros(1, 1, length)
        x = self.input(x)
        for layer in self.layers:
            x = layer(x)
        return x.numel()

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.input(x)

        for layer in self.layers:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        
        x = self.out1(x)
        x = self.out2(x)

        return x