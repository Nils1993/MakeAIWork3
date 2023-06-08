from torch import nn
import torch.nn.functional


class CNN(nn.Module):
    
    def __init__(self, img_size=128, c_in=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(c_in, 16, 5, padding=2),
            nn.ReLU(),
            nn. Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn. Conv2d(64, 96, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(0),
        
            

       )


    def forward(self, data):

        return self.model(data)


