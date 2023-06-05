import torch.nn as nn

class TFBindReward1HOT(nn.Module):
    def __init__(self):
        super(TFBindReward1HOT, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(40, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1))
        
    def forward(self,x):
        return self.model(x)
    
