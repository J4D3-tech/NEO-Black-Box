import torch
import torch.nn as nn

class NeoBlackBox(nn.Module):
    def __init__(self, input_size):
        super(NeoBlackBox, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # WYJŚCIE: Tylko 1 neuron! Uczy się wyłącznie MOID na podstawie ukrytych wzorców w danych
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.net(x)

def blackbox_loss(y_pred, y_true_scaled):
    # y_pred i y_true_scaled mają kształt [N, 1]
    moid_pred_scaled = y_pred[:, 0]
    moid_true_scaled = y_true_scaled[:, 0]
    
    # System wag (skupienie na groźnych obiektach blisko Ziemi) - to logika dziedzinowa, nie fizyka
    danger_weights = torch.exp(-moid_true_scaled).clamp(max=10.0)
    mse_loss_per_element = (moid_pred_scaled - moid_true_scaled) ** 2
    
    # Zwykłe ważone MSE, całkowity brak kary fizycznej
    total_loss = torch.mean(mse_loss_per_element * danger_weights)
    return total_loss
