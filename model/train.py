from model.model import AcademicStressDetector
from pytorch_lightning import Trainer
import torch

import const.parameters as parameters
from utils.dataset import input_size


trainer = Trainer(max_epochs=parameters.num_epochs, fast_dev_run=False)
model = AcademicStressDetector(
    input_size=input_size, 
    hidden_size=parameters.hidden_size,
    num_classes=parameters.num_classes,
    dropout=parameters.dropout,
    gamma=parameters.gamma,
    lr=parameters.learning_rate
)
trainer.fit(model)
torch.save(model.state_dict(), "./models/model.pth")