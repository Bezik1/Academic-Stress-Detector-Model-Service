import torch

from model.model import AcademicStressDetector
import const.parameters as parameters
from utils.dataset import input_size
from utils.dataset import scaler

x_manual = torch.tensor([[4,1,3,4,2,1,1,1,3,1,4,1,4,4,5]], dtype=torch.float32)

y_manual = torch.tensor([1], dtype=torch.long)

x_manual_scaled = scaler.transform(x_manual)
x_manual_scaled = torch.tensor(x_manual_scaled, dtype=torch.float32)

model = AcademicStressDetector(
    input_size=input_size, 
    hidden_size=parameters.hidden_size,
    num_classes=parameters.num_classes,
    dropout=parameters.dropout,
    gamma=parameters.gamma,
    lr=parameters.learning_rate
)

model.load_state_dict(torch.load("./models/model.pth"))
model.eval()

with torch.no_grad():
    logits = model(x_manual_scaled)
    prediction = torch.argmax(logits, dim=1)
    print("Prediction:", prediction.item())