import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

from utils.dataset import train_loader, test_loader, val_loader


class AcademicStressDetector(pl.LightningModule):
    """
    PyTorch Lightning model for predicting academic stress level based on
    physical and psychological features.

    Features:
        anxiety_level                   (int, 0-21)
        self_esteem                     (int, 0-30)
        mental_health_history           (int, 0-1)
        depression                      (int, 0-27)
        headache                        (int, 0-5)
        blood_pressure                  (int, 0-3)
        sleep_quality                    (int, 0-5)
        breathing_problem                (int, 0-5)
        noise_level                      (int, 0-5)
        living_conditions                (int, 0-5)
        safety                           (int, 0-5)
        basic_needs                      (int, 0-5)
        academic_performance             (int, 0-5)
        study_load                       (int, 0-5)
        teacher_student_relationship     (int, 0-5)
        future_career_concerns           (int, 0-5)
        social_support                   (int, 0-3)
        peer_pressure                    (int, 0-5)
        extracurricular_activities       (int, 0-5)
        bullying                         (int, 0-5)
    
    Target:
        stress_level                     (int, 0-2) [Eustress, Distress, No Stress]

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in hidden layers.
        num_classes (int): Number of output classes.
        lr (float, optional): Learning rate. Default: 1e-3.
        dropout (float, optional): Dropout probability. Default: 0.5.
        gamma (float, optional): Learning rate decay factor. Default: 0.95.

    Methods:
        forward(x): Computes logits for input tensor x.
        training_step(batch, batch_idx): Training step with loss and accuracy logging.
        validation_step(batch, batch_idx): Validation step with loss and accuracy logging.
        test_step(batch, batch_idx): Test step with accuracy logging.
        configure_optimizers(): Sets up optimizer and learning rate scheduler.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, lr=1e-3, dropout=0.5, gamma=0.95):
        super(AcademicStressDetector, self).__init__()
        self.save_hyperparameters()
        
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        output = self.sequential(x)
        return output
    
    def train_dataloader(self):
        return train_loader
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def val_dataloader(self):
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def test_dataloader(self):
        return test_loader
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }
    