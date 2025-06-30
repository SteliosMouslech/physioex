import torch
import torch.nn as nn
from physioex.train.networks.base import SleepModule
import torch.optim as optim
from torchmetrics.functional import r2_score


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(config["in_channels"], 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # output shape: (B, 256, 1)

            nn.Flatten()  # → (B, 256)
        )

    def forward(self, x):
        return self.cnn(x)

class AgeRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.regressor(x)



class AgeNet(nn.Module):
    def __init__(self, module_config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(module_config)
        self.regressor = AgeRegressor(module_config)

    def encode(self, x):
        # x shape: (B, S, C, T)
        batch_size, seq_len, channels, time = x.shape
        x = x.reshape(batch_size * seq_len, channels, time)

        features = self.feature_extractor(x)          # (B*S, 256)
        preds = self.regressor(features)              # (B*S, 1)

        # Reshape back to (B, S, 1)
        features = features.view(batch_size, seq_len, -1)
        preds = preds.view(batch_size, seq_len, 1)
        return features, preds

    def forward(self, x):
        _, preds = self.encode(x)
        return preds


class AgePredNet(SleepModule):
    def __init__(self, module_config):
        super().__init__(AgeNet(module_config), module_config)

        # Buffers for correlation plotting
        self.train_outputs = []
        self.train_targets = []
        self.val_outputs = []
        self.val_targets = []

   

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return self.opt
    
    def compute_loss(self, embeddings, outputs, targets, log: str = "train", log_metrics: bool = False):
        # Save flattened copies for correlation plots
        flat_preds = outputs.view(-1).detach()
        flat_targets = targets.view(-1).detach()

        if log == "train":
            self.train_outputs.append(flat_preds)
            self.train_targets.append(flat_targets)
        elif log == "val":
            self.val_outputs.append(flat_preds)
            self.val_targets.append(flat_targets)

        # Now delegate actual loss computation + logging to the base class
        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)



    def on_train_epoch_end(self):
        self._log_correlation(self.train_outputs, self.train_targets, log="train")
        self.train_outputs.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        self._log_correlation(self.val_outputs, self.val_targets, log="val")

    # Compute and log R2 using all validation predictions
        preds = torch.cat(self.val_outputs)
        targets = torch.cat(self.val_targets)
        r2_val = r2_score(preds, targets)
        self.log("val_r2_epoch", r2_val, prog_bar=True, sync_dist=True)


        self.val_outputs.clear()
        self.val_targets.clear()

    def _log_correlation(self, outputs_list, targets_list, log="train"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not outputs_list:
            return

        preds = torch.cat(outputs_list).cpu().numpy()
        targets = torch.cat(targets_list).cpu().numpy()

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=targets, y=preds)
        plt.xlabel("True Age")
        plt.ylabel("Predicted Age")
        plt.title(f"{log.capitalize()} Correlation")

        self.logger.experiment.add_figure(f"{log}_age_correlation", plt.gcf(), self.current_epoch)
        plt.close()


    def training_step(self, batch, batch_idx):
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets, subjects, dataset_idx = batch
        embeddings, outputs = self.encode(inputs)

        # outputs: (B, S, 1) → (B, 1, 1)
        outputs = outputs.mean(dim=1, keepdim=True)
        # targets: (B,) → (B, 1)
        targets = targets.unsqueeze(1)

        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Validation Logic
        inputs, targets, subjects, dataset_idx = batch
        embeddings , outputs = self.regression_voting_strategy(inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets, subjects, dataset_idx = batch

        embeddings , outputs = self.regression_voting_strategy(inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)
    

    def regression_voting_strategy(self, inputs: torch.Tensor, L: int = 1):
        """
        Overlapping window voting strategy for age regression. For models with no sequence dependency,
        set L=1 to simply average per-epoch predictions.
        
        Args:
            inputs: (batch, total_seq_len, channels, timepoints)
            L: window size. Use 1 for per-epoch processing without overlaps.

        Returns:
            mean_embeddings: (batch, feat_dim)
            mean_preds: (batch, 1)
        """
        batch_size, total_len, nchan, nsamp = inputs.shape
        n_windows = total_len - L + 1

        all_preds = []
        all_embeds = []

        for i in range(n_windows):
            segment = inputs[:, i:i+L]  # shape: (B, L, C, T)
            emb, pred = self.encode(segment)  # pred: (B, L, 1)

            # Optionally collapse predictions across L (not needed if L=1)
            pred = pred.mean(dim=1, keepdim=True)  # (B, 1)
            emb = emb.mean(dim=1)                  # (B, feat_dim)

            all_preds.append(pred)
            all_embeds.append(emb)

        preds = torch.stack(all_preds, dim=1)   # (B, n_windows, 1)
        embeds = torch.stack(all_embeds, dim=1) # (B, n_windows, feat_dim)

        mean_preds = preds.mean(dim=1)          # (B, 1)
        mean_embeds = embeds.mean(dim=1)        # (B, feat_dim)

        return mean_embeds, mean_preds

