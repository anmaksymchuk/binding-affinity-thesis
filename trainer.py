import os
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        device,
        criterion,
        optimizer,
        scheduler=None,
        y_stats=None,
        grad_clip_norm=1.0,
        checkpoint_path="best_model_rmse.pt",
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.y_stats = y_stats
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_path = checkpoint_path

        if self.y_stats is None:
            raise ValueError(
                "y_stats must be provided. Compute it only on the training split."
            )

        if "mean" not in self.y_stats or "std" not in self.y_stats:
            raise ValueError("y_stats must contain 'mean' and 'std'.")

        if self.y_stats["std"] < 1e-8:
            raise ValueError("y_stats['std'] is too small for stable normalization.")

    def normalize_y(self, y):
        """
        Normalize target values using train-set statistics.
        """
        return (y - self.y_stats["mean"]) / self.y_stats["std"]

    def denormalize_y(self, y_norm):
        """
        Convert normalized predictions back to the original pKd scale.
        """
        return y_norm * self.y_stats["std"] + self.y_stats["mean"]

    def train_step(self, loader):
        """
        One training epoch.

        The model predicts normalized pKd values.
        The target is normalized using train-set statistics.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            y_norm = self.normalize_y(batch.y).view(-1)
            pred_norm = self.model(batch).view(-1)

            loss = self.criterion(pred_norm, y_norm)

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @staticmethod
    def concordance_index(y_true, y_pred):
        """
        Concordance index.

        Measures how well the model preserves pairwise ordering.

        Returns:
            1.0 = perfect ranking
            0.5 = random ranking
            0.0 = completely wrong ranking
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        true_diff = y_true[:, None] - y_true[None, :]
        pred_diff = y_pred[:, None] - y_pred[None, :]

        valid = true_diff > 0
        num_valid = np.sum(valid)

        if num_valid == 0:
            return 0.5

        num_correct = np.sum((pred_diff > 0) & valid)
        num_tied = np.sum((pred_diff == 0) & valid)

        return float((num_correct + 0.5 * num_tied) / num_valid)

    @staticmethod
    def pearson_corr(y_true, y_pred):
        """
        Pearson correlation coefficient between true and predicted pKd values.

        Returns 0.0 if either vector has almost zero variance.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
            return 0.0

        return float(np.corrcoef(y_true, y_pred)[0, 1])

    def eval_step(self, loader):
        """
        Evaluation step.

        Returns metrics computed on the original pKd scale:
            RMSE
            MAE
            CI
            Pearson
        """
        self.model.eval()

        ys_raw = []
        preds_norm = []

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device)

                pred_norm = self.model(batch).view(-1)

                ys_raw.append(batch.y.view(-1).cpu().numpy())
                preds_norm.append(pred_norm.cpu().numpy())

        ys_raw = np.concatenate(ys_raw)
        preds_norm = np.concatenate(preds_norm)

        preds_raw = self.denormalize_y(preds_norm)

        rmse = np.sqrt(np.mean((ys_raw - preds_raw) ** 2))
        mae = np.mean(np.abs(ys_raw - preds_raw))
        ci = self.concordance_index(ys_raw, preds_raw)
        pearson = self.pearson_corr(ys_raw, preds_raw)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "ci": float(ci),
            "pearson": float(pearson),
        }

    def train(
        self,
        epochs,
        train_loader,
        val_loader,
        early_stop=True,
        patience=15,
    ):
        """
        Full training loop.

        Early stopping is based on validation RMSE.
        The best model is saved to self.checkpoint_path.
        """
        best_rmse = float("inf")
        counter = patience

        history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_ci": [],
            "val_pearson": [],
        }

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_step(train_loader)
            val_metrics = self.eval_step(val_loader)

            rmse = val_metrics["rmse"]
            mae = val_metrics["mae"]
            ci = val_metrics["ci"]
            pearson = val_metrics["pearson"]

            if self.scheduler is not None:
                self.scheduler.step(rmse)

            history["train_loss"].append(train_loss)
            history["val_rmse"].append(rmse)
            history["val_mae"].append(mae)
            history["val_ci"].append(ci)
            history["val_pearson"].append(pearson)

            status = ""

            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(self.model.state_dict(), self.checkpoint_path)
                status = " [BEST RMSE]"
                counter = patience
            else:
                counter -= 1

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"RMSE: {rmse:.4f} | "
                f"MAE: {mae:.4f} | "
                f"CI: {ci:.4f} | "
                f"Pearson: {pearson:.4f}"
                f"{status}"
            )

            if early_stop and counter <= 0:
                print("\nEarly stopping triggered.")
                break

        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
        else:
            print(
                f"Warning: checkpoint {self.checkpoint_path} was not found. "
                "Returning the current model."
            )

        return self.model, history