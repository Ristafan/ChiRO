import torch
from torch import nn
from tqdm import tqdm

import wandb


def train_attention_alpha(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        for batch_idx, (spectrograms, labels) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Convert to half precision to save memory
            spectrograms = spectrograms.half()

            optimizer.zero_grad()

            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(spectrograms)  # Process whole batch at once
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * train_correct / train_total})

            # Clear memory every few batches if needed
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                # Debug memory
                # print(f"Memory after batch {batch_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # --- Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "training_loss": train_loss,
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        # --- Early Stopping ---
        if config.early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
