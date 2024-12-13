import torch as th
from tqdm import trange
from src.plots import custom_confusion_matrix

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def data_epoch(model, loader, criterion, optimizer, Train=True):
    ys     = []
    y_hats = []
    
    running_loss        = 0.0
    correct_predictions = 0.0
    total_samples = 0.0
    
    for X, y in loader:
        X, y = X.float(), y.float()

        if Train:
            # Zero the parameter gradients
            optimizer.zero_grad()
        
        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y)

        if Train:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        y_hat = y_hat.argmax(1)
        y_true = y.argmax(1)
        correct_predictions += (y_hat == y_true).sum().item()
        total_samples += y_true.size(0)

        ys += y.argmax(1).tolist()
        y_hats += y_hat.tolist()
        
        running_loss += loss.item()

    # Calculate and print epoch loss
    loss = running_loss / len(loader)
    acc  = correct_predictions / total_samples

    return loss, acc, ys, y_hats

def train_loop(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, early_stopping=False):
    train_losses     = []
    test_losses      = []
    train_accuracies = []
    test_accuracies  = []

    early = EarlyStopper() if early_stopping else None
        
    # model.to(device)
    bar = trange(num_epochs)
    
    for epoch in bar:
        model.train()

        train_loss, train_acc, _, _ = data_epoch(model, train_loader, criterion, optimizer)
        
        with th.no_grad():
            test_loss, test_acc, _, _ = data_epoch(model, test_loader, criterion, optimizer, Train=False)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        bar.set_postfix({"Train Loss": train_loss, 
                        "Test Loss": test_loss,
                        "Train Acc": train_acc, 
                        "Test Acc": test_acc})

        if early and early(test_loss):
            print(f"Stopped early, epoch: {epoch}")
            break

    return {"Train Loss": train_losses, 
            "Test Loss": test_losses,
            "Train Accuracy": train_accuracies, 
            "Test Accuracy": test_accuracies}

def test_model(model, test_loader, criterion):
    with th.no_grad():
        test_loss, test_acc, y, y_hat = data_epoch(model, test_loader, criterion, None, Train=False)
    
    print(f"Test Loss: {test_loss}", f"Test Accuracy: {test_acc}", sep="\n")

    custom_confusion_matrix(y, y_hat)

def init_weights(layer):
    if isinstance(layer, th.nn.Linear) or isinstance(layer, th.nn.Conv2d):
        th.nn.init.normal_(layer.weight, mean=0, std=0.01)
        layer.bias.data.fill_(0)