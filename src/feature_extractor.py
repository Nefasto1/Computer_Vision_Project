import torch as th

def extract_features(model, loader):
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    new_X = []
    new_y = []
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        new_X.append(model(X))
        new_y.append(y)
    
    new_X = th.cat(new_X).detach().cpu()
    new_y = th.cat(new_y).detach().cpu().argmax(1)

    return new_X, new_y

def extract_svm_inputs(model, train_loader, validation_loader, test_loader):
    train      = extract_features(model, train_loader)
    validation = extract_features(model, validation_loader)
    test       = extract_features(model, test_loader)

    return train, validation, test