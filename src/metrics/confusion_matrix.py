from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch

def conf_mat(test_loader, model, device):
    nb_classes = 2

    y_pred = []
    y_true = []

    device = 'cpu'
    with torch.no_grad():
        model.to(device)
        for inputs, classes in test_loader:
            inputs.to(device)
            outputs = torch.round(model(inputs))

            # Append batch prediction results
            y_pred.extend(outputs.to(device).detach().numpy().flatten())
            y_true.extend(classes.to(device).detach().numpy().flatten()) # Save Truth

    # Confusion matrix
    set_pred = set(y_pred)
    set_true = set(y_true)
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    print('classes pred', set_pred, 'true', set_true)