from sklearn.metrics import confusion_matrix
import torch.nn as nn

def conf_mat(test_loader, model, device):
    nb_classes = 2

    y_pred = []
    y_true = []

    device = 'cpu'
    model.to(device)
    for inputs, classes in test_loader:
        inputs.to(device)
        outputs = model(inputs)

        # Append batch prediction results
        y_pred.extend(outputs.to(device).detach().numpy().flatten())
        y_true.extend(classes.to(device).detach().numpy().flatten()) # Save Truth

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)