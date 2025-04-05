from sklearn.metrics import classification_report, confusion_matrix
import torch

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
