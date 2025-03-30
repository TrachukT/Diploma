import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def train(model, criterion, optimizer,device,train_loader):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_accuracy = correct / total
    train_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return train_loss / len(train_loader), train_accuracy, train_precision, train_recall, train_f1


def evaluate(model,device,test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    eval_accuracy = correct / total
    eval_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    eval_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    eval_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return eval_accuracy, eval_precision, eval_recall, eval_f1

