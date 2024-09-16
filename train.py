import numpy as np
import matplotlib.pyplot as plt

import torch
from torchinfo import summary
import segmentation_models_pytorch as smp

from tqdm import tqdm

from model import MAnet

from data_processing import create_dataloader, img_size

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path='best_checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)

        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def train_step(model, dataloader, loss_fn, optimizer, num_classes):

    model.train()

    train_loss = 0.
    train_iou = 0.

    for (img, mask) in tqdm(dataloader):
        img = img.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)

        optimizer.zero_grad()

        pred = model(img)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()

        pred_prob = pred.softmax(dim=1)
        pred_class = pred_prob.argmax(dim=1) 

        tp, fp, fn, tn = smp.metrics.get_stats(pred_class.detach().cpu().long(), 
                                               mask.detach().cpu(), 
                                               mode="multiclass", 
                                               ignore_index=-1,
                                               num_classes=num_classes)

        train_loss += loss.item()
        train_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')        

    train_loss /= len(dataloader)
    train_iou /= len(dataloader)

    return train_loss, train_iou

def val_step(model, dataloader, loss_fn, num_classes):

    model.eval()

    val_loss = 0.
    val_iou = 0.

    with torch.no_grad():
        for i, (img, mask) in enumerate(dataloader):
            img = img.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            pred = model(img)

            loss = loss_fn(pred, mask)

            val_loss += loss.item()
            pred_prob = pred.softmax(dim=1)
            pred_class = pred_prob.argmax(dim=1)

            tp, fp, fn, tn = smp.metrics.get_stats(pred_class.detach().cpu().long(), 
                                                   mask.detach().cpu(), 
                                                   mode="multiclass", 
                                                   ignore_index=-1,
                                                   num_classes=num_classes)

            val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

    val_loss /= len(dataloader)
    val_iou /= len(dataloader)

    return val_loss, val_iou


def train(model, train_dataset, val_dataset, 
          loss_fn, optimizer, num_classes, num_epochs, early_stopping):

    results = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    for epoch in range(num_epochs):

        train_loss, train_iou = train_step(model, train_dataset, loss_fn, optimizer, num_classes)
        val_loss, val_iou = val_step(model, val_dataset, loss_fn, num_classes)

        results["train_loss"].append(train_loss)
        results["train_iou"].append(train_iou)
        results["val_loss"].append(val_loss)
        results["val_iou"].append(val_iou)

        results["smooth_val_loss"] = np.mean(results["val_loss"][-10:])
        results["smooth_val_iou"] = np.mean(results["val_iou"][-10:])

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"Smooth Val Loss: {results['smooth_val_loss']:.4f}, Smooth Val IoU: {results['smooth_val_iou']:.4f}")

        early_stopping(1-val_iou, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return results

def visualize_loss_and_metric(results):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    ax[0].plot(results['train_loss'], label='Train Loss', color='blue')
    ax[0].plot(results['val_loss'], label='Val Loss', color='red')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(results['train_iou'], label='Train IoU', color='blue')
    ax[1].plot(results['val_iou'], label='Val IoU', color='red')
    ax[1].set_title('IoU')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('IoU')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def get_test_preds(test_dataloader, best_model):
    checkpoint = torch.load(best_model)

    loaded_model = MAnet(in_channels=3, classes=num_classes).to(device)
    loaded_model.load_state_dict(checkpoint)
    loaded_model.eval()

    pred_mask_test = []

    with torch.no_grad():
        for i, (img, _) in enumerate(test_dataloader):
            img = img.to(device, dtype=torch.float32)

            pred = loaded_model(img)
            pred_prob = pred.softmax(dim=1)
            pred_class = pred_prob.argmax(dim=1)

            pred_mask_test.append(pred_class.detach().cpu())
    
    return torch.cat(pred_mask_test)


def get_test_iou(pred_mask_test, test_dataloader):

    gt_mask_test = []

    for i, (_, mask) in enumerate(test_dataloader):
        gt_mask_test.append(mask)

    gt_mask_test = torch.cat(gt_mask_test)

    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask_test.long(),
                                        gt_mask_test.long(),
                                            mode="multiclass",
                                            ignore_index=-1,
                                            num_classes=num_classes)

    iou_test = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

    return iou_test


if __name__ == "__main__":

    seed = 43

    # Get data
    batch_size = 16
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(batch_size)

    # Get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 5
    model = MAnet(classes=num_classes).to(device)
    print(summary(model, input_size=(batch_size, 3, *img_size)))

    # Freeze the encoder layer
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Loss function and optimizer
    loss_fn = smp.losses.FocalLoss(mode='multiclass', ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train
    num_epochs = 1000
    early_stopping = EarlyStopping(patience=25, delta=0.0)

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    
    results = train(model,
                    train_dataloader,
                    val_dataloader,
                    loss_fn,
                    optimizer,
                    num_classes,
                    num_epochs,
                    early_stopping)

    visualize_loss_and_metric(results)
    

    # Test
    best_model = 'best_checkpoint.pt'
    pred_mask_test = get_test_preds(test_dataloader, best_model)
    iou_test = get_test_iou(pred_mask_test, test_dataloader)
    print(f"Test IoU: {iou_test:.4f}")

