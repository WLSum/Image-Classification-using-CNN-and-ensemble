import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import random
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################### Subroutines #####################################################################

#----------- Pre-processing ----------
class DataPreprocessing:
    """ Preprocessing steps for input data """
    def __init__(self):
        """Define label mapping for class names to numeric labels"""
        self.label_map = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 
                          'Insidecity': 6, 'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10,
                          'OpenCountry': 11, 'store': 12, 'Street': 13, 'Suburb': 14, 'TallBuilding': 15}


    def transform_train(self):
        """ Data augmentation for training data """
        return transforms.Compose([
            Multisize_Input(sizes=(180, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])

    def transform_test(self):
        """ Preprocessing pipeline for validation/testing (no augmentation) """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def collate_fn_multisize(self, batch):
        """Collate function for handling multi-sized images in the batch."""
        images, labels = zip(*batch)
        custom_labels = [self.label_map[self.classes[label]] for label in labels]
        return list(images), torch.tensor(custom_labels)

    def collate_fn_singlesize(self, batch):
        """Collate function for handling single-sized images in the batch."""
        images, labels = zip(*batch)
        custom_labels = [self.label_map[self.classes[label]] for label in labels]
        return torch.stack(images, dim=0), torch.tensor(custom_labels)
    
class Multisize_Input:
    """ Resize image randomly to either 180x180 or 224x224 """
    def __init__(self, sizes=(180, 224)):
        self.sizes = sizes

    def __call__(self, img):
        size = random.choice(self.sizes)
        return transforms.Resize((size, size))(img)

#----------- Training ----------
class Train_Test:
    def __init__(self, model_class, lr=0.001, saved_path="model_dir"):
        """Initialize model, loss, optimizer and learning rate scheduler"""
        self.model = model_class(15).to(device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.saved_path = saved_path
        self.train_acc_list = []
        self.val_acc_list = []

    def pad_to_max_size(self, batch):
        """Pad images in a batch to the same size"""
        max_h = max([img.size(1) for img in batch])
        max_w = max([img.size(2) for img in batch])
        padded_batch = []
        for img in batch:
            channel, height, width = img.shape
            pad_h = max_h - height
            pad_w = max_w - width
            padded = F.pad(img, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
            padded_batch.append(padded)
        return torch.stack(padded_batch)

    def train(self, train_loader, val_loader, epochs=100):
        best_val_acc = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for inputs, labels in train_loader:
                inputs = [img.to(device) for img in inputs]
                inputs = self.pad_to_max_size(inputs)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels - 1)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted += 1
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total * 100
            self.train_acc_list.append(train_acc)

            val_acc = self.evaluate(val_loader)
            self.val_acc_list.append(val_acc)

            self.scheduler.step(val_acc)

            # Save model if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.saved_path)
                # print(f"Epoch {epoch+1}/{epochs}, Model saved - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    def evaluate(self, data_loader):
        """Evaluate model accuracy on a train dataset (multi-sized)"""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = [img.to(device) for img in inputs]
                inputs = self.pad_to_max_size(inputs)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted +=1
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total * 100

class EnsembleModel(nn.Module):
    """Ensemble for the 3 models by soft voting"""
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):  
        out1 = torch.softmax(self.model1(x), dim=1)
        out2 = torch.softmax(self.model2(x), dim=1)
        out3 = torch.softmax(self.model3(x), dim=1)
        return (out1 + out2 + out3) / 3  


def test_evaluate(model, data_loader):
    """Evaluate model accuracy on a validation/test dataset (single-sized)"""
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted +=1
            correct += ((predicted) == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100


###################################### Main train and test Function ####################################################
def train(train_data_dir, model_dir):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    #----------- Load & Pre-process input data ----------
    # Load and transform the data
    data_preprocessor = DataPreprocessing()
    dataset_train = datasets.ImageFolder(root=train_data_dir, transform=data_preprocessor.transform_train())
    dataset_val = datasets.ImageFolder(root=train_data_dir, transform=data_preprocessor.transform_test())
    data_preprocessor.classes = dataset_train.classes

    # Split 80% training and 20% validation with stratification
    targets = [label for _, label in dataset_train.samples]
    train_idx, val_idx = train_test_split(list(range(len(dataset_train))), test_size=0.2, stratify=targets, random_state=123)

    # Create train and validation sets
    train_subset = Subset(dataset_train, train_idx)
    val_subset = Subset(dataset_val, val_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=data_preprocessor.collate_fn_multisize)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=data_preprocessor.collate_fn_singlesize)

    #----------- Construction and Training CNN  ----------
    # Constructed self-built CNN, custom MobileNetv2 and custom VGG2 in model.py

    # Train self-built CNN
    if os.path.exists("trained_custom_cnn.pth"): 
        Built_CNN_model = BuiltCNN(num_classes=15).to(device)
        Built_CNN_model.load_state_dict(torch.load("trained_custom_cnn.pth", weights_only=True))
    else:
        cnn_trainer = Train_Test(BuiltCNN, lr=0.001, saved_path="trained_custom_cnn.pth")
        cnn_trainer.train(train_loader, val_loader, epochs=100)
        Built_CNN_model = cnn_trainer.model 

    # Train custom MobileNetv2
    if os.path.exists("trained_mobilenet.pth"): 
        mobilenet_model = MobileNetV2_Custom(num_classes=15).to(device)
        mobilenet_model.load_state_dict(torch.load("trained_mobilenet.pth", weights_only=True))
    else:
        mobilenet_trainer = Train_Test(MobileNetV2_Custom, lr=0.001, saved_path="trained_mobilenet.pth")
        mobilenet_trainer.train(train_loader, val_loader, epochs=50)
        mobilenet_model = mobilenet_trainer.model  

    # Train custom VGG16
    if os.path.exists("trained_vgg.pth"): 
        vgg_model = VGG16_Custom(num_classes=15).to(device)
        vgg_model.load_state_dict(torch.load("trained_vgg.pth", weights_only=True))
    else:
        vgg_trainer = Train_Test(VGG16_Custom, lr=0.0005, saved_path="trained_vgg.pth")
        vgg_trainer.train(train_loader, val_loader, epochs=50)
        vgg_model = vgg_trainer.model  

    # Combine to form an Ensemble model (Final) - saved as trained_cnn.pth
    ensemble_model = EnsembleModel(Built_CNN_model, mobilenet_model, vgg_model)
    torch.save(ensemble_model.state_dict(), model_dir)

    # Determine training accuracy
    evaluator = Train_Test(lambda num_classes: ensemble_model)
    evaluator.model = ensemble_model 
    evaluator.model.eval()
    train_acc = evaluator.evaluate(train_loader)
    return train_acc


def test(test_data_dir, model_dir):
    #----------- Load & Pre-process input data ----------
    # Load and transform the data
    data_preprocessor = DataPreprocessing()
    dataset_test = datasets.ImageFolder(root=test_data_dir, transform=data_preprocessor.transform_test())
    data_preprocessor.classes = dataset_test.classes
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=data_preprocessor.collate_fn_singlesize)
    
    #----------- Load Pre-Trained ----------
    # Load the different models, and the trained ensemble model 
    builtcnn = BuiltCNN(num_classes=15).to(device)
    mobilenet = MobileNetV2_Custom(num_classes=15).to(device)
    vgg16 = VGG16_Custom(num_classes=15).to(device)

    ensemble_model = EnsembleModel(builtcnn, mobilenet, vgg16)
    ensemble_model.load_state_dict(torch.load(model_dir, weights_only=True))
    ensemble_model.eval()

    #----------- Evaluate ----------
    test_acc = test_evaluate(ensemble_model,test_loader)
    return test_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
