import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd


class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_shape, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class NNCreditScorer:
    def __init__(
        self,
        path_training_data,
        path_testing_data,
        device,
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
    ):
        self.X = None
        self.id = None
        self.criterion = None
        self.optimizer = None
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCHS = EPOCHS
        self.device = device
        self.train_loader = self.load_data(path_training_data)
        self.test_loader = self.load_data(path_testing_data, train=False)
        self.load_model()

    def load_data(self, path, train=True):
        """Loads and preprocess training or testing data.

        Args:
            path (string): path towards training or testing data
            train (bool, optional): if True load training data else load testing data. Defaults to True.

        Returns:
            dataloader: training or testing dataloader
        """
        df = pd.read_csv(path)
        df = df.fillna(df.mean())
        # Preprocessing
        X = df.drop(["SeriousDlqin2yrs", "Unnamed: 0"], axis=1).values
        scaler = StandardScaler().fit(X)
        self.X = scaler.transform(X)
        y = df["SeriousDlqin2yrs"].values

        if train == True:
            # Use dataloader to process data for Pytorch
            train_data = trainData(torch.FloatTensor(X), torch.FloatTensor(y))
            self.train_loader = DataLoader(
                dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True
            )
            return self.train_loader

        # Load testing data
        else:
            self.id = df["Unnamed: 0"].values
            test_data = testData(torch.FloatTensor(self.X))
            self.test_loader = DataLoader(dataset=test_data, batch_size=1)
            return self.test_loader

    def load_model(self):
        self.model = Net(self.X.shape[1])
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def binary_acc(self, y_pred, y_test):
        # Use sigmoid function to output probabilities
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        # Use accuracy for simplicity as evaluation metric
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def train(self):
        """Trains Neural Network for binary classification. Outputs loss and accuracy at each epoch."""
        self.model.train()
        for e in range(1, self.EPOCHS + 1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()

                y_pred = self.model(X_batch)

                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                acc = self.binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            print(
                f"Epoch {e+0:03}: | Loss: {epoch_loss/len(self.train_loader):.5f} | Acc: {epoch_acc/len(self.train_loader):.3f}"
            )

    def submit_kaggle(self, path_submission):
        """Model predicts probability for each sample of testing sets and creates submission dataframe.

        Args:
            path_submission (string): path of csv saved for kaggle submission
        """
        y_pred_list = []
        self.model.eval()
        with torch.no_grad():
            for X_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

        submission = pd.DataFrame(
            list(zip(self.id, y_pred_list)), columns=["Id", "Probability"]
        )
        submission.to_csv(path_submission, index=False)
