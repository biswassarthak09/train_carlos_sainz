import torch
from agent.networks import CNN


class BCAgent:

    def __init__(self):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.net = CNN(history_length=5, n_classes=5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        X_batch = torch.tensor(X_batch).permute(0, 3, 1, 2)
        y_batch = torch.LongTensor(y_batch).view(-1)

        outputs = self.predict(X_batch)
        loss = self.criterion(outputs, y_batch)

        self.optimizer.zero_grad()
   
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        return self.net(X)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
