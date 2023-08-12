"""Linear regression model for predicting cab fares using PyTorch.

Adapted from https://gist.github.com/pdet/e8d38734232c08e6c15aba79b4eb8368#file-taxi_prediction_example-py.
"""
from __future__ import annotations

import pyarrow as pa
import torch
import tqdm
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, distances):
        return self.linear(distances)


class PredictCabFare:
    def __init__(self, data, learning_rate: float = 0.01, epochs: int = 100) -> None:
        # Define the input and output dimensions
        input_dim = 1
        output_dim = 1

        # Create a linear regression model instance
        self.data = data
        self.model = LinearRegression(input_dim, output_dim)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self):
        distances = self.data["trip_distance"].reshape(-1, 1)
        fares = self.data["fare_amount"].reshape(-1, 1)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for _ in tqdm.trange(self.epochs):
            # Forward pass
            y_pred = self.model(distances)

            # Compute loss
            loss = criterion(y_pred, fares)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, input):
        with torch.no_grad():
            return self.model(input)

    def __call__(self, input: pa.ChunkedArray):
        # Convert the input to numpy so it can be fed to the model
        #
        # .copy() to avoid the warning about undefined behavior from torch
        input = torch.from_numpy(input.to_numpy().copy())[:, None]
        predicted = self.predict(input).ravel()
        return pa.array(predicted.numpy())
