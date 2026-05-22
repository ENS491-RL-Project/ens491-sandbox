import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveColumn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        y = self.out(h2)

        return y, [h1, h2]


class ProgressiveColumnWithLateral(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_previous_columns):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        self.lateral1 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_previous_columns)
        ])

        self.lateral2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_previous_columns)
        ])

    def forward(self, x, previous_activations):
        h1 = self.fc1(x)

        for i, activations in enumerate(previous_activations):
            prev_h1 = activations[0]
            h1 = h1 + self.lateral1[i](prev_h1)

        h1 = F.relu(h1)

        h2 = self.fc2(h1)

        for i, activations in enumerate(previous_activations):
            prev_h2 = activations[1]
            h2 = h2 + self.lateral2[i](prev_h2)

        h2 = F.relu(h2)

        y = self.out(h2)

        return y, [h1, h2]


class CustomProgressiveNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.columns = nn.ModuleList()

    def add_column(self):
        if len(self.columns) == 0:
            column = ProgressiveColumn(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
            )
        else:
            self.freeze_existing_columns()

            column = ProgressiveColumnWithLateral(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_previous_columns=len(self.columns),
            )

        self.columns.append(column)

    def freeze_existing_columns(self):
        for column in self.columns:
            for param in column.parameters():
                param.requires_grad = False

    def forward(self, x, column_index):
        if column_index >= len(self.columns):
            raise ValueError("Column index out of range")

        if column_index == 0:
            y, activations = self.columns[0](x)
            return y

        previous_activations = []

        with torch.no_grad():
            for i in range(column_index):
                _, activations = self.columns[i](x)
                previous_activations.append(activations)

        y, _ = self.columns[column_index](x, previous_activations)

        return y