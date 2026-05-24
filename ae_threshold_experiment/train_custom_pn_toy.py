import torch
import torch.nn as nn
import torch.optim as optim

from models.custom_progressive_network import CustomProgressiveNetwork


def make_toy_data(num_samples=512, input_dim=147, task="sum_positive"):
    x = torch.randn(num_samples, input_dim)

    if task == "sum_positive":
        # Task 0: input toplamı pozitif mi?
        y = (x.sum(dim=1) > 0).long()

    elif task == "first_half_positive":
        # Task 1: input'un ilk yarısının toplamı pozitif mi?
        y = (x[:, : input_dim // 2].sum(dim=1) > 0).long()

    else:
        raise ValueError(f"Unknown task: {task}")

    return x, y


def train_column(model, column_index, x, y, epochs=30, lr=1e-3):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        logits = model(x, column_index=column_index)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Column {column_index} | "
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc:.4f}"
            )


def evaluate_column(model, column_index, x, y):
    with torch.no_grad():
        logits = model(x, column_index=column_index)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

    return acc


def main():
    input_dim = 147
    hidden_dim = 64
    output_dim = 2

    model = CustomProgressiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    print("\nCustom PN Toy Training")
    print("----------------------")

    # ----------------------------
    # Task 0
    # ----------------------------
    x0, y0 = make_toy_data(task="sum_positive", input_dim=input_dim)

    model.add_column()
    print("\nTraining Column 0 on Task 0")
    train_column(model, column_index=0, x=x0, y=y0)

    acc0_before = evaluate_column(model, 0, x0, y0)
    print(f"Column 0 accuracy on Task 0: {acc0_before:.4f}")

    # ----------------------------
    # Task 1
    # ----------------------------
    x1, y1 = make_toy_data(task="first_half_positive", input_dim=input_dim)

    model.add_column()
    print("\nTraining Column 1 on Task 1")
    train_column(model, column_index=1, x=x1, y=y1)

    acc1 = evaluate_column(model, 1, x1, y1)
    acc0_after = evaluate_column(model, 0, x0, y0)

    print("\nFinal Evaluation")
    print("----------------")
    print(f"Column 1 accuracy on Task 1: {acc1:.4f}")
    print(f"Column 0 accuracy on Task 0 before Column 1: {acc0_before:.4f}")
    print(f"Column 0 accuracy on Task 0 after Column 1:  {acc0_after:.4f}")

    if abs(acc0_before - acc0_after) < 0.01:
        print("Forgetting check: PASSED")
    else:
        print("Forgetting check: FAILED")


if __name__ == "__main__":
    main()