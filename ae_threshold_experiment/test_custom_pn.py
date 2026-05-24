import torch

from models.custom_progressive_network import CustomProgressiveNetwork


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    input_dim = 147      # 7 * 7 * 3 MiniGrid observation flattened
    hidden_dim = 64
    output_dim = 7       # MiniGrid action space gibi düşün

    model = CustomProgressiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    print("Custom Progressive Network Test")
    print("-------------------------------")

    # Add first column
    model.add_column()
    print(f"Number of columns after first add: {len(model.columns)}")

    x = torch.randn(4, input_dim)
    y = model(x, column_index=0)

    print(f"Column 0 output shape: {y.shape}")
    print(f"Trainable parameters after column 0: {count_trainable_parameters(model)}")

    # Add second column
    model.add_column()
    print(f"\nNumber of columns after second add: {len(model.columns)}")

    y = model(x, column_index=1)

    print(f"Column 1 output shape: {y.shape}")
    print(f"Trainable parameters after column 1: {count_trainable_parameters(model)}")

    # Check frozen first column
    column_0_trainable = any(
        param.requires_grad for param in model.columns[0].parameters()
    )

    column_1_trainable = any(
        param.requires_grad for param in model.columns[1].parameters()
    )

    print("\nFreeze check")
    print("------------")
    print(f"Column 0 trainable: {column_0_trainable}")
    print(f"Column 1 trainable: {column_1_trainable}")

    if y.shape == (4, output_dim) and not column_0_trainable and column_1_trainable:
        print("\nTEST PASSED")
    else:
        print("\nTEST FAILED")


if __name__ == "__main__":
    main()