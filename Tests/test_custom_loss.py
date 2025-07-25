import sys
import torch
sys.path.extend(['..', '.'])
print("sys.path:", sys.path)

from MIDS_script import CustomLossFunction, MinBCEWithLogitsLoss



def test():
    # First graph - 3 node cycle
    # Second graph - 3 node path
    # Third graph - 4 node square with one diagonal

    # Batch tensor indicating which nodes belong to which graph
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    # torch.manual_seed(0)

    # Fake logits (random predictions for each node)
    logits_stacked = [
        torch.randn(3, 1),
        torch.randn(3, 1),
        torch.randn(4, 1)
    ]
    logits_padded = torch.cat(logits_stacked, dim=0)


    # Target values (stacked for the original function, padded for the modified function)
    y_stacked = [
        torch.tensor([0,0,1,0,1,0,1,0,0], dtype=torch.float).unsqueeze(1),  # Flattened stacked target values
        torch.tensor([0,1,0], dtype=torch.float).unsqueeze(1),  # Flattened stacked target values
        torch.tensor([1,0,0,0,0,0,1,0], dtype=torch.float).unsqueeze(1)   # Flattened stacked target values
    ]
    y_padded = torch.tensor([
        [0, 0, 1],    # First graph (3 valid options)
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, -1],  # Second graph (1 valid option)
        [1, -1, -1],
        [0, -1, -1],
        [1, 0, -1],   # Third graph (2 valid options)
        [0, 0, -1],
        [0, 1, -1],
        [0, 0, -1]
    ], dtype=torch.float)


    # Compute loss using both loss functions
    original_loss_fn = CustomLossFunction()
    min_bce_loss_fn = MinBCEWithLogitsLoss()

    original_loss = 0
    for logits, y in zip(logits_stacked, y_stacked):
        loss = original_loss_fn(logits, y)
        original_loss += loss.item()
    original_loss /= len(logits_stacked)

    min_bce_loss = min_bce_loss_fn(logits_padded.squeeze(), y_padded, batch)
    min_bce_loss = min_bce_loss.item()

    are_equal = torch.isclose(torch.tensor(original_loss), torch.tensor(min_bce_loss))

    # Print results
    # print("Original Custom Loss:", original_loss)
    # print("Batch-based Min BCE With Logits Loss:", min_bce_loss)
    # print("Losses are equal:", torch.isclose(torch.tensor(original_loss), torch.tensor(min_bce_loss)))

    return are_equal


if __name__ == '__main__':
    for i in range(10000):
        if not test():
            print("Test failed")
            break
    else:
        print("All tests passed")

