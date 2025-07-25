import torch
import pandas as pd
from collections import Counter

def print_dataset_splits(train_dataset, val_dataset, test_dataset):
    train_counter = Counter([data.x.shape[0] for data in train_dataset])  # type: ignore
    val_counter = Counter([data.x.shape[0] for data in val_dataset])  # type: ignore
    test_counter = Counter([data.x.shape[0] for data in test_dataset])  # type: ignore

    sizes = set(train_counter + val_counter + test_counter)
    total_per_size = {size: train_counter[size] + val_counter[size] + test_counter[size] for size in sizes}
    train_splits_per_size = {size: round(train_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}
    val_splits_per_size = {size: round(val_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}
    test_splits_per_size = {size: round(test_counter.get(size, 0) / total_per_size[size], 2) for size in sizes}

    data = []
    for size in sorted(sizes):
        data.append([size, train_counter.get(size, 0), val_counter.get(size, 0), test_counter.get(size, 0),
                    train_splits_per_size[size], val_splits_per_size[size], test_splits_per_size[size]])

    df = pd.DataFrame(data, columns=["Size", "Train", "Val", "Test", "Train Split", "Val Split", "Test Split"])
    df.loc["Total"] = ["Total", train_counter.total(), val_counter.total(), test_counter.total(), "", "", ""]
    df = df.set_index("Size").T  # Transpose the dataframe

    df_str = df.to_string()
    lines = df_str.split('\n')
    separator = '-' * len(lines[0])
    lines.insert(1, separator)  # Insert separator after the first row
    lines.insert(5, separator)  # Insert separator after the fourth row
    lines.append(separator)  # Append separator at the end
    print('\n'.join(lines))


def print_memory_state():
    # Get the current GPU memory usage in bytes
    allocated_memory = torch.cuda.memory_allocated()  # Memory occupied by tensors
    reserved_memory = torch.cuda.memory_reserved()    # Memory reserved by the allocator

    # Convert bytes to MB
    allocated_memory_MB = allocated_memory / 1024**2
    reserved_memory_MB = reserved_memory / 1024**2

    print(f"Allocated GPU Memory: {allocated_memory_MB:.2f} MB")
    print(f"Reserved GPU Memory: {reserved_memory_MB:.2f} MB")
    print(torch.cuda.memory_summary(device="cuda"))
    print()