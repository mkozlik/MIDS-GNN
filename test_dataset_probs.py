from MIDS_script import load_dataset


def calculate_error(data):
    errors = data.x[:, 10] - data.x[:, 9]
    avg_error = errors.mean()
    avg_abs_error = errors.abs().mean()
    std_error = errors.std()
    print(f"Average error: {avg_error:.3f}")
    print(f"Average absolute error: {avg_abs_error:.3f}")
    print(f"Standard deviation of the error: {std_error:.3f}")


if __name__ == "__main__":

    train_data_obj, val_data_obj, test_data_obj, dataset_config, features, dataset_props = load_dataset(
        batch_size=1.0,
        split=(0.6, 0.2),
        suppress_output=False,
    )

    print("Training data:")
    calculate_error(train_data_obj)

    print("\nValidation data:")
    calculate_error(val_data_obj)

    print("\nTest data:")
    calculate_error(test_data_obj)
