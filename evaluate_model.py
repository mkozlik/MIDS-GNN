import os


import torch

from MIDS_script import CustomLossFunction, LossWrapper, MinBCEWithLogitsLoss, evaluate, load_dataset
from Utilities.gnn_models import GATLinNet, GNNWrapper

if "PBS_O_HOME" in os.environ:
    # We are on the HPC - adjust for the CPU count and VRAM.
    BATCH_SIZE = 1/3
else:
    BATCH_SIZE = 0.25


def main(config):
    # GLOBALS: device

    # Helper boolean flags.
    plot_graphs = False
    make_table = False

    if "PBS_O_HOME" in os.environ:
        # We are on the HPC - paralel runs use the same disk.
        BEST_MODEL_PATH = "/lustre/home/mkrizman/MIDS-GNN/Models/prob_model_best.pth"
    else:
        BEST_MODEL_PATH = "/home/marko/PROJECTS/MIDS-GNN/Models/prob_model_trained.pth"

    # Load the dataset.
    train_data_obj, val_data_obj, test_data_obj, dataset_config, features, dataset_props = load_dataset(
        selected_extra_feature=config.get("selected_extra_feature"),
        batch_size=BATCH_SIZE,
        split=config.get("dataset", {}).get("split", (0.6, 0.2)),
        suppress_output=False,
    )

    criterion_options = {
        "true_labels_all_padded": MinBCEWithLogitsLoss,
        "true_labels_all_stacked": CustomLossFunction,
        "true_labels_single": torch.nn.BCEWithLogitsLoss,
        "true_probabilities": torch.nn.L1Loss,
    }
    criterion = LossWrapper(criterion_options[dataset_config["target"]]())

    model = torch.load(BEST_MODEL_PATH)
    model.to(device="cuda")

    # Run evaluation.
    print("\nEvaluation results:")
    print("===================")
    eval_results = evaluate(
        model,
        0,
        criterion,
        train_data_obj,
        val_data_obj,
        plot_graphs,
        make_table,
        suppress_output=False,
    )

    print("\nEvaluation results:")
    print("===================")
    eval_results = evaluate(
        model,
        0,
        criterion,
        train_data_obj,
        test_data_obj,
        plot_graphs,
        make_table,
        suppress_output=False,
    )


if __name__ == "__main__":

    config = {
        "selected_extra_feature": "",
    }

    main(config=config)
