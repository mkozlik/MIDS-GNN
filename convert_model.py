from pathlib import Path

import torch
from Utilities.gnn_models import GNNWrapper, custom_gnns, premade_gnns


LOAD_MODEL_PATH = Path(__file__).parents[0] / "Models" / "best_model.pth"
SAVE_MODEL_PATH = Path(__file__).parents[0] / "Models" / "prob_model.pth"


def generate_model(architecture, in_channels, hidden_channels, num_layers, out_channels=1, **kwargs):
    """Generate a Neural Network model based on the architecture and hyperparameters."""
    # GLOBALS: device, premade_gnns, custom_gnns
    if architecture in premade_gnns:
        model = GNNWrapper(
            gnn_model=premade_gnns[architecture],
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels ,
            **kwargs,
        )
    else:
        MyGNN = custom_gnns[architecture]
        model = MyGNN(in_channels, hidden_channels, num_layers, out_channels=1, **kwargs)

    model = model.to("cpu")
    return model

if __name__ == "__main__":
    config = {
        ## Model configuration
        "architecture": "GraphSAGE",
        "hidden_channels": 32,
        "gnn_layers": 3,
        "activation": "relu",
        "jk": "none",
        "dropout": 0.0,
        "feature_dim": 8,
    }

    model_kwargs = {}
    if config["architecture"] == "GIN":
        model_kwargs = {"train_eps": True}
    elif config["architecture"] == "GAT":
        model_kwargs = {"v2": True}

    model = generate_model(
        config["architecture"],
        config["feature_dim"],
        config["hidden_channels"],
        config["gnn_layers"],
        act=config["activation"],
        dropout=float(config["dropout"]),
        jk=config["jk"] if config["jk"] != "none" else None,
        **model_kwargs,
    )

    checkpoint = torch.load(LOAD_MODEL_PATH, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    torch.save(model, SAVE_MODEL_PATH)
