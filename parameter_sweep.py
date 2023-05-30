import ta3n_training_script

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

# 2: Define the search space
hyperparam_search_config = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'best_acc'
    },
    'parameters':
    {
        'epochs': {
            "values": [30]
        },
        'dataset': {
            "values": [
                {"shift": "D1-D3"},
                {"shift": "D2-D3"},
                {"shift": "D3-D2"},
            ]
        },
        "fc_dim": {
            "values": [1024]
        },
        "lr": {
            "values": [6e-2]
        },
        "lr-adaptive": {
            "values": ["dann"]
        },
        "weight_decay": {
            "values": [1e-4]
        }
    }
}

dataset_shift_configuration = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'best_acc'
    },
    'parameters':
    {

        "frame_aggregation": {
            "values": [
                "avgpool",
                "trn-m"
            ]
        },
        "place_adv": {
            "values": [
                ["N", "N", "N"]
                # ["N", "N", "Y"],
                # ["N", "Y", "N"],
                # ["Y", "N", "N"],
                # ["Y", "Y", "Y"],
            ]
        },
        'dataset.shift': {
            "values": [
                "D1-D2",
                "D1-D3",
                "D2-D1",
                "D2-D3",
                "D3-D1",
                "D3-D2",
            ]
        },
        "use_attn": {
            "values": [
                "none",
                # "TransAttn"
            ],
        },
        "add_loss_DA": {
            "values": [
                "none",
                # "attentive_entropy",
            ]
        },
        'epochs': {
            "values": [30]
        },
        "lr": {
            "values": [3e-2]
        },
        "train.loss_weights": {
            "values": [
                [1, 1, 1, 1, 0.003],  # Original
                # [1, 0.5, 0.5, 0.5, 0.0015],
                # [1, 1, 1, 1, 0.03],
                # [0.5, 1, 1, 1, 0.0015],
                # [0.5, 1, 1, 1, 0.003],
                # [1, 0.5, 0.5, 0.5, 0.03]
            ]
        }

    }
}

if __name__ == "__main__":
    if False:
        ta3n_training_script.main()
    else:
        # 3: Start the sweep
        sweep_id = wandb.sweep(
            sweep=dataset_shift_configuration,
            project='mldl23-ego-ta3n',
        )
        wandb.agent(sweep_id, function=ta3n_training_script.main)
