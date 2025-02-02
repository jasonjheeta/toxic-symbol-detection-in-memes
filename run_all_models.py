import argparse
from subprocess import call


def run_scripts(modality, models, class_labels, mode):
    """
    Run the correct scripts based on given parameters using subprocess call.

    Args:
        modality: The modality to determine correct file to run.
        models: A list of models.
        class_labels: A list of class labels.
        mode: Determines if only training, only analysis or both is needed.
    """
    # for model in models:
    #     for class_label in class_labels:
    #         if mode == "analyse":
    #             call(["python", "analyse_data.py", model, class_label])
    #         elif mode == "training":
    #             call(["python", f"{modality}_model.py", model, class_label])
    #         elif mode == "all":
    #             call(["python", f"{modality}_model.py", model, class_label])
    #             call(["python", "analyse_data.py", model, class_label])
    if mode == "analyse" or mode == "all":
        call(["python", "error_analysis.py", "--modality", modality])


def main():
    """
    Main function to run all models.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        choices=["text", "vision", "multimodal", "all"],
        default="all",
        help="Choose the modality: text, vision, multimodal or all",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "analyse", "all"],
        default="all",
        help="Choose the mode: train, analyse or all",
    )

    args = parser.parse_args()

    class_labels = [
        "binary-harmless-toxic",
        "binary-harmless-unethical",
        "binary-unethical-toxic",
        "harmless-unethical-toxic",
        "toxic-symbols",
    ]

    modality_models = {
        "text": ["distilbert", "roberta"],
        "vision": ["resnet", "vit"],
        "multimodal": [
            "resnet-distilbert",
            "resnet-roberta",
            "vit-distilbert",
            "vit-roberta",
            "clip",
        ],
    }

    if args.modality in modality_models:
        models = modality_models[args.modality]
        run_scripts(args.modality, models, class_labels, args.mode)
    elif args.modality == "all":
        for modality, models in modality_models.items():
            run_scripts(modality, models, class_labels, args.mode)


if __name__ == "__main__":
    main()
