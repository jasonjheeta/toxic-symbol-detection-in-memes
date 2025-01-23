import pandas as pd
import json
import argparse


def merge_force_suffix(left, right, models, **kwargs):
    """
    Modification of pandas merge to force suffix.

    Args:
        left: The left column used for merge.
        right: The right column used for merge.
        models: All models that are used for modality.
    """
    # https://github.com/pandas-dev/pandas/issues/17834
    on_col = kwargs["on"]
    suffix_tupple = kwargs["suffixes"]

    def suffix_col(col, suffix):
        if col != on_col and not any(model in col for model in models):
            return str(col) + suffix
        else:
            return col

    left_suffixed = left.rename(
        columns=lambda x: suffix_col(x, suffix_tupple[0])
    )
    right_suffixed = right.rename(
        columns=lambda x: suffix_col(x, suffix_tupple[1])
    )
    del kwargs["suffixes"]
    return pd.merge(left_suffixed, right_suffixed, **kwargs)


def merge_dfs(dfs, models):
    """
    Merge dataframes using inner join on id.

    Args:
        dfs: Dataframes for all the files.
        models: All models that are used for modality.
    """
    data = dfs[0]

    for i in range(len(models) - 1):
        data = merge_force_suffix(
            data,
            dfs[i + 1],
            models,
            on="id",
            suffixes=(f"_{models[i]}", f"_{models[i + 1]}"),
        )

    data["label"] = data[f"label_{models[0]}"]
    data = data.drop(columns=[f"label_{model}" for model in models])

    data["count"] = data.groupby("label")["label"].transform("count")
    data = data.sort_values(["count", "label"], ascending=[False, True]).drop(
        columns=["count"]
    )

    return data


def error_analysis(models, class_label, modality):
    """
    Save error analysis for certain modality and class label.

    Args:
        models: All models that are used for modality.
        class_label: The class label used for error analysis.
        modality: The modality used for error analysis.
    """
    files = [
        f"{class_label}/{model}/stats_per_image_misclassified.csv"
        for model in models
    ]
    dfs = [pd.read_csv(file) for file in files]
    common_ids = merge_dfs(dfs, models)

    if class_label == "toxic-symbols":
        with open("OnToxMeme_dataset/OnToxMeme_dict.json") as f:
            symbol_descriptions = json.load(f)

        for model in models:
            common_ids[f"prediction_{model}"] = common_ids[
                f"prediction_{model}"
            ].map(lambda x: symbol_descriptions[x]["Title"])

        common_ids["label"] = common_ids["label"].map(
            lambda x: symbol_descriptions[x]["Title"]
        )

    common_ids = common_ids[
        ["id", "label"] + [f"prediction_{model}" for model in models]
    ]

    common_ids.to_csv(
        f"error_analysis/{class_label}/{modality}/results.csv", index=False
    )


def main():
    """
    Main function to run error analysis.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        choices=["text", "vision", "multimodal", "all"],
        default="all",
        help="Choose the modality: text, vision, multimodal or all",
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

    modality = args.modality

    for class_label in class_labels:
        error_analysis(modality_models[modality], class_label, modality)


if __name__ == "__main__":
    main()
