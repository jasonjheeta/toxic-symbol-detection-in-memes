import pickle
import argparse
import evaluate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import csv
from math import floor


def load_data(model, class_label):
    """
    Loads the data from the given model and class label.

    Args:
        model: The model used to determine the directory.
        class_label: The label of the class used to determine the directory.
    Returns:
        The data that is connected to the correct class label and model.
    """
    with open(f"{class_label}/{model}/analysis_data.pkl", "rb") as f:
        data = pickle.load(f)
        return data


def metrics_per_epoch(data, model, class_label):
    """
    Calculates the metric for every epoch.

    Args:
        data: The data obtained during training.
        model: The model used to determine the directory.
        class_label: The label of the class used to determine the directory.
    Returns:
        A tuple containing the best epoch and epoch accuracies.
    """
    accuracy_metric = evaluate.load("accuracy")
    combined_metric = evaluate.combine(["precision", "recall", "f1"])

    best_epoch = 0
    highest_accuracy = 0
    epoch_accuracies = []

    with open(
        f"{class_label}/{model}/metrics_per_epoch.csv", "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["accuracy", "precision", "recall", "f1"]
        )
        writer.writeheader()
        for epoch in range(len(data["all_predictions"])):
            for predictions, references in zip(
                data["all_predictions"][epoch],
                data["all_references"][epoch],
            ):
                accuracy_metric.add_batch(
                    predictions=predictions, references=references
                )
                combined_metric.add_batch(
                    predictions=predictions, references=references
                )

            accuracy = accuracy_metric.compute()
            combined = combined_metric.compute(average="weighted")
            epoch_accuracies.append(accuracy["accuracy"])
            writer.writerow(accuracy | combined)

            if accuracy["accuracy"] > highest_accuracy:
                highest_accuracy = accuracy["accuracy"]
                best_epoch = epoch

        for predictions, references in zip(
            data["all_predictions"][best_epoch],
            data["all_references"][best_epoch],
        ):

            accuracy_metric.add_batch(
                predictions=predictions, references=references
            )
            combined_metric.add_batch(
                predictions=predictions, references=references
            )

        accuracy = accuracy_metric.compute()
        combined = combined_metric.compute(average="weighted")
        metrics = {
            "accuracy": f"{floor(accuracy['accuracy'] * 100) / 100:.2f}",
            "precision": f"{floor(combined['precision'] * 100) / 100:.2f}",
            "recall": f"{floor(combined['recall'] * 100) / 100:.2f}",
            "f1": f"{floor(combined['f1'] * 100) / 100:.2f}",
        }
        writer.writerow(metrics)

    return best_epoch, epoch_accuracies


def validation_accuracy(epoch_accuracies, model, class_label):
    """
    Saves a figure containing the accuracy on the validation set.

    Args:
        epoch_accuracies: The accuracies for every epoch.
        model: The model used to determine the directory.
        class_label: The label of the class used to determine the directory.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(epoch_accuracies) + 1),
        epoch_accuracies,
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{class_label}/{model}/validation_accuracy_plot.png")


def confusion_matrix(data, best_epoch, model, class_label):
    """
    Saves a figure containing the confusion matrix of the best epoch.

    Args:
        data: The data obtained during training.
        best_epoch: The epoch with the highest accuracy.
        model: The model used to determine the directory.
        class_label: The label of the class used to determine the directory.
    """
    confusion_metric = evaluate.load("confusion_matrix")

    for predictions, references in zip(
        data["all_predictions"][best_epoch], data["all_references"][best_epoch]
    ):
        confusion_metric.add_batch(
            predictions=predictions, references=references
        )

    confusion_matrix = confusion_metric.compute(normalize="all")
    labels = list(data["id2label"].values())
    confusion_matrix = pd.DataFrame(
        confusion_matrix["confusion_matrix"], labels, labels
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"{class_label}/{model}/confusion_matrix.png")


def stats_per_image(data, best_epoch, model, class_label):
    """
    Saves the prediction and ground truth for every classification.

    Args:
        data: The data obtained during training.
        best_epoch: The epoch with the highest accuracy.
        model: The model used to determine the directory.
        class_label: The label of the class used to determine the directory.
    """
    with open(
        f"{class_label}/{model}/stats_per_image.csv", "w", newline=""
    ) as f, open(
        f"{class_label}/{model}/stats_per_image_misclassified.csv",
        "w",
        newline="",
    ) as f2:
        writer = csv.writer(f)
        writer2 = csv.writer(f2)
        writer.writerow(["id", "label", "prediction"])
        writer2.writerow(["id", "label", "prediction"])

        rows = []
        misclassified_rows = []

        for image_ids, references, predictions in zip(
            data["all_ids"][best_epoch],
            data["all_references"][best_epoch],
            data["all_predictions"][best_epoch],
        ):
            for image_id, reference, prediction in zip(
                image_ids, references, predictions
            ):
                image_id_value = image_id.item()
                reference_label = data["id2label"][reference.item()]
                prediction_label = data["id2label"][prediction.item()]

                rows.append(
                    [image_id_value, reference_label, prediction_label]
                )
                if prediction_label != reference_label:
                    misclassified_rows.append(
                        [image_id_value, reference_label, prediction_label]
                    )

        rows.sort(key=lambda x: x[1])
        misclassified_rows.sort(key=lambda x: x[1])

        writer.writerows(rows)
        writer2.writerows(misclassified_rows)


def main():
    """
    Main function to analyse the training process.
    """
    parser = argparse.ArgumentParser()

    models = [
        "distilbert",
        "roberta",
        "resnet",
        "vit",
        "resnet-distilbert",
        "resnet-roberta",
        "vit-distilbert",
        "vit-roberta",
        "clip",
    ]

    parser.add_argument(
        "model",
        choices=models,
        help=f"Choose a model: {', '.join(models)}",
    )

    class_label = [
        "binary-harmless-toxic",
        "binary-harmless-unethical",
        "binary-unethical-toxic",
        "harmless-unethical-toxic",
        "toxic-symbols",
    ]

    parser.add_argument(
        "class_label",
        choices=class_label,
        help=f"Choose one of the class_label: {', '.join(class_label)}",
    )

    args = parser.parse_args()
    model, class_label = args.model, args.class_label

    data = load_data(model, class_label)

    best_epoch, epoch_accuracies = metrics_per_epoch(data, model, class_label)

    validation_accuracy(epoch_accuracies, model, class_label)

    if class_label != "toxic-symbols":
        confusion_matrix(data, best_epoch, model, class_label)

    stats_per_image(data, best_epoch, model, class_label)


if __name__ == "__main__":
    main()
