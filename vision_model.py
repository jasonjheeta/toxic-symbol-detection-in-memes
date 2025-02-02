import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torchvision.transforms import (
    RandomResizedCrop,
    Compose,
    Normalize,
    ToTensor,
)
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import evaluate
import argparse
from tqdm import tqdm
import pickle


def load_model(model_name, num_labels):
    """
    Load a processor and model for image classification.

    Args:
        model_name: The name of the pretrained model.
        num_labels: The number of labels for the classification task.

    Returns:
        A tuple containing the processor and the model.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return processor, model


def choose_dataset(num_labels):
    """
    Choose the appropriate dataset based on the number of labels.

    Args:
        num_labels: The number of labels.

    Returns:
        The correct dataset based on the number of labels.
    """
    if num_labels == 56:
        return pd.read_json("OnToxMeme_dataset/toxic_symbolism_entries.json")
    else:
        return pd.read_json("OnToxMeme_dataset/combined_entries.json")


def transform(examples, transform_compose, image_folder):
    """
    The transformation of the images using the image processor.

    Args:
        examples: The dataset.
        transform_compose: The fuction that transforms the images.
        image_folder: The folder containing the images.

    Returns:
        The dataset with the pixel values without the image column.
    """
    examples["pixel_values"] = [
        transform_compose(Image.open(image_folder + img).convert("RGB"))
        for img in examples["img"]
    ]
    del examples["img"]
    return examples


def collate_fn(examples):
    """
    This function creates tensors of the data in the dataset.

    Args:
        examples: The dataset.

    Returns:
        The data returned as a dictionary.
    """
    pixel_values = torch.stack(
        [example["pixel_values"] for example in examples]
    )
    labels = torch.tensor([example["labels"] for example in examples])
    ids = torch.tensor([example["id"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels, "id": ids}


def preprocess_dataset(dataset, num_labels, model, processor, class_label):
    """
    Preprocess the dataset based on the number of labels.

    Args:
        dataset: The dataset used for classification.
        num_labels: The number of labels.
        model: The model used for classification.
        processor: The processor used to transform image.
        class_label: The class corresponding to the task.

    Returns:
        A tuple containing the train and testdataset and a list of unique
        labels.
    """
    if num_labels == 2:
        binary_cases = {
            "binary-harmless-toxic": 1,
            "binary-harmless-unethical": 2,
            "binary-unethical-toxic": 0,
        }
        if class_label in binary_cases:
            exclude_label = binary_cases[class_label]
            dataset = dataset[
                dataset["labels"].apply(lambda x: x != [exclude_label])
            ]
        dataset.loc[:, "labels"] = dataset["labels"].apply(lambda x: x[0])
        labels = dataset["labels"].unique()
        id2label = {idx: label for idx, label in enumerate(labels)}
        label2id = {label: idx for idx, label in enumerate(labels)}
        model.config.id2label, model.config.label2id = id2label, label2id
        dataset.loc[:, "labels"] = dataset["labels"].map(model.config.label2id)
    elif num_labels == 3:
        dataset.loc[:, "labels"] = dataset["labels"].apply(lambda x: x[0])
        labels = dataset["labels"].unique()
        id2label = {idx: label for idx, label in enumerate(labels)}
        label2id = {label: idx for idx, label in enumerate(labels)}
        model.config.id2label, model.config.label2id = id2label, label2id
    elif num_labels == 56:
        unique_symbols = dataset["symbol_id"].unique()
        id2label = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
        label2id = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        model.config.id2label, model.config.label2id = id2label, label2id
        dataset.loc[:, "labels"] = dataset["symbol_id"].map(
            model.config.label2id
        )

    image_folder = "OnToxMeme_dataset/combined_images/"

    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    transform_compose = Compose(
        [RandomResizedCrop(size), ToTensor(), normalize]
    )

    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["labels"],
        random_state=45,
    )

    train_dataset = Dataset.from_pandas(train_dataset, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_dataset, preserve_index=False)

    train_dataset = train_dataset.remove_columns(["text", "symbol_id"])
    test_dataset = test_dataset.remove_columns(["text", "symbol_id"])

    train_dataset.set_transform(
        lambda examples: transform(examples, transform_compose, image_folder)
    )
    test_dataset.set_transform(
        lambda examples: transform(examples, transform_compose, image_folder)
    )

    return train_dataset, test_dataset, id2label


def select_device():
    """
    Select the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    """
    Create DataLoaders for train and test datasets.

    Args:
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        batch_size: The batch size for the dataloaders.

    Returns:
        A tuple containing the train_dataloader and test_dataloader.
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_dataloader, test_dataloader


def prepare_training_components(
    model, train_dataloader, learning_rate=5e-5, num_epochs=50
):
    """
    Prepare the optimizer and learning rate scheduler.

    Args:
        model: The model being trained.
        train_dataloader: The DataLoader for the training dataset.
        learning_rate: The learning rate for the optimizer.
        num_epochs: The number of training epochs.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler


def training_loop(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    lr_scheduler,
    device,
    architecture,
    class_label,
    id2label,
    num_epochs=100,
):
    """
    Perform the training loop and save variables for error analysis.

    Args:
        model: The model being trained.
        train_dataloader: The DataLoader for the training dataset.
        test_dataloader: The DataLoader for the testing dataset.
        optimizer: The optimizer for model parameters.
        lr_scheduler: The learning rate scheduler.
        device: The device on which computations are performed.
        architecture: The architecture of the pretrained model.
        class_label: The class corresponding to the task.
        id2label: The conversion between the ids to the labels.
        num_epochs: The number of training epochs.

    Returns:
        A dictionary containing all relevant variables for error analysis.
    """
    best_accuracy = 0
    patience = 20
    counter = 0
    training_losses = []
    learning_rates = []
    all_predictions = []
    all_references = []
    all_ids = []

    for _ in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        epoch_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        training_losses.append(epoch_loss / len(train_dataloader))
        learning_rates.append(optimizer.param_groups[0]["lr"])

        accuracy_metric = evaluate.load("accuracy")
        model.eval()
        epoch_predictions = []
        epoch_references = []
        epoch_ids = []

        for batch in test_dataloader:
            old_batch = batch
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits.cpu()
            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"].cpu()

            epoch_predictions.append(predictions)
            epoch_references.append(labels)
            epoch_ids.append(old_batch["id"])

            accuracy_metric.add_batch(
                predictions=predictions, references=labels
            )

        accuracy = accuracy_metric.compute()
        cur_accuracy = accuracy["accuracy"]
        all_predictions.append(epoch_predictions)
        all_references.append(epoch_references)
        all_ids.append(epoch_ids)

        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            counter = 0
            model.save_pretrained(f"{class_label}/{architecture}")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    analysis_data = {
        "id2label": id2label,
        "training_losses": training_losses,
        "learning_rates": learning_rates,
        "all_predictions": all_predictions,
        "all_references": all_references,
        "all_ids": all_ids,
    }

    with open(f"{class_label}/{architecture}/analysis_data.pkl", "wb") as f:
        pickle.dump(analysis_data, f)

    return


def main():
    """
    Main function to execute the training process.
    """
    parser = argparse.ArgumentParser()

    models = {
        "resnet": "microsoft/resnet-152",
        "vit": "google/vit-base-patch16-224",
    }

    models_list = list(models.keys())

    parser.add_argument(
        "architecture",
        choices=models_list,
        help=f"Choose a model: {', '.join(models_list)}",
    )

    class_labels2num_labels = {
        "binary-harmless-toxic": 2,
        "binary-harmless-unethical": 2,
        "binary-unethical-toxic": 2,
        "harmless-unethical-toxic": 3,
        "toxic-symbols": 56,
    }

    class_labels = list(class_labels2num_labels.keys())

    parser.add_argument(
        "class_label",
        choices=class_labels,
        help=f"Choose one of the class labels: {', '.join(class_labels)}",
    )

    args = parser.parse_args()

    model_name = models[args.architecture]

    num_labels = class_labels2num_labels[args.class_label]

    processor, model = load_model(model_name, num_labels)

    dataset = choose_dataset(num_labels)

    train_dataset, test_dataset, id2label = preprocess_dataset(
        dataset, num_labels, model, processor, args.class_label
    )

    device = select_device()
    model = model.to(device)

    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset, test_dataset
    )

    optimizer, lr_scheduler = prepare_training_components(
        model, train_dataloader
    )

    training_loop(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
        device,
        args.architecture,
        args.class_label,
        id2label,
    )


if __name__ == "__main__":
    main()
