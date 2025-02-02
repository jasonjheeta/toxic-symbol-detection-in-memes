import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPTextModel,
)
from torch.utils.data import DataLoader, Dataset, Subset
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
import torch.nn as nn
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndNoAttention,
)


class ResNetModel(nn.Module):
    """
    This is a custom PyTorch module that adapts a pre-trained ResNet model for
    compatibility with CLIP's architecture.

    Attributes:
        resnet: The pre-trained ResNet model.
        config: A CLIPVisionConfig adjusted to support ResNet.
    """

    def __init__(self, model):
        super().__init__()
        self.resnet = model
        self.config = CLIPVisionConfig(hidden_size=2048, num_hidden_layers=0)

    def forward(self, pixel_values, *args, **kwargs):
        kwargs.pop("output_attentions", None)
        outputs = self.resnet(pixel_values=pixel_values, *args, **kwargs)

        last_hidden_state = outputs.last_hidden_state

        # https://github.com/huggingface/transformers/issues/22366#issuecomment-1483792271
        batch_size, num_channels, height, width = last_hidden_state.shape
        last_hidden_state = last_hidden_state.permute(0, 2, 3, 1)
        last_hidden_state = last_hidden_state.reshape(
            batch_size, height * width, num_channels
        )

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=outputs.pooler_output.reshape(
                batch_size, num_channels
            ),
            hidden_states=outputs.hidden_states,
        )


class ForMultiModalClassification(nn.Module):
    """
    This is a custom PyTorch module for multimodal classification combining
    text and image modalities.

    Attributes:
        model_name: The name of the pretrained model.
        text_model: The text encoder model.
        vision_model: The vision encoder model.
        projection_dim: The dimensionality for the projection layers.
        text_map: The mapping layer to project text features to shared space.
        image_map: The mapping layer to project image features to shared space.
        classifier: The final classification head combining text and image.

    """

    def __init__(self, model_name, class_label, num_labels):
        super().__init__()
        if model_name == "clip":
            self.text_model = CLIPTextModel.from_pretrained(
                f"{model_name['text']['pretrained_model']}"
            )
            self.vision_model = CLIPVisionModel.from_pretrained(
                f"{model_name['vision']['pretrained_model']}"
            )
        elif model_name["vision"]["architecture"] == "resnet":
            self.text_model = AutoModel.from_pretrained(
                f"{model_name['text']['pretrained_model']}"
            )
            self.vision_model = ResNetModel(
                AutoModel.from_pretrained(
                    f"{model_name['vision']['pretrained_model']}"
                )
            )
        else:
            self.text_model = AutoModel.from_pretrained(
                f"{model_name['text']['pretrained_model']}"
            )
            self.vision_model = AutoModel.from_pretrained(
                f"{model_name['vision']['pretrained_model']}"
            )
        self.model_name = model_name

        if class_label == "toxic-symbols":
            self.projection_dim = 512
        else:
            self.projection_dim = 256

        if model_name["text"]["architecture"] == "distilbert":
            text_hidden_size = self.text_model.config.dim
            self.pre_classifier_distilbert = nn.Linear(
                text_hidden_size, text_hidden_size
            )
            self.relu_distilbert = nn.ReLU()
            self.dropout_distilbert = nn.Dropout(
                self.text_model.config.seq_classif_dropout
            )
        elif model_name["text"]["architecture"] == "clip":
            text_hidden_size = self.text_model.config.text_config.hidden_size
        else:
            text_hidden_size = self.text_model.config.hidden_size

        self.text_map = nn.Sequential(
            nn.Linear(
                text_hidden_size,
                self.projection_dim,
            ),
            nn.Dropout(0.2),
        )

        if model_name["vision"]["architecture"] == "clip":
            vision_hidden_size = (
                self.vision_model.config.vision_config.hidden_size
            )
        else:
            vision_hidden_size = self.vision_model.config.hidden_size

        self.image_map = nn.Sequential(
            nn.Linear(
                vision_hidden_size,
                self.projection_dim,
            ),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim * 2, num_labels), nn.Dropout(0.2)
        )

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        labels=None,
    ):

        if self.model_name["text"]["architecture"] == "clip":
            text_outputs = self.text_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=not return_dict,
            )
            vision_outputs = self.vision_model.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=not return_dict,
            )
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if (
            self.model_name["text"]["architecture"] == "roberta"
            or self.model_name["text"]["architecture"] == "clip"
        ):
            pooler_output = text_outputs[1]
        elif self.model_name["text"]["architecture"] == "distilbert":
            hidden_state = text_outputs[0]
            pooler_output = hidden_state[:, 0]
            pooler_output = self.pre_classifier_distilbert(pooler_output)
            pooler_output = self.relu_distilbert(pooler_output)
            pooler_output = self.dropout_distilbert(pooler_output)

        image_embeds = self.image_map(vision_outputs[1])
        text_embeds = self.text_map(pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        combined_embeds = torch.cat((text_embeds, image_embeds), 1)
        logits = self.classifier(combined_embeds)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class CustomDataset(Dataset):
    """
    This is a custom PyTorch Dataset class to handle multimodal data.

    Attributes:
        df: The dataset packaged as DataFrame.
        tokenizer: The tokenizer used for classification.
        transform_compose: The fuction that transforms the images.
        image_folder: The folder containing the images.
    """

    def __init__(self, df, tokenizer, transform_compose, image_folder):
        self.df = df
        self.tokenizer = tokenizer
        self.transform_compose = transform_compose
        self.image_folder = image_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        image_id = self.df.iloc[idx]["img"]
        tokens = self.tokenizer(
            text, padding="max_length", truncation=True, return_tensors="pt"
        )

        label = torch.tensor(self.df.iloc[idx]["labels"], dtype=torch.long)
        pixel_values = self.transform_compose(
            Image.open(self.image_folder + image_id).convert("RGB")
        )

        data = {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": label,
            "id": self.df.iloc[idx]["id"],
        }

        return data


def load_model(model_name, class_label, num_labels):
    """
    Load a image_processor, processor and model for multimodal classification.

    Args:
        model_name: The name of the pretrained model.
        class_label: The class corresponding to the task.
        num_labels: The number of labels for the classification task.

    Returns:
        A tuple containing the image_processor, processor and the model.
    """
    model = ForMultiModalClassification(model_name, class_label, num_labels)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name["text"]["pretrained_model"]
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name["vision"]["pretrained_model"]
    )
    return tokenizer, image_processor, model


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


def preprocess_dataset(
    dataset, num_labels, model, tokenizer, image_processor, class_label
):
    """
    Preprocess the dataset based on the number of labels.

    Args:
        dataset: The dataset used for classification.
        num_labels: The number of labels.
        model: The model used for classification.
        tokenizer: The tokenizer used for classification.
        image_processor: The image processor used to transform image.
        class_label: The class corresponding to the task.

    Returns:
        A tuple containing the dataset and a list of unique labels.
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
        dataset.loc[:, "labels"] = dataset["labels"].map(label2id)
    elif num_labels == 3:
        dataset.loc[:, "labels"] = dataset["labels"].apply(lambda x: x[0])
        labels = dataset["labels"].unique()
        id2label = {idx: label for idx, label in enumerate(labels)}
    elif num_labels == 56:
        unique_symbols = dataset["symbol_id"].unique()
        id2label = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
        label2id = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        dataset.loc[:, "labels"] = dataset["symbol_id"].map(label2id)

    image_folder = "OnToxMeme_dataset/combined_images/"

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    transform_compose = Compose(
        [RandomResizedCrop(size), ToTensor(), normalize]
    )

    labels = dataset["labels"]
    dataset = CustomDataset(
        dataset, tokenizer, transform_compose, image_folder
    )

    return dataset, id2label, labels


def select_device():
    """
    Select the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset(dataset, labels):
    """
    Split the dataset into train and test subsets.

    Args:
        dataset: The tokenized dataset.
        labels: The different labels of the classes.

    Returns:
        A tuple containing the train_dataset and test_dataset.
    """
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=45,
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=16):
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
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
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
            torch.save(
                model.state_dict(),
                f"{class_label}/{architecture}/best_model.pt",
            )
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
        "resnet-distilbert": {
            "vision": {
                "architecture": "resnet",
                "pretrained_model": "microsoft/resnet-152",
            },
            "text": {
                "architecture": "distilbert",
                "pretrained_model": "distilbert-base-uncased",
            },
        },
        "resnet-roberta": {
            "vision": {
                "architecture": "resnet",
                "pretrained_model": "microsoft/resnet-152",
            },
            "text": {
                "architecture": "roberta",
                "pretrained_model": "roberta-base",
            },
        },
        "vit-distilbert": {
            "vision": {
                "architecture": "vit",
                "pretrained_model": "google/vit-base-patch16-224",
            },
            "text": {
                "architecture": "distilbert",
                "pretrained_model": "distilbert-base-uncased",
            },
        },
        "vit-roberta": {
            "vision": {
                "architecture": "vit",
                "pretrained_model": "google/vit-base-patch16-224",
            },
            "text": {
                "architecture": "roberta",
                "pretrained_model": "roberta-base",
            },
        },
        "clip": {
            "vision": {
                "architecture": "clip",
                "pretrained_model": "openai/clip-vit-base-patch16",
            },
            "text": {
                "architecture": "clip",
                "pretrained_model": "openai/clip-vit-base-patch16",
            },
        },
    }

    models_list = list(models.keys())

    parser.add_argument(
        "model",
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

    classes_labels = list(class_labels2num_labels.keys())

    parser.add_argument(
        "class_label",
        choices=classes_labels,
        help=f"Choose one of the class labels: {', '.join(classes_labels)}",
    )

    args = parser.parse_args()

    model_name = models[args.model]

    num_labels = class_labels2num_labels[args.class_label]

    tokenizer, image_processor, model = load_model(
        model_name, args.class_label, num_labels
    )

    dataset = choose_dataset(num_labels)

    dataset, id2label, labels = preprocess_dataset(
        dataset,
        num_labels,
        model,
        tokenizer,
        image_processor,
        args.class_label,
    )

    train_dataset, test_dataset = split_dataset(dataset, labels)

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
        args.model,
        args.class_label,
        id2label,
    )


if __name__ == "__main__":
    main()
