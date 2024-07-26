import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import os


def create_data_loaders(data_dir):
    """
    Create data loaders for training and validation datasets.

    This function takes a directory path containing the training and validation image data,
    applies appropriate transformations to the images, and then creates data loaders for
    both datasets. The training dataset is augmented with random resizing and horizontal
    flipping, while the validation dataset is resized and center-cropped.

    Parameters:
    data_dir (str): The directory containing 'train' and 'valid' subdirectories, which
                    should include image data for training and validation respectively.

    Returns:
    tuple:
        - train_loader (DataLoader): DataLoader object for the training dataset.
        - valid_loader (DataLoader): DataLoader object for the validation dataset.
        - class_to_idx (dict): A dictionary mapping class names to indices.
    """
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"

    # Define transforms for the training, validation sets
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, train_dataset.class_to_idx


def create_model(arch, hidden_units):
    """
    Create a model with a specified architecture and hidden units.

    This function initializes a pre-trained model based on the specified architecture,
    replaces its classifier with a new feed-forward network, and prepares it for transfer
    learning. The model's parameters are frozen to prevent backpropagation through them.
    The number of output units in the classifier is determined by the number of categories
    in the `cat_to_name.json` file.

    Parameters:
    arch (str): The architecture of the pre-trained model to use. Must be either 'vgg13' or 'vgg16'.
    hidden_units (int): The number of hidden units in the new classifier layer.

    Returns:
    tuple:
        - model (torch.nn.Module): The modified pre-trained model with a new classifier.
        - output_size (int): The number of output units in the classifier, 
            corresponding to the number of categories.
    """
    try:
        with open("cat_to_name.json", "r", encoding="utf-8") as f:
            cat_to_name = json.load(f)
            output_size = len(cat_to_name)
    except (FileNotFoundError, json.JSONDecodeError):
        output_size = 102

    # Load pre-trained model
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    # Freeze the parameters to not backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace model's classifier with a new feed-forward network
    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1),
    )

    return model, output_size


def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    """
    Train a deep learning model on a dataset and save the trained model as a checkpoint.

    This function sets up data loaders for training and validation datasets, initializes a 
    model with the specified architecture, and trains the model using the provided hyperparameters.
    It evaluates the model on the validation set after each epoch and prints the training and 
    validation losses and accuracy. The trained model is then saved as a checkpoint.

    Parameters:
    data_dir (str): Directory path containing 'train' and 'valid' subdirectories with images for 
        training and validation.
    save_dir (str): Directory path where the checkpoint will be saved.
    arch (str): The architecture of the pre-trained model to use. Options are 'vgg13' or 'vgg16'.
    learning_rate (float): Learning rate for the optimizer.
    hidden_units (int): Number of hidden units in the new classifier layer.
    epochs (int): Number of epochs to train the model.
    gpu (bool): Flag indicating whether to use GPU for training. If True and a GPU is available, 
        the model will be trained on the GPU.

    Returns:
    None
    """
    # Create training and validaiton data loaders
    train_loader, valid_loader, class_to_idx = create_data_loaders(data_dir)

    # Create model
    model, output_size = create_model(arch, hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Check GPU is available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the network
    train_loss = 0
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            # Move to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate validation loss
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)

                valid_loss += criterion(logps, labels).item()

                # Calculate accuracy
                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))

        # Print training and validation loss
        print(
            f"Epoch {epoch+1}/{epochs}"
            f"Train loss: {train_loss/len(train_loader):.3f}"
            f"Valid loss: {valid_loss/len(valid_loader):.3f}"
            f"Valid accuracy: {accuracy/len(valid_loader):.3f}"
        )
        train_loss = 0
        model.train()

    # Save the checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        "input_size": model.classifier[0].in_features,
        "output_size": output_size,
        "hidden_layers": hidden_units,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }

    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the flower classifier")
    parser.add_argument("data_dir", type=str, help="Directory of data")
    parser.add_argument(
        "--save_dir", type=str, default=".", help="Directory to save checkpoint"
    )
    parser.add_argument(
        "--arch", type=str, default="vgg16", help="Architecture vgg13 or vgg16"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=1024, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU to train")

    args = parser.parse_args()
    train_model(
        args.data_dir,
        args.save_dir,
        args.arch,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
        args.gpu,
    )
