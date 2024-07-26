import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json

MIN_RESIZE = 256
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def load_checkpoint(checkpoint_path):
    """
    Load a trained model from a checkpoint file.

    This function loads a saved checkpoint from the specified path and restores the
    model architecture, classifier, and state dictionary to
    use the model for inference.

    Parameters:
    checkpoint_path (str): The path to the checkpoint file.

    Returns:
    torch.nn.Module: The model loaded with the state dictionary and class-to-index mapping.
    """
    checkpoint = torch.load(checkpoint_path)

    # Create model
    model = models.vgg16(pretrained=True)

    # Create classifier
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"]),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint["hidden_layers"], checkpoint["output_size"]),
        nn.LogSoftmax(dim=1),
    )

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path):
    """
    Process an image file into a tensor suitable for a PyTorch model.

    This function takes the path to an image file, processes it by resizing,
    cropping, normalizing, and transposing the image, and returns the processed
    image as a NumPy array.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    np.ndarray: The processed image tensor with shape (3, 224, 224).
    """

    # Load the image
    image = Image.open(image_path)

    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    width, height = image.size

    if width > height:
        resize_height = MIN_RESIZE
        resize_width = int(round(MIN_RESIZE * (width / height)))
    else:
        resize_width = MIN_RESIZE
        resize_height = int(round(MIN_RESIZE * (height / width)))

    image = image.resize((resize_width, resize_height))

    # Crop out the center 224x224 portion of the image.
    left = (image.size[0] - CROP_SIZE) / 2
    top = (image.size[1] - CROP_SIZE) / 2
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE
    image = image.crop((left, top, right, bottom))

    # Convert color channels to 0-1
    image = np.array(image) / 255.0

    # Normalize the image
    image = (image - MEAN) / STD

    # Transpose the image
    image = image.transpose((2, 0, 1))

    return image


def predict(image_path, checkpoint, top_k, category_names, gpu):
    """
    Predict the class of an image using a trained deep learning model.

    This function loads a pre-trained model from a checkpoint, processes the input image,
    and predicts the top K most likely classes along with their associated probabilities.

    Parameters:
    image_path (str): The path to the image file to be predicted.
    checkpoint (str): The path to the checkpoint file containing the trained model.
    top_k (int): The number of top predicted classes to return.
    category_names (str): The path to a JSON file mapping category indices to class names.
    gpu (bool): Flag indicating whether to use GPU for prediction. If True and a GPU is available,
        the prediction will be performed on the GPU.

    Returns:
    None: Prints the top K predicted classes and their associated probabilities.
    """
    # Check GPU is available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(checkpoint)

    image = torch.tensor(process_image(image_path), dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.to(device)
    model = model.to(device)

    model.eval()

    # Run model to get log probabilities
    with torch.no_grad():
        logps = model.forward(image)

    # Calculate the probabilities
    ps = torch.exp(logps)

    # Get topk probabilities
    top_ps, top_idx = ps.topk(top_k, dim=1)

    # Invert the dictionary
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Calculate the classes
    top_classes = [idx_to_class[idx.item()] for idx in top_idx[0]]

    # Map classes
    try:
        with open(category_names, "r", encoding="utf-8") as f:
            cat_to_name = json.load(f)
            top_classes = [cat_to_name[cat] for cat in top_classes]
    except (FileNotFoundError, json.JSONDecodeError):
        print("Category names error!")

    for name, probability in zip(top_classes, top_ps.cpu().numpy()[0]):
        print(f"{probability:.4f} - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predict the image")
    parser.add_argument("image_path", type=str, help="Path to the image to predict")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="JSON file mapping class indices to names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()
    predict(
        args.image_path,
        args.checkpoint,
        args.top_k,
        args.category_names,
        args.gpu,
    )
