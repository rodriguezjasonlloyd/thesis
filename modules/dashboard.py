from pathlib import Path

import numpy
import pytorch_grad_cam
import torch
from gradio import (
    Blocks,
    Button,
    Checkbox,
    Dropdown,
    File,
    Image,
    Label,
    Markdown,
    Row,
)
from PIL import Image as PillowImage
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from torch import Tensor
from torch.nn import Module

from modules.data import get_class_names, get_data_root_path, transform_image_to_tensor
from modules.model import load_model
from modules.trainer import truncate


def predict_image(
    image: PillowImage.Image, uploaded_model: File, with_fsa: bool
) -> tuple[str, str]:
    if image is None:
        return ("No image provided", 0.0)

    if uploaded_model is None:
        return ("No model file uploaded", 0.0)

    try:
        model_path = Path(uploaded_model.name)
    except Exception as exception:
        print(f"Error reading uploaded file: {exception}")
        return ("Error reading uploaded file", 0.0)

    try:
        model = load_model(model_path, with_fsa=with_fsa)
    except Exception as exception:
        print(f"Model load error: {exception}")
        return ("Model load error", 0.0)

    try:
        tensor = transform_image_to_tensor(image)
    except Exception as exception:
        print(f"Image transform error: {exception}")
        return ("Image transform error", 0.0)

    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            output: Tensor = model(tensor.unsqueeze(0).to(device))
            probability = torch.sigmoid(output).cpu().item()
            prediction_index = int(probability > 0.5)

        classes = get_class_names(get_data_root_path())
        label = classes[prediction_index]
        predicted_class_probability = (
            probability if prediction_index == 1 else (1.0 - probability)
        )
        confidence = f"{truncate(predicted_class_probability * 100.0, 2)}%"

        return (label, confidence)
    except Exception as exception:
        print(f"Prediction error: {exception}")
        return ("Prediction error", 0.0)


def get_all_convolutional_layers(model: Module) -> list[tuple[str, str]]:
    convolutional_layers = []

    for name, module in model.named_modules():
        name: str
        module: Module

        if isinstance(module, torch.nn.Conv2d):
            display_name = name.replace(".", " > ")
            convolutional_layers.append((display_name, name))

    return convolutional_layers


def update_layer_choices(uploaded_model: File, with_fsa: bool):
    if uploaded_model is None:
        return Dropdown(choices=[], value=None)

    try:
        model_path = Path(uploaded_model.name)
        model = load_model(model_path, with_fsa=with_fsa)
        layers = get_all_convolutional_layers(model)
        default_value = layers[-1][1] if layers else None

        return Dropdown(choices=layers, value=default_value)
    except Exception as exception:
        print(f"Update layer error {exception}")
        return Dropdown(choices=[], value=None)


def generate_gradcam(image, uploaded_model, with_fsa, layer_name):
    if image is None or uploaded_model is None or layer_name is None:
        return None

    image = image.resize((224, 224))

    try:
        model_path = Path(uploaded_model.name)
        model = load_model(model_path, with_fsa=with_fsa)
        rgb_image = numpy.float32(image) / 255
        input_tensor = transform_image_to_tensor(image).unsqueeze(0)

        target_layer = model.get_submodule(layer_name)
        target_layers = [target_layer]

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            visualization = pytorch_grad_cam.utils.image.show_cam_on_image(
                rgb_image, cam(input_tensor=input_tensor)[0, :], use_rgb=True
            )

        return visualization
    except Exception as exception:
        print(f"Grad-CAM error: {exception}")
        return None


def make_dashboard() -> Blocks:
    with Blocks() as dashboard:
        Markdown(
            "# ConvNext V2 Model with Focal Self-Attention and Grad-CAM++ for PCOM Classification using Ultrasound Images"
        )

        Markdown("Note: The uploaded model architecture should match the FSA checkbox.")
        fsa_checkbox = Checkbox(label="Use Focal Self-Attention (FSA)")
        upload_model = File(label="Upload model (.pt)")

        image_input = Image(type="pil", label="Upload image", height=224)
        predict_button = Button("Predict")

        with Row():
            predicted_label = Label(label="Predicted label")
            predicted_confidence = Label(label="Confidence")

        layer_selector = Dropdown(label="Target Layer", choices=[], value=None)
        show_gradcam_button = Button("Show Grad-CAM++")
        gradcam_output = Image(label="Grad-CAM++ Visualization")

        upload_model.change(
            fn=update_layer_choices,
            inputs=[upload_model, fsa_checkbox],
            outputs=[layer_selector],
        )

        predict_button.click(
            fn=predict_image,
            inputs=[image_input, upload_model, fsa_checkbox],
            outputs=[predicted_label, predicted_confidence],
        )

        show_gradcam_button.click(
            fn=generate_gradcam,
            inputs=[image_input, upload_model, fsa_checkbox, layer_selector],
            outputs=gradcam_output,
        )

    return dashboard
