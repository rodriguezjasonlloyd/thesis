from pathlib import Path

import torch
import torchcam
from gradio import (
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    File,
    Image,
    Label,
    Markdown,
    Radio,
    Row,
)
from gradio.utils import NamedString
from PIL import Image as PillowImage
from torch import Tensor
from torchcam.methods import GradCAMpp
from torchvision import transforms

from modules import data, model, utilities
from modules.preprocessing import PreprocessingMode


def predict_image(
    uploaded_model: NamedString | None,
    uploaded_image: PillowImage.Image | None,
    architecture: str,
    with_fsa: bool,
    preprocessing: str,
) -> tuple[str, str]:
    if uploaded_model is None:
        return ("No model uploaded", "")

    if uploaded_image is None:
        return ("No image uploaded", "")

    try:
        loaded_model = model.load_model(
            Path(uploaded_model.name), architecture=architecture, with_fsa=with_fsa
        )
    except Exception:
        return ("Something went wrong with loading the model", "")

    image_tensor = utilities.image_to_tensor(
        uploaded_image, preprocessing=PreprocessingMode(preprocessing)
    )

    try:
        with torch.no_grad():
            device = next(loaded_model.parameters()).device
            output: Tensor = loaded_model(image_tensor.unsqueeze(0).to(device))
            probability = torch.sigmoid(output).cpu().item()
            prediction_index = int(probability > 0.5)

        predicted_class_probability = (
            probability if prediction_index == 1 else (1.0 - probability)
        )

        return (
            data.get_class_names(data.get_data_root_path())[prediction_index],
            f"{utilities.truncate(predicted_class_probability * 100.0, 2)}%",
        )
    except Exception:
        return ("Something went wrong with prediction", "")


def update_layer_choices(
    uploaded_model: NamedString | None,
    architecture: str,
    with_fsa: bool,
) -> Dropdown:
    if uploaded_model is None:
        choice = "Upload a model first"
        return Dropdown(choices=[choice], value=choice, interactive=False)

    try:
        loaded_model = model.load_model(
            Path(uploaded_model.name), architecture=architecture, with_fsa=with_fsa
        )
        layers = model.get_all_convolutional_layers(loaded_model)
        default_value = layers[-1][1] if layers else None

        return Dropdown(choices=layers, value=default_value, interactive=True)
    except Exception:
        choice = "Something went wrong with updating layer choices"
        return Dropdown(choices=[choice], value=choice, interactive=False)


def generate_cam(
    uploaded_model: NamedString,
    uploaded_image: PillowImage.Image,
    architecture: str,
    with_fsa: bool,
    layer_name: str,
    preprocessing: str,
) -> PillowImage.Image | None:
    try:
        loaded_model = model.load_model(
            Path(uploaded_model.name), architecture=architecture, with_fsa=with_fsa
        )
        cam_extractor = GradCAMpp(loaded_model, target_layer=layer_name)
        device = next(loaded_model.parameters()).device
        output = loaded_model(
            utilities.image_to_tensor(
                uploaded_image, preprocessing=PreprocessingMode(preprocessing)
            )
            .unsqueeze(0)
            .to(device)
        )

        class_index = int(torch.sigmoid(output).cpu().item() > 0.5)

        if class_index == 0:
            activation_map = cam_extractor(class_index, output)
            heatmap = transforms.functional.to_pil_image(
                activation_map[0].squeeze(0), mode="F"
            )
            result = torchcam.utils.overlay_mask(uploaded_image, heatmap, alpha=0.5)
        else:
            result = uploaded_image

        cam_extractor.clear_hooks()

        return result
    except Exception:
        return None


def make_dashboard() -> Blocks:
    with Blocks() as dashboard:
        Markdown("# ConvNext V2 with Grad-CAM++ Dashboard for PCOM Classification")

        with Column():
            preprocessing_dropdown = Dropdown(
                label="Preprocessing Mode",
                choices=[
                    ("None", "none"),
                    ("CLAHE", "clahe"),
                    ("Otsu Threshold", "otsu_threshold"),
                    ("Deep Contrast", "deep_contrast"),
                    ("All (Aggressive Composite)", "all"),
                ],
                value="none",
            )

        with Column():
            architecture_radio = Radio(
                label="Architecture",
                choices=[("Base CNN", "base"), ("ConvNeXt V2", "convnext")],
                value="convnext",
            )

        with Column():
            fsa_checkbox = Checkbox(label="Use Focal Self-Attention (FSA)")

        upload_model = File(label="Upload Model", file_types=[".pt"])
        upload_image = Image(type="pil", label="Upload Image", height=300)
        predict_button = Button("Predict")

        with Row():
            predicted_label = Label(label="Predicted Label")
            predicted_confidence = Label(label="Confidence")

        layer_selector = Dropdown(
            label="Target Layer",
            choices=["Upload a model first"],
            value="Upload a model first",
            interactive=False,
        )
        show_cam_button = Button("Show Grad-CAM++")
        cam_output = Image(label="Grad-CAM++ Visualization", height=300)

        architecture_radio.change(
            fn=update_layer_choices,
            inputs=[upload_model, architecture_radio, fsa_checkbox],
            outputs=[layer_selector],
        )

        upload_model.change(
            fn=update_layer_choices,
            inputs=[upload_model, architecture_radio, fsa_checkbox],
            outputs=[layer_selector],
        )

        fsa_checkbox.change(
            fn=update_layer_choices,
            inputs=[upload_model, architecture_radio, fsa_checkbox],
            outputs=[layer_selector],
        )

        predict_button.click(
            fn=predict_image,
            inputs=[
                upload_model,
                upload_image,
                architecture_radio,
                fsa_checkbox,
                preprocessing_dropdown,
            ],
            outputs=[predicted_label, predicted_confidence],
        )

        show_cam_button.click(
            fn=generate_cam,
            inputs=[
                upload_model,
                upload_image,
                architecture_radio,
                fsa_checkbox,
                layer_selector,
                preprocessing_dropdown,
            ],
            outputs=cam_output,
        )

    return dashboard
