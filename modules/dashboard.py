from pathlib import Path

from gradio import (
    Blocks,
    Button,
    Checkbox,
    Dropdown,
    File,
    Image,
    Markdown,
    Number,
)
from gradio.components.label import Label
from gradio.layouts.row import Row
from numpy import float32, ndarray
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import no_grad
from torch.nn.functional import softmax

from modules.data import get_class_names, get_data_root_path, transform_image_to_tensor
from modules.model import load_model


def predict_image(
    image: Image, uploaded_model: File, with_fsa: bool
) -> tuple[str, float]:
    if image is None:
        return ("No image provided", 0.0, None)

    if uploaded_model is None:
        return ("No model file uploaded", 0.0, None)

    try:
        model_path = Path(uploaded_model.name)
    except Exception as exception:
        print(f"Error reading uploaded file: {exception}")
        return ("Error reading uploaded file", 0.0, None)

    try:
        model = load_model(model_path, with_fsa=with_fsa)
    except Exception as exception:
        print(f"Model load error: {exception}")
        return ("Model load error", 0.0, None)

    try:
        tensor = transform_image_to_tensor(image)
    except Exception as exception:
        print(f"Image transform error: {exception}")
        return ("Image transform error", 0.0)

    try:
        with no_grad():
            device = next(model.parameters()).device
            output = model(tensor.unsqueeze(0).to(device))
            probabilities = softmax(output, dim=1).cpu().numpy()[0]
            prediction_index = int(probabilities.argmax())

        classes = get_class_names(get_data_root_path())
        label = classes[prediction_index]
        confidence = float(probabilities[prediction_index])

        return (label, confidence)
    except Exception as exception:
        print(f"Prediction error: {exception}")
        return ("Prediction error", 0.0)


def generate_gradcam(
    image: Image, uploaded_model: File, with_fsa: bool, target_layer_index: int
) -> ndarray:
    if image is None or uploaded_model is None:
        return None

    try:
        model_path = Path(uploaded_model.name)
        model = load_model(model_path, with_fsa=with_fsa)
        rgb_image = float32(image) / 255
        input_tensor = preprocess_image(rgb_image)
        target_layers = [model.stages[target_layer_index]]

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            visualization = show_cam_on_image(
                rgb_image, cam(input_tensor=input_tensor)[0, :], use_rgb=True
            )

        return visualization
    except Exception as exception:
        print(f"Grad-CAM error: {exception}")
        return None


def update_layer_choices(uploaded_model: File):
    if uploaded_model is None:
        return Dropdown(choices=[], value=None)

    try:
        model_path = Path(uploaded_model.name)
        model = load_model(model_path, with_fsa=True)
        num_stages = len(model.stages)
        choices = [(f"Stage {index}", index) for index in range(num_stages)]
        return Dropdown(choices=choices, value=num_stages - 1)
    except Exception:
        return Dropdown(choices=[], value=None)


def make_dashboard() -> Blocks:
    with Blocks() as dashboard:
        Markdown(
            "# ConvNext V2 Model with Focal Self-Attention and Grad-CAM++ for PCOM Classification using Ultrasound Images"
        )

        Markdown("Note: The uploaded model architecture should match the FSA checkbox.")
        fsa_checkbox = Checkbox(label="Use Focal Self-Attention (FSA)")
        upload_model = File(label="Upload model (.pt)")

        image_input = Image(type="pil", label="Upload image")
        predict_button = Button("Predict")

        with Row():
            predicted_label = Label(label="Predicted label")
            predicted_confidence = Number(label="Confidence")

        layer_selector = Dropdown(label="Target Layer", choices=[], value=None)
        show_gradcam_button = Button("Show Grad-CAM++")
        gradcam_output = Image(label="Grad-CAM++ Visualization")

        upload_model.change(
            fn=update_layer_choices, inputs=[upload_model], outputs=[layer_selector]
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
