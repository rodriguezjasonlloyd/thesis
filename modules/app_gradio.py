from pathlib import Path
from typing import Any

from gradio import Blocks, Button, File, Image, Markdown, Number, Tab, Textbox
from torch import no_grad
from torch.nn.functional import softmax

from modules.data import get_class_names, get_data_root_path, transform_image_to_tensor
from modules.model import load_model


def predict_image(image: Any, uploaded_model: object) -> tuple[str, float]:
    if image is None:
        return ("No image provided", 0.0, None)

    if uploaded_model is None:
        return ("No model file uploaded", 0.0, None)

    try:
        model_path = Path(uploaded_model.name)
    except Exception:
        return ("Error reading uploaded file", 0.0, None)

    try:
        model = load_model(model_path, pretrained=False, with_fsa=False)
    except Exception as exception:
        return (f"Model load error: {exception}", 0.0, None)

    try:
        tensor = transform_image_to_tensor(image, pretrained=False)
    except Exception as exception:
        return (f"Image transform error: {exception}", 0.0, None)

    with no_grad():
        device = next(model.parameters()).device
        output = model(tensor.unsqueeze(0).to(device))
        probabilities = softmax(output, dim=1).cpu().numpy()[0]
        prediction_index = int(probabilities.argmax())

    try:
        classes = get_class_names(get_data_root_path())
        label = classes[prediction_index]
    except Exception:
        label = str(prediction_index)

    confidence = float(probabilities[prediction_index])

    return (label, confidence)


def make_dashboard() -> Blocks:
    with Blocks() as demo:
        Markdown("# Thesis")

        with Tab("Predict"):
            Markdown(
                "### ConvNext V2 Model with Focal Self-Attention and Grad-CAM++ for PCOM Classification using Ultrasound Images"
            )

            upload_model = File(label="Upload model (.pt)")
            image_input = Image(type="pil", label="Upload image")
            predict_button = Button("Predict")
            predicted_label = Textbox(label="Predicted label")
            predicted_confidence = Number(label="Confidence")

            predict_button.click(
                fn=predict_image,
                inputs=[image_input, upload_model],
                outputs=[predicted_label, predicted_confidence],
            )

    return demo
