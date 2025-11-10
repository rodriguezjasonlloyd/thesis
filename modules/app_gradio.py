from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import torch
from torch.nn.functional import softmax

from modules.data import get_class_names, get_data_root_path, transform_image_to_tensor
from modules.model import build_model


def load_model(model_path: Path, pretrained: bool, with_fsa: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=pretrained, with_fsa=with_fsa)
    model = model.to(device)

    if model_path and model_path.exists():
        state = torch.load(model_path, map_location=device)

        try:
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

        except Exception:
            for value in state.values():
                if isinstance(value, dict):
                    try:
                        model.load_state_dict(value)
                        break
                    except Exception:
                        continue

    model.eval()

    return model


def predict_image(image: Any, uploaded_model: object):
    if image is None:
        return ("No image provided", 0.0, None)

    if uploaded_model is None:
        return ("No model file uploaded", 0.0, None)

    try:
        model_path = Path(uploaded_model.name)
    except Exception:
        return ("Error reading uploaded file", 0.0, None)

    if model_path.suffix.lower() != ".pth":
        return ("Invalid file type: please upload a .pth model file.", 0.0, None)

    try:
        model = load_model(model_path, pretrained=False, with_fsa=False)
    except Exception as exception:
        return (f"Model load error: {exception}", 0.0, None)

    try:
        tensor = transform_image_to_tensor(image, pretrained=False)
    except Exception as exception:
        return (f"Image transform error: {exception}", 0.0, None)

    with torch.no_grad():
        device = next(model.parameters()).device
        out = model(tensor.unsqueeze(0).to(device))
        probs = softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    try:
        classes = get_class_names(get_data_root_path())
        label = classes[pred_idx]
    except Exception:
        label = str(pred_idx)

    confidence = float(probs[pred_idx])

    # heatmap = generate_placeholder_heatmap(image)

    return (label, confidence, image)


def predict_uploaded_only(image: Any, uploaded_model: object):
    return predict_image(image, uploaded_model)


def make_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Thesis")

        with gr.Tab("Predict"):
            gr.Markdown(
                "### ConvNext V2 Model with Focal Self-Attention and Grad-CAM++ for PCOM Classification using Ultrasound Images"
            )

            upload_model = gr.File(label="Upload model (.pth)")

            img_input = gr.Image(type="pil", label="Upload image")

            predict_btn = gr.Button("Predict")

            pred_label = gr.Textbox(label="Predicted label")
            pred_conf = gr.Number(label="Confidence")
            heatmap_output = gr.Image(label="GradCAM++ Heatmap")

            predict_btn.click(
                fn=predict_uploaded_only,
                inputs=[img_input, upload_model],
                outputs=[pred_label, pred_conf, heatmap_output],
            )

    return demo
