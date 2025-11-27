from pathlib import Path
from typing import Literal

from gradio import Blocks, Button, Dropdown, Image, Label, Markdown, Row
from gradio.themes.default import Default
from gradio.utils import NamedString
from PIL.Image import Image as PillowImage

from modules import dashboard

ModelChoice = Literal[
    "convnext",
    "convnext_fsa",
    "convnext_pre",
    "convnext_fsa_pre",
    "convnext_imagenet",
    "convnext_fsa_imagenet",
    "convnext_pre_imagenet",
    "convnext_fsa_pre_imagenet",
    "base_cnn",
]

MODEL_CONFIGS: dict[ModelChoice, dict[str, str | bool]] = {
    "convnext": {
        "path": "demo/convnext.pt",
        "architecture": "convnext",
        "with_fsa": False,
        "target_layer": "stages.1.blocks",
        "preprocessing": "none",
        "pretrained": False,
    },
    "convnext_fsa": {
        "path": "demo/convnext_fsa.pt",
        "architecture": "convnext",
        "with_fsa": True,
        "target_layer": "stages.1.blocks",
        "preprocessing": "none",
        "pretrained": False,
    },
    "convnext_pre": {
        "path": "demo/convnext_pre.pt",
        "architecture": "convnext",
        "with_fsa": False,
        "target_layer": "stages.1.blocks",
        "preprocessing": "all",
        "pretrained": False,
    },
    "convnext_fsa_pre": {
        "path": "demo/convnext_fsa_pre.pt",
        "architecture": "convnext",
        "with_fsa": True,
        "target_layer": "stages.1.blocks",
        "preprocessing": "all",
        "pretrained": False,
    },
    "convnext_imagenet": {
        "path": "demo/convnext_imagenet.pt",
        "architecture": "convnext",
        "with_fsa": False,
        "target_layer": "stages.1.blocks",
        "preprocessing": "none",
        "pretrained": True,
    },
    "convnext_fsa_imagenet": {
        "path": "demo/convnext_fsa_imagenet.pt",
        "architecture": "convnext",
        "with_fsa": True,
        "target_layer": "stages.1.blocks",
        "preprocessing": "none",
        "pretrained": True,
    },
    "convnext_pre_imagenet": {
        "path": "demo/convnext_pre_imagenet.pt",
        "architecture": "convnext",
        "with_fsa": False,
        "target_layer": "stages.1.blocks",
        "preprocessing": "all",
        "pretrained": True,
    },
    "convnext_fsa_pre_imagenet": {
        "path": "demo/convnext_fsa_pre_imagenet.pt",
        "architecture": "convnext",
        "with_fsa": True,
        "target_layer": "stages.1.blocks",
        "preprocessing": "all",
        "pretrained": True,
    },
    "base_cnn": {
        "path": "demo/base_cnn.pt",
        "architecture": "base",
        "with_fsa": False,
        "target_layer": "features.6",
        "preprocessing": "none",
        "pretrained": False,
    },
}


def predict_wrapper(
    image: PillowImage | None, model_choice: ModelChoice
) -> tuple[str, str]:
    if image is None:
        return ("No image uploaded", "")

    config = MODEL_CONFIGS[model_choice]
    model_path = Path(str(config["path"]))

    if not model_path.exists():
        return (f"Model not found: {model_path}", "")

    model_file = NamedString(str(model_path))
    model_file.name = str(model_path)

    return dashboard.predict_image(
        uploaded_model=model_file,
        uploaded_image=image,
        architecture=str(config["architecture"]),
        with_fsa=bool(config["with_fsa"]),
        preprocessing=str(config["preprocessing"]),
        pretrained=bool(config["pretrained"]),
    )


def cam_wrapper(
    image: PillowImage | None, model_choice: ModelChoice
) -> PillowImage | None:
    if image is None:
        return None

    config = MODEL_CONFIGS[model_choice]
    model_path = Path(str(config["path"]))

    if not model_path.exists():
        return None

    model_file = NamedString(str(model_path))
    model_file.name = str(model_path)

    return dashboard.generate_cam(
        uploaded_model=model_file,
        uploaded_image=image,
        architecture=str(config["architecture"]),
        with_fsa=bool(config["with_fsa"]),
        layer_name=str(config["target_layer"]),
        preprocessing=str(config["preprocessing"]),
        pretrained=bool(config["pretrained"]),
    )


def make_demo() -> Blocks:
    with Blocks(theme=Default(text_size="lg")) as demo:
        Markdown("# ConvNext V2 with Grad-CAM++ Dashboard for PCOM Classification")

        model_dropdown = Dropdown(
            label="Model Choice",
            choices=[
                ("ConvNeXt V2", "convnext"),
                ("ConvNeXt V2 with Focal Self Attention", "convnext_fsa"),
                ("ConvNeXt V2 (With Preprocessing)", "convnext_pre"),
                ("ConvNeXt V2 with FSA (With Preprocessing)", "convnext_fsa_pre"),
                ("ConvNeXt V2 ImageNet", "convnext_imagenet"),
                ("ConvNeXt V2 ImageNet with FSA", "convnext_fsa_imagenet"),
                (
                    "ConvNeXt V2 ImageNet (With Preprocessing)",
                    "convnext_pre_imagenet",
                ),
                (
                    "ConvNeXt V2 ImageNet with FSA (With Preprocessing)",
                    "convnext_fsa_pre_imagenet",
                ),
                ("Base CNN", "base_cnn"),
            ],
            value="convnext",
            interactive=True,
        )

        upload_image = Image(type="pil", label="Upload Image", height=300)
        predict_button = Button("Predict")

        with Row():
            predicted_label = Label(label="Predicted Label")
            predicted_confidence = Label(label="Confidence")

        show_cam_button = Button("Show Grad-CAM++")
        cam_output = Image(label="Grad-CAM++ Visualization", height=300)

        predict_button.click(
            fn=predict_wrapper,
            inputs=[upload_image, model_dropdown],
            outputs=[predicted_label, predicted_confidence],
        )

        show_cam_button.click(
            fn=cam_wrapper,
            inputs=[upload_image, model_dropdown],
            outputs=cam_output,
        )

    return demo
