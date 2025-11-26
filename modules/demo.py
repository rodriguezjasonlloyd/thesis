from pathlib import Path

from gradio import Blocks, Button, Dropdown, Image, Label, Markdown, Row
from gradio.themes.default import Default
from gradio.utils import NamedString

from modules import dashboard

MODEL_CONFIGS = {
    "convnext": {
        "path": "demo/convnext.pt",
        "architecture": "convnext",
        "with_fsa": False,
        "target_layer": "stages.1.blocks",
    },
    "convnext_fsa": {
        "path": "demo/convnext_fsa.pt",
        "architecture": "convnext",
        "with_fsa": True,
        "target_layer": "stages.1.blocks",
    },
    "base_cnn": {
        "path": "demo/base_cnn.pt",
        "architecture": "base",
        "with_fsa": False,
        "target_layer": "features.6",
    },
}


def predict_wrapper(image, preprocessing, model_choice):
    if image is None:
        return ("No image uploaded", "")

    config = MODEL_CONFIGS[model_choice]
    model_path = Path(config["path"])

    if not model_path.exists():
        return (f"Model not found: {model_path}", "")

    model_file = NamedString(str(model_path))
    model_file.name = str(model_path)

    return dashboard.predict_image(
        uploaded_model=model_file,
        uploaded_image=image,
        architecture=config["architecture"],
        with_fsa=config["with_fsa"],
        preprocessing=preprocessing,
    )


def cam_wrapper(image, preprocessing, model_choice):
    if image is None:
        return None

    config = MODEL_CONFIGS[model_choice]
    model_path = Path(config["path"])

    if not model_path.exists():
        return None

    model_file = NamedString(str(model_path))
    model_file.name = str(model_path)

    return dashboard.generate_cam(
        uploaded_model=model_file,
        uploaded_image=image,
        architecture=config["architecture"],
        with_fsa=config["with_fsa"],
        layer_name=config["target_layer"],
        preprocessing=preprocessing,
    )


def make_demo() -> Blocks:
    with Blocks(theme=Default(text_size="lg")) as dashboard:
        Markdown("# ConvNext V2 with Grad-CAM++ Dashboard for PCOM Classification")

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
            interactive=True,
        )

        model_dropdown = Dropdown(
            label="Model Choice",
            choices=[
                ("ConvNeXt V2", "convnext"),
                ("ConvNeXt V2 with Focal Self Attention", "convnext_fsa"),
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
            inputs=[upload_image, preprocessing_dropdown, model_dropdown],
            outputs=[predicted_label, predicted_confidence],
        )

        show_cam_button.click(
            fn=cam_wrapper,
            inputs=[upload_image, preprocessing_dropdown, model_dropdown],
            outputs=cam_output,
        )

    return dashboard
