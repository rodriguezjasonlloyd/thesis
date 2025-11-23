from enum import Enum
from typing import Sequence

import cv2
import numpy
from cv2 import GaussianBlur
from PIL import Image
from torch.nn import Module


class PreprocessingMode(Enum):
    NONE = "none"
    CLAHE = "clahe"
    OTSU_THRESHOLD = "otsu_threshold"
    DEEP_CONTRAST = "deep_contrast"
    ALL = "all"


class CLAHE(Module):
    def __init__(self, clip_limit: float = 4.0, tile_grid_size: Sequence[int] = (8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def forward(self, image: Image.Image) -> Image.Image:
        return Image.fromarray(
            cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            ).apply(numpy.array(image.convert("L")))
        ).convert("RGB")


class OtsuThreshold(Module):
    def __init__(self, blur_kernel_size: tuple[int, int] = (3, 3)):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size

    def forward(self, image: Image.Image) -> Image.Image:
        _, thresholded = cv2.threshold(
            GaussianBlur(numpy.array(image.convert("L")), self.blur_kernel_size, 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return Image.fromarray(thresholded).convert("RGB")


class DeepContrast(Module):
    def __init__(self, gamma: float = 2.5, blur_strength: int = 9):
        super().__init__()
        self.gamma = gamma
        self.blur_strength = blur_strength

    def forward(self, image: Image.Image) -> Image.Image:
        return Image.fromarray(
            (
                numpy.power(
                    cv2.bilateralFilter(
                        numpy.array(image.convert("L")),
                        d=self.blur_strength,
                        sigmaColor=75,
                        sigmaSpace=75,
                    )
                    / 255.0,
                    self.gamma,
                )
                * 255
            )
        ).convert("RGB")


class AggressiveComposite(Module):
    def __init__(self):
        super().__init__()
        self.clahe_filter = CLAHE(clip_limit=8.0, tile_grid_size=(2, 2))
        self.otsu_filter = OtsuThreshold(blur_kernel_size=(3, 3))
        self.deep_contrast_filter = DeepContrast(gamma=1.5, blur_strength=3)

    def forward(self, img: Image.Image) -> Image.Image:
        channel1 = numpy.array(self.clahe_filter(img).convert("L"))
        channel2 = numpy.array(self.otsu_filter(img).convert("L"))
        channel3 = numpy.array(self.deep_contrast_filter(img).convert("L"))
        return Image.fromarray(numpy.stack([channel1, channel2, channel3], axis=2))
