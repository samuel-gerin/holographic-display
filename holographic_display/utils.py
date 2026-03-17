import torch
from numpy.typing import NDArray


ArrayLike = NDArray | torch.Tensor


def normalize(array: ArrayLike) -> ArrayLike:
    return array / abs(array).max()


def crop_center(array: ArrayLike, crop_height: int | None = None, crop_width: int | None = None) -> ArrayLike:
    y, x = array.shape
    if crop_height is None:
        crop_height = y // 2
    if crop_width is None:
        crop_width = x // 2

    start_x = x // 2 - (crop_width // 2)
    start_y = y // 2 - (crop_height // 2)
    return array[start_y:start_y + crop_height, start_x:start_x + crop_width]


def create_grid(
        num_pixels_x: int,
        dx: float | None = None,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if dx is None:
        dx = 1

    if dy is None:
        dy = dx

    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    width = dx * num_pixels_x
    height = dy * num_pixels_y
    x = torch.linspace(-width / 2, width / 2, num_pixels_x)
    y = torch.linspace(-height / 2, height / 2, num_pixels_y)
    return torch.meshgrid(x, y, indexing="ij")
