import torch

from torch.nn.functional import interpolate
from torchvision.transforms import CenterCrop
from tqdm.auto import trange

from holographic_display.utils import create_grid, normalize


def compute_complex_coherence_factor(source_intensity: torch.Tensor) -> torch.Tensor:
    return normalize(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(source_intensity))))


def compute_angular_spectrum(
        field: torch.Tensor, z: float, dx: float, wavelength: float,
) -> torch.Tensor:
    def quadratic_position_index(n: int):
        return (wavelength / dx / n * torch.arange(-n / 2, n / 2)) ** 2

    qx = quadratic_position_index(field.shape[1])
    qy = quadratic_position_index(field.shape[0])
    qy = qy[:, torch.newaxis]
    propagation_kernel = torch.exp(2j * torch.pi / wavelength * z * torch.sqrt(1 - qx - qy))
    return torch.fft.ifft2(torch.fft.fft2(field) * torch.fft.ifftshift(propagation_kernel))


def _propagate(
        source: torch.Tensor,
        phase: torch.Tensor,
        phase_factor: torch.Tensor,
        nx_source_numerical: int,
        z_object_camera: float,
        dx: float,
        wavelength: float,
        reference_wavelength: float,
):
    source_interpolated = interpolate(
        source[torch.newaxis, torch.newaxis],
        size=(nx_source_numerical, nx_source_numerical),
        mode="bilinear",
    )[0, 0]
    complex_coherence_factor = compute_complex_coherence_factor(
        CenterCrop([nx_source_numerical * 2, nx_source_numerical * 2])(source_interpolated)
    )
    real = interpolate(
        torch.real(complex_coherence_factor)[torch.newaxis, torch.newaxis],
        size=phase.size(),
        mode="bilinear",
    )[0, 0]
    imag = interpolate(
        torch.imag(complex_coherence_factor)[torch.newaxis, torch.newaxis],
        size=phase.size(),
        mode="bilinear",
    )[0, 0]

    object = torch.exp(2j * torch.pi * phase * wavelength / reference_wavelength) * phase_factor
    object_spectral_density = torch.abs(
        compute_angular_spectrum(object, z_object_camera, dx, wavelength)
    ) ** 2
    object_autocorrelation = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(object_spectral_density)))

    convolved_fields = object_autocorrelation * torch.complex(real, imag)
    propagated_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(convolved_fields)))
    return torch.abs(propagated_field)


def propagate(
        source: torch.Tensor,
        phase: torch.Tensor,
        wavelengths: list[float],
        reference_wavelength: float,
        dx_source: float,
        dx_slm: float,
        z_source_slm: float,
        propagation_distances: list[float],
        nx_camera: int,
        ny_camera: int,
):
    ny_source, nx_source, _ = source.shape
    ny_slm, nx_slm = phase.shape
    nx_source_numerical = [
        round(z / z_source_slm * dx_source / dx_slm * nx_source)
        for z in propagation_distances
    ]

    x, y = create_grid(nx_slm * 2, dx_slm, ny_slm * 2)
    phase_factors = [
        torch.exp(1j * torch.pi / wavelength * (1 / z_source_slm) * (x ** 2 + y ** 2))
        for wavelength in wavelengths
    ]

    results = []
    max_intensities = []
    for j in trange(len(propagation_distances)):
        rgb = []
        for k in range(3):
            result = _propagate(
                source[:, :, k],
                CenterCrop([ny_slm * 2, nx_slm * 2])(phase),
                phase_factors[k],
                nx_source_numerical[j],
                propagation_distances[j],
                dx_slm,
                wavelengths[k],
                reference_wavelength
            )
            if j == 0:
                max_intensities.append(result.max())
            result = CenterCrop([ny_camera, nx_camera])(result / max_intensities[k])
            rgb.append(result)

        estimated_image = torch.stack(rgb, dim=-1)
        results.append(estimated_image)

    return results
