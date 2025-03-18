from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.integrate import simpson
from scipy.fft import fftfreq, fftshift, rfftn
import numpy.typing as npt
from loguru import logger
from matplotlib.figure import Figure
from rich.progress import track
from rich.console import Console

Distance = StrEnum("Distance", "UNIFORM GAUSSIAN R_POWER_MINUS_2 R_POWER_1")


class Settings(BaseSettings):
    c: float = Field(default=299792.458, description="Speed of light in km/s")
    omega_m: float = Field(default=0.3089, description="Matter density parameter")
    distance: Distance = Field(default=Distance.UNIFORM, description="Distance function")
    angles: list[float] = Field(default=[22.5, 45.0, 90.0, 180.0], description="Angles in degrees")
    radius: float = Field(default=150.0, description="Radius in Mpc", gt=0)
    grid_size: int = Field(default=250, description="Grid size. Must be even.", gt=0)

    model_config = SettingsConfigDict(cli_parse_args=True)

    @property
    def box_size(self) -> float:
        return self.radius * 15

    @property
    def omega_l(self) -> float:
        return 1.0 - self.omega_m

    @field_validator("grid_size")
    @classmethod
    def check_grid_size(cls, v):
        if v % 2 != 0:
            raise ValueError("Grid size must be even.")
        return v


def survey_window_function(
    x: npt.NDArray,
    y: npt.NDArray,
    z: npt.NDArray,
    max_dist: float,
    dist: Distance,
    phi_cutoff: float,
) -> npt.NDArray | None:
    r = np.sqrt(x**2 + y**2 + z**2)

    if dist == Distance.UNIFORM:
        phidata = np.arccos(z / r)
        prob = np.ones(r.shape)
        prob[phidata > phi_cutoff] = 0.0
        prob[r > max_dist] = 0.0
        return prob

    elif dist == Distance.GAUSSIAN:
        phidata = np.arccos(z / r)
        prob = np.exp(-(r**2) / ((max_dist / 3) ** 2))
        prob[phidata > phi_cutoff] = 0.0
        prob[r > max_dist] = 0.0

        return prob

    elif dist == Distance.R_POWER_MINUS_2:
        phidata = np.arccos(z / r)
        prob = 1.0 / (r**2)
        prob[phidata > phi_cutoff] = 0.0
        prob[r > max_dist] = 0.0

        return prob

    elif dist == Distance.R_POWER_1:
        phidata = np.arccos(z / r)
        prob = r
        prob[phidata > phi_cutoff] = 0.0
        prob[r > max_dist] = 0.0

        return prob

    raise ValueError(f"Unknown distance function: {dist=}")


def plot_power_spectrum(settings: Settings, console: Console) -> Figure:
    fig, ax = plt.subplots()

    # setting up window function data
    x_data = np.linspace(-settings.box_size, settings.box_size, settings.grid_size)
    y_data = x_data.copy()
    z_data = x_data.copy()

    # set up 3D coordinate data to get W in real space in 3D numerically
    x_grid, y_grid, z_grid = np.array(np.meshgrid(x_data, y_data, z_data))

    # numpy fft routine
    x_range = float(np.max(x_data) - np.min(x_data))
    y_range = float(np.max(y_data) - np.min(y_data))
    z_range = float(np.max(z_data) - np.min(z_data))

    # set up k-space frequencies
    # get k modes - factor of 2pi is so that these
    # are consistent with k = 2pi factor in normalization for k modes in matter power spectrum
    ksx = fftshift((fftfreq(settings.grid_size + 1, x_range / len(x_data)))) * (2.0 * np.pi)
    ksy = fftshift((fftfreq(settings.grid_size + 1, y_range / len(y_data)))) * (2.0 * np.pi)
    ksz = fftshift((fftfreq(settings.grid_size + 1, z_range / len(z_data)))) * (2.0 * np.pi)

    # set up the radius of the equivalent spherical volume
    volstandard = 4.0 * np.pi * (settings.radius**3) / 3.0  # volume of sphere with this r = maxdistgal

    for i, theta in enumerate(track(settings.angles, console=console, description="Calculating power spectrum")):
        console.log(f"Calculating power spectrum for {theta=}")

        max_dist_gal = settings.radius

        if theta < 180.0:
            angle = np.pi * theta / 180.0
            # adjust the grid maxdistgal so the volume is constant for cones with opening angle theta < 180
            max_dist_gal = np.power(volstandard * 3.0 / (2.0 * np.pi * (1.0 - np.cos(angle))), 1.0 / 3.0)

            if max_dist_gal > settings.box_size:
                msg = f"Size of space volume is too small to fit the survey inside: {max_dist_gal=} > {settings.box_size=}"
                logger.error(msg)
                raise ValueError(msg)

        w_real_space = survey_window_function(
            x_grid,
            y_grid,
            z_grid,
            max_dist_gal,
            settings.distance,
            phi_cutoff=theta * np.pi / 180.0,
        )

        # now get the window function in Fourier space numerically
        int_over_wrealsp = simpson(simpson(simpson(w_real_space, z_data), y_data), x_data)

        # calculate 3D fourier transform - fast fourier transform - requires different normalization with (2pi)^(3/2) for using ifft() inverse
        w_fourier_space = (
            rfftn(w_real_space, (settings.grid_size + 1, settings.grid_size + 1, settings.grid_size + 1))
            * (np.max(x_data) - np.min(x_data)) ** 3  # type: ignore
            / (len(x_data) ** 3 * int_over_wrealsp)
        )

        w_fourier_space = np.concatenate(
            (w_fourier_space[:, :, :], np.conj(np.flip(w_fourier_space[:, :, 1:], axis=2))), axis=2
        )
        w_fourier_space = fftshift(w_fourier_space)

        ks_v = np.logspace(-3, 2, 200, base=10)
        ks_cn = (ks_v[1:] + ks_v[0:-1]) / 2.0
        ks_x_grid, ks_y_grid, ks_z_grid = np.meshgrid(ksx, ksy, ksz)
        ks_abs_reshape = np.sqrt(ks_x_grid**2 + ks_y_grid**2 + ks_z_grid**2).reshape(-1)
        fft_reshape = abs(w_fourier_space.reshape(-1))

        index_arr = np.zeros((len(ks_abs_reshape)))
        for i, _ in enumerate(ks_cn):
            index_arr = np.where(np.logical_and(ks_abs_reshape >= ks_v[i], ks_abs_reshape < ks_v[i + 1]), i, index_arr)

        binned_df = (
            pd.DataFrame({"fftval": fft_reshape, "index": index_arr})
            .groupby("index")
            .agg(sum=("fftval", "sum"), counts=("fftval", "count"))
            .reset_index()
            .assign(power=lambda x: x["sum"] / x["counts"], index=lambda x: x["index"].astype(int))
            .assign(powah=lambda x: x["power"].abs() ** 2)
            .fillna(0.0)
        )

        ax.semilogx(ks_cn[binned_df["index"]], binned_df["powah"], label=f"$\\theta$ = {theta:0.2f}")

    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum for Survey Geometry using {settings.distance}")
    console.log("Returning figure")
    return fig


if __name__ == "__main__":
    console = Console()
    settings = Settings()
    console.log("Running with settings:")
    console.print_json(settings.model_dump_json(indent=2))

    fig = plot_power_spectrum(settings, console)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"powerspectrum_surveygeometry_{settings.distance}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    console.log(f"[bold green]Saved figure to {output_file}[/bold green]")
