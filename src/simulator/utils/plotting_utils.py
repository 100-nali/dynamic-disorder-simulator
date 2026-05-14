"""
Any plotting functions should be located here

Copyright © 2023 QuantrolOx Ltd.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from simulator.computational_graph.abstract_computational_graph import ComputationalGraph
from simulator.computational_graph.graph_nodes.nodes.initial_electrostatic_potential import (
    InitialElectroStaticPotential,
)
from simulator.computational_graph.utils.device_config import DeviceConfig, GateTypeIndices


def get_gate_type_indices(graph: ComputationalGraph) -> GateTypeIndices:
    """
    Allows the user to identify which indices are allocated to which gate.
    """

    iep = [x for x in graph.components if isinstance(x, InitialElectroStaticPotential)][0]

    categories_dict: Dict = {
        "separator_gates": [],
        "sensor_barrier_gates": [],
        "sensor_plunger_gates": [],
        "qubit_barrier_gates": [],
        "qubit_plunger_gates": [],
    }
    categories = list(categories_dict.keys())

    print("Categories to be classified:")
    for j, category in enumerate(categories):
        print(f"{j+1}: {category}")

    for gate_idx, gate_arr in enumerate(iep.gate_split):
        plt.figure()
        plt.title(f"gate {gate_idx}")
        plt.imshow(gate_arr)
        plt.show(block=False)

        # Get user input and ensure it is valid
        while True:
            try:
                category_index = int(input("Enter the number of the category: ")) - 1
                if 0 <= category_index < len(categories):
                    key = categories[category_index]
                    categories_dict[key].append(gate_idx)
                    plt.close()
                    break

                print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    gate_types = GateTypeIndices(**categories_dict)
    return gate_types


def array_to_image(arr: np.ndarray, filename: str, cmap: str = "gray"):
    """
    Convert a 1D numpy array into an image with a vertical column of squares.
    The color of the squares correspond to the values in the array on a monochromatic scale.

    Parameters:
    - arr (numpy array): 1D array with values between 0 (black) and 1 (white).

    Returns:
    - None, but saves the image as 'output_image.png'.
    """

    # Ensure the array is 1D
    assert len(arr.shape) == 1, "The input array must be 1D."

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(1, len(arr)))

    # Display the array as an image
    ax.imshow(arr.reshape(-1, 1), cmap=cmap, aspect="auto")

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# pylint:disable=too-many-arguments
def save_array_with_colorbar(
    filename: str,
    data: np.ndarray,
    cmap: str = "gray",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Saves a 2D array as an image with a colorbar to the specified filename.

    Args:
        filename (str): The name of the file to save the image to.
        data (np.ndarray): A 2D array containing the data to be saved as an image.
        cmap (str, optional): The color map to use for the image. Defaults to 'gray'.
        title (str, optional): An optional title for the plot
        xaxis, yaxis (str, optional): Optional labels for the axes of the plot
    """
    curr_backend = plt.get_backend()
    plt.switch_backend("Agg")
    # Reduce axes of size 1
    data = np.squeeze(data)
    if len(data.shape) != 2:
        raise ValueError("Data should be 2d.")

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)

    # Set the title and labels if they are not None
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.colorbar(im, ax=ax, orientation="vertical")
    fig.savefig(filename)
    plt.close(fig)
    plt.switch_backend(curr_backend)


def plot_potential_landscape(potential: np.ndarray, dim: str = "2D"):
    """
    Plots the potential landscape in 2D or 3D
    args:
        potential: 2D array of potential landscape magnitude
    """

    fig = plt.figure()
    if dim.lower() == "3d":
        ny, nx = potential.shape
        x = np.arange(nx)
        y = np.arange(ny)
        xm, ym = np.meshgrid(x, y)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xm, ym, potential)
        ax.set_title("Potential Landscape")
        plt.show(block=False)
    else:
        plt.imshow(potential)
        plt.colorbar()
        plt.title("Potential Landscape")
        plt.show(block=False)


def show_potential(potential: np.ndarray, title: str = "Potential < 0") -> None:
    """
    Shows the potential sign change boundary to the user.

    Args:
        potential (np.ndarray): numpy array representing the electrostatic potential
        title (str): a title for the plot
    """

    plt.figure()
    plt.title(title)
    plt.imshow(potential < 0)
    plt.show(block=False)


def show_potential_with_locales(
    potential: np.ndarray, locales: List[np.ndarray], title: str = "Potential <0"
) -> None:
    """
    Plots the potential with markers for locales.

    If the locales values are negative, they are assumed to wrap around.

    Args:
        potential (np.ndarray): the potential to show
        locales (List[np.ndarray]): a list of locations to plot on the potential
            note the index ordering is different to usual!
        title (str): a title for the plot
    """
    plt.figure()
    plt.title(title)
    plt.imshow(potential < 0)
    for l in locales:
        if l[1] < 0:
            l[1] += potential.shape[1]
        if l[0] < 0:
            l[0] += potential.shape[1]
        plt.scatter(l[1], l[0], marker="x", c="r", s=50)

    plt.show(block=False)


def plot_gate_voltages(
    voltages: np.ndarray, device_config: DeviceConfig, filename: str, cmap: str = "cividis"
) -> None:
    """
    Plots a visual represenation of the voltages applied to the gate.

    Args:
        voltages: the voltages applied to the gates
        device_config: the configuration of the device
    """
    assert device_config
    device_image = device_config.gate_config.gate_img_array
    if not isinstance(device_image, np.ndarray):
        raise ValueError("Device config does not have a gate image array")

    flattened = np.mean(device_image, axis=2)
    values = np.unique(flattened)

    if len(values) != len(voltages) + 1:
        raise ValueError("Gate architecture has mismatched number of gates")

    voltage_img_array = np.zeros_like(flattened)

    # Apply voltages to gates
    for value, voltage in zip(values[:-1], voltages):
        voltage_img_array[np.where(flattened == value)] = voltage

    save_array_with_colorbar(
        filename=filename, data=voltage_img_array, cmap=cmap, title="Gate voltages"
    )
