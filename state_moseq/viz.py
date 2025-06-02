import os
import cv2
import imageio
import numpy as np
import tqdm
from vidio.read import OpenCVReader
from typing import Dict, Tuple, List, Union
from jaxtyping import Array, Float, Int
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from .util import sample_instances, compare_states, get_frequencies

def crop_image(
    image: Array,
    centroid: Tuple[int, int],
    crop_size: Union[int, Tuple[int, int]],
) -> Array:
    """Crop an image around a centroid.

    Args:
        image: Image to crop, shape (height, width, channels).
        centroid: (x, y) coordinates of the centroid.
        crop_size: Size of crop around centroid (int or (width, height)).

    Returns:
        cropped_image: Cropped (and zero-padded if needed) image.
    """
    if isinstance(crop_size, tuple):
        w, h = crop_size
    else:
        w, h = crop_size, crop_size
    x, y = int(centroid[0]), int(centroid[1])

    x_min = max(0, x - w // 2)
    y_min = max(0, y - h // 2)
    x_max = min(image.shape[1], x + w // 2)
    y_max = min(image.shape[0], y + h // 2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h, w, *image.shape[2:]), dtype=image.dtype)
    pad_x = max(w // 2 - x, 0)
    pad_y = max(h // 2 - y, 0)
    padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
    return padded


def write_video_clip(
    frames: Array,
    path: str,
    fps: int = 30,
    quality: int = 7,
) -> None:
    """Write a video clip to a file using imageio.

    Args:
        frames: Video frames of shape (n_frames, height, width, channels).
        path: Output file path.
        fps: Frames per second.
        quality: Video encoding quality.
    """
    with imageio.get_writer(path, pixelformat="yuv420p", fps=fps, quality=quality) as writer:
        for frame in frames:
            writer.append_data(frame)


def grid_movie(
    instances: List[Tuple[str, int, int]],
    rows: int,
    cols: int,
    videos: Dict,
    centroids: Dict[str, Float[Array, "n_timesteps 2"]],
    window_size: int,
    max_duration: int = 1200,
    dimming_factor: float = 0.5,
) -> Int[Array, "n_frames window_size*rows window_size*cols 3"]:
    """Generate a grid movie from video instances.

    Args:
        instances: List of (key, start, end) tuples.
        rows: Number of rows in grid.
        cols: Number of columns in grid.
        videos: Dictionary of video readers keyed by name.
        centroids: Dictionary of centroids for each frame.
        window_size: Size of cropped window around centroid.
        max_duration: Maximum duration of the grid movie.
        dimming_factor: Multiplicative factor for dimming after state instance has ended.

    Returns:
        frames: Combined movie frames as a single array.
    """
    tiles = []
    instance_durations = [end - start for _, start, end in instances]
    max_duration = min(max_duration, max(instance_durations))
    for key, start, end in instances:
        tile = []
        video_length = len(videos[key])
        for t in range(start, start + max_duration):
            if t >= video_length:
                frame = np.zeros((window_size, window_size, 3), np.uint8)
            else:
                cen = centroids[key][t]
                frame = crop_image(videos[key][t], cen, window_size)
                if t >= end:
                    frame = np.uint8(frame * dimming_factor)
                frame[:,0] = 100
                frame[:,-1] = 100
                frame[0,:] = 100
                frame[-1,:] = 100
            tile.append(frame)
        tiles.append(np.stack(tile))

    for _ in range(rows * cols - len(tiles)):
        tiles.append(np.zeros_like(tiles[0]))

    tiles = np.stack(tiles).reshape(rows, cols, max_duration, window_size, window_size, 3)
    frames = np.concatenate(np.concatenate(tiles, axis=2), axis=2)
    return frames


def generate_grid_movies(
    states_dict: Dict[str, Int[Array, "n_timesteps"]],
    video_paths: Dict[str, str],
    centroids: Dict[str, Float[Array, "n_timesteps 2"]],
    output_dir: str = "grid_movies",
    rows: int = 3,
    cols: int = 4,
    max_duration: int = 1200,
    dimming_factor: float = 0.25,
    quality: int = 7,
    window_size: int = 256,
) -> Dict[int, List[Tuple[str, int, int]]]:
    """Generate and save grid movies for high-level states.

    Args:
        states_dict: Dictionary of state sequences.
        video_paths: Dictionary of paths to video files.
        centroids: Dictionary of centroids for each frame.
        output_dir: Directory to save output movies.
        rows: Number of rows in each grid.
        cols: Number of columns in each grid.
        max_duration: Maximum duration of each movie.
        dimming_factor: Factor to dim frames after state instance has ended.
        quality: Video encoding quality.
        window_size: Size of window around centroid.

    Returns:
        instances: Sampled instances used for grid movies.
    """
    os.makedirs(output_dir, exist_ok=True)
    videos = {k: OpenCVReader(path) for k, path in video_paths.items()}
    fps = videos[list(video_paths.keys())[0]].fps

    instances = sample_instances(states_dict, rows * cols)
    for state_ix, inst_list in tqdm.tqdm(instances.items(), desc="Generating grid movies", ncols=72):
        frames = grid_movie(
            inst_list,
            rows,
            cols,
            videos,
            centroids,
            window_size,
            max_duration=max_duration,
            dimming_factor=dimming_factor,
        )
        path = os.path.join(output_dir, f"state{state_ix}.mp4")
        write_video_clip(frames, path, fps=fps, quality=quality)

    return instances



def plot_sankey(
    states_dict1: Dict[str, Int[Array, "n_timesteps"]],
    states_dict2: Dict[str, Int[Array, "n_timesteps"]],
) -> go.Figure:
    """Create a Sankey diagram representing the relationship between two sets of states.

    Args:
        states_dict1: First dictionary of state sequences.
        states_dict2: Second dictionary of state sequences.

    Returns:
        fig: Plotly figure object containing the Sankey diagram.
    """
    import plotly.io as pio
    pio.renderers.default = "notebook"

    confusion_matrix = compare_states(states_dict1, states_dict2)[0]
    frequencies1 = get_frequencies(states_dict1)
    frequencies2 = get_frequencies(states_dict2)
    n1, n2 = confusion_matrix.shape
    
    cmap = plt.get_cmap("tab20")
    node_colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)" 
                   for r, g, b, _ in [cmap(i % 20) for i in range(max(n1, n2))]]
    node_colors_combined = node_colors[:n1] + node_colors[:n2]
    node_labels = [f"state {i}" for i in range(n1)] + [f"state {i}" for i in range(n2)]

    sources, targets, values, link_colors = [], [], [], []
    for i in range(n1):
        for j in range(n2):
            val = confusion_matrix[i, j] * frequencies1[i]
            if val > 0:
                sources.append(i)
                targets.append(n1 + j)  # offset right side
                values.append(val)
                link_colors.append(node_colors[i])

    # Build Sankey
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors_combined
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    ))
    return fig