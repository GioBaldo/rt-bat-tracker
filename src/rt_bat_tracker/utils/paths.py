from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProjectPaths:
    project_dir: Path
    data_dir: Path
    audio_dir: Path
    mic_layout_dir: Path
    groundtruth_dir: Path
    results_dir: Path
    config_dir: Path
    gif_dir: Path
    png_dir: Path


def get_project_paths() -> ProjectPaths:
    project_dir = Path(__file__).resolve().parents[3]

    paths = ProjectPaths(
        project_dir=project_dir,
        data_dir=project_dir / "data",
        audio_dir=project_dir / "data" / "input_audio_files",
        mic_layout_dir=project_dir / "data" / "mic_layout",
        groundtruth_dir=project_dir / "data" / "groundtruth",
        results_dir=project_dir / "results",
        config_dir=project_dir / "config",
        gif_dir=project_dir / "results" / "gif",
        png_dir=project_dir / "results" / "png",
    )

    return paths
