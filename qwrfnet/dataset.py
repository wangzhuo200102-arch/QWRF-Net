"""Dataset utilities for radar precipitation nowcasting."""

from __future__ import annotations

import json
import os
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from tqdm import tqdm


def process_h5_file_worker(args):
    """Quality-check one HDF5 file and return valid event identifiers."""
    file_path, max_vil = args
    passed_events = []
    try:
        with h5py.File(file_path, "r") as handle:
            if "vil" not in handle or handle["vil"].ndim != 4:
                return []
            all_events = handle["vil"]
            for event_idx in range(all_events.shape[0]):
                reference_frame = all_events[event_idx, :, :, 9]
                total_pixels = reference_frame.size
                too_many_zeros = np.sum(reference_frame == 0) > 0.8 * total_pixels
                too_many_saturated = np.sum(reference_frame == max_vil) > 0.8 * total_pixels
                if too_many_zeros or too_many_saturated:
                    continue

                event_data = all_events[event_idx, :, :, :18]
                if event_data.max() <= 0:
                    continue

                event_thw = np.transpose(event_data, (2, 0, 1)).astype(np.float32)
                stats = [event_thw.max(), event_thw.min(), event_thw.mean(), event_thw.var()]
                passed_events.append(((file_path, event_idx), stats))
    except Exception:
        return []
    return passed_events


def prepare_dataset_index(data_path: str, max_vil: float, index_file_path: str, years=None):
    """Create a JSON index of valid SEVIR VIL samples."""
    years = years or ["2017", "2018", "2019"]
    all_h5_files = []
    for year in years:
        year_dir = os.path.join(data_path, year)
        if not os.path.exists(year_dir):
            continue
        for root, _, files in os.walk(year_dir):
            for filename in files:
                if filename.endswith(".h5"):
                    all_h5_files.append(os.path.join(root, filename))

    if not all_h5_files:
        raise RuntimeError(f"No .h5 files were found under {data_path}.")

    tasks = [(path, max_vil) for path in all_h5_files]
    quality_passed, stats_for_kmeans = [], []
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_h5_file_worker, tasks), total=len(tasks)))

    for file_results in results:
        for identifier, stats in file_results:
            quality_passed.append(identifier)
            stats_for_kmeans.append(stats)

    if not quality_passed:
        raise RuntimeError("No events passed the quality check.")

    stats_array = np.array(stats_for_kmeans)
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(scaler.fit_transform(stats_array))
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    valid_cluster_id = np.argmin(centers[:, 2])
    valid_identifiers = [
        identifier for i, identifier in enumerate(quality_passed) if labels[i] == valid_cluster_id
    ]

    os.makedirs(os.path.dirname(index_file_path) or ".", exist_ok=True)
    with open(index_file_path, "w", encoding="utf-8") as handle:
        json.dump(valid_identifiers, handle)
    return valid_identifiers


class SEVIRPrecipitationDataset(Dataset):
    """SEVIR VIL dataset for 6-frame to 12-frame nowcasting."""

    def __init__(self, index_file_path: str, max_vil: float = 255.0, img_size=(288, 288)) -> None:
        self.max_vil = max_vil
        self.resize = Resize(img_size, antialias=True)
        try:
            with open(index_file_path, "r", encoding="utf-8") as handle:
                self.sample_info = json.load(handle)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Index file not found: {index_file_path}") from exc

    def __len__(self) -> int:
        return len(self.sample_info)

    def __getitem__(self, idx: int):
        file_path, frame_idx = self.sample_info[idx]
        with h5py.File(file_path, "r") as handle:
            vil_hwt = handle["vil"][frame_idx, ..., :18].astype(np.float32)
        vil_thw = np.transpose(vil_hwt, (2, 0, 1))
        vil_tensor = torch.from_numpy(vil_thw).contiguous()
        inputs = self.resize(vil_tensor[:6]) / self.max_vil
        targets = self.resize(vil_tensor[6:18]) / self.max_vil
        return inputs, targets

