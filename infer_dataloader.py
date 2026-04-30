from torch.utils.data import DataLoader

from dataset_source import WaveInferDataset


def create_infer_dataloader(
    spectrum_dir,
    wave_nc,
    wind_nc,
    sample_indices,
    time_index,
    lead_time,
    batch_size,
    num_workers,
    rescale_forcings=False,
    norm_stats_path=None,
    sparse_mask_path=None,
):
    """Create an inference DataLoader using WaveInferDataset with non-contiguous indices."""
    dataset = WaveInferDataset(
        spectrum_root_dir=spectrum_dir,
        wave_nc_path=wave_nc,
        wind_nc_path=wind_nc,
        indices=sample_indices,
        time_index=time_index,
        lead_time=lead_time,
        rescale_forcings=rescale_forcings,
        norm_stats_path=norm_stats_path,
        sparse_mask_path=sparse_mask_path,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )