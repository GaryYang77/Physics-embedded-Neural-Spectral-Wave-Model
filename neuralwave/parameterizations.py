"""Wave parameterization modules used by the public inference path."""
import torch
from typing import Dict, Optional, Union, List
from torch import nn
from tqdm import tqdm
import gin

from waden_hpc import wavetypings
from waden_hpc import flux_calculation
from neuralwave import mappers
from neuralwave import transforms
from neuralwave import embeddings
from networks.unet_custom import UNet

@gin.register
class BaseParameterization(nn.Module):
    """
    Base class for wave parameterizations.
    This class serves as a no-op parameterization, passing the input spectra through unchanged.
    """
    
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.StotSpectra:
        """
        Apply the parameterization to the input wave spectra.

        Args:
            spectra: Input wave spectra [B, H, W, D, F]
            environdata: Optional environdata (e.g., wind, bathymetry)
            mask: Optional mask to apply to the spectra

        Returns:
            spectra: Pa1rameterized wave spectra [B, H, W, D, F]
        """
        return torch.zeros_like(spectra)  # No-op by default

    def compute_input_norm(self, train_loader, model, device):
        """Default placeholder for parameterization-specific input norm computation."""
        return NotImplemented

@gin.register
class MLPParameterization(BaseParameterization):
    """
    Use a simple MLP to learn source/sink terms from environmental data.
    """
    
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Optional[Dict[str, embeddings.BaseFeatureGenerator]] = gin.REQUIRED,
        mlp_input_dim: int = gin.REQUIRED,
        **kwargs
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper
        self.feature_generators = nn.ModuleDict(feature_generators) if feature_generators is not None else nn.ModuleDict()
        self.mlp_input_dim = mlp_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 720)
        )

    def compute_input_norm(self, train_loader, model, device):
        """
        Compute per-channel min/max normalization stats for MLP input features.
        
        Returns: tuple (min_vals, scale, mode) where:
            - min_vals: per-channel minimum [mlp_input_dim]
            - scale: per-channel (max - min) [mlp_input_dim]
            - mode: normalization mode string ("minmax")
        """
        if not hasattr(self, "mlp_input_dim"):
            return NotImplemented

        min_ = torch.full((self.mlp_input_dim,), float("inf"), device=device, dtype=torch.float64)
        max_ = torch.full((self.mlp_input_dim,), -float("inf"), device=device, dtype=torch.float64)

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing input norm"):
                spec_ini, forcings, bathy, spec_label, mask, sparse_mask, _ = batch
                spec_ini = spec_ini.squeeze(1).to(device)
                bathy_step0 = bathy[:, 0].squeeze(-1).to(device)
                forc_step0 = forcings[:, 0].to(device)
                mask_in = mask.squeeze(-1).squeeze(1).to(device)
                spec_ini[torch.isnan(spec_ini)] = 0.0

                environ = {
                    "depth": bathy_step0,
                    "bathymetry": forc_step0[:, :, :, 0],
                    "wind_u": forc_step0[:, :, :, 1],
                    "wind_v": forc_step0[:, :, :, 2],
                }

                freq = model.metadata.get("freq", None)
                depth = environ.get("depth").unsqueeze(-1).unsqueeze(-1)
                wavenum = model.wavenum_solver(freq=freq, depth=depth, metadata=model.metadata)
                group_velocity = model.group_velocity_calculator(freq=freq, wavenum=wavenum, depth=depth)
                metadata = model.metadata.copy()
                metadata.update({"wavenum": wavenum, "Cg": group_velocity})

                feats = self.build_combined_features(spec_ini, environ, metadata, mask_in)
                if feats.shape[1] != self.mlp_input_dim:
                    raise ValueError(
                        f"Normalization dimension mismatch: expected {self.mlp_input_dim}, got {feats.shape[1]}"
                    )

                feats = feats.reshape(-1, self.mlp_input_dim).double()
                batch_min = feats.min(dim=0).values
                batch_max = feats.max(dim=0).values
                min_ = torch.minimum(min_, batch_min)
                max_ = torch.maximum(max_, batch_max)

        # Handle zero-scale: for minmax mode, use global max; for scale mode, use 10
        global_max = torch.max(max_)
        global_max_safe = torch.clamp(global_max, min=1e-8)
        scale = torch.abs(max_ - min_)
        scale = torch.where(scale == 0, torch.full_like(scale, global_max_safe), scale)

        return min_.float(), scale.float(), "minmax"
    
    def _apply_input_norm(
        self,
        combined_features: torch.Tensor,
        input_norm: Optional[Dict[str, torch.Tensor]],
        expected_dim: int,
    ) -> torch.Tensor:
        """
        Apply input normalization based on mode.
        
        Supports:
        - "minmax": (x - min) / scale, zero-scale -> global max
        - "scale": x / scale, zero-scale -> 10
        - Legacy (no mode): treated as scale-only division
        """
        if input_norm is None:
            return combined_features
        
        scale = input_norm.get("scale") if isinstance(input_norm, dict) else None
        if scale is None:
            raise ValueError("input_norm must include 'scale'")
        if scale.numel() != expected_dim:
            raise ValueError(
                f"Normalization dimension mismatch: expected {expected_dim}, got scale {scale.numel()}"
            )
        
        mode = input_norm.get("mode", "scale")  # default to legacy scale mode
        
        if mode == "minmax":
            min_vals = input_norm.get("min")
            if min_vals is None:
                raise ValueError("input_norm mode='minmax' must include 'min'")
            # Handle zero-scale: use global max for minmax mode
            global_max = torch.max(scale)
            global_max_safe = torch.clamp(global_max, min=1e-8)
            scale_safe = torch.where(scale == 0, torch.full_like(scale, global_max_safe), scale)
            return (combined_features - min_vals) / scale_safe
        else:
            # scale mode: x / scale, zero-scale -> 10
            scale_safe = torch.where(scale == 0, torch.full_like(scale, 10.0), scale)
            return combined_features / scale_safe

    def build_combined_features(
        self,
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None,
    ) -> torch.Tensor:

        spectra_masked = self.spectra_entransform(spectra, mask=mask)
        spectra_mapped = self.spectra_mapper(spectra_masked, environdata)
        b, h, w, d, f = spectra_mapped.shape

        if len(self.feature_generators) > 0:
            num_spatial_points = b * h * w
            spectra_points = spectra_mapped.reshape(num_spatial_points, 1, d, f)

            # Flatten each channel to tolerate mismatched trailing dims (e.g., D x F vs D x 1).
            flattened_channels: List[torch.Tensor] = [spectra_points.reshape(num_spatial_points, -1)]

            for key, generator in self.feature_generators.items():
                feature_map = generator(spectra_points, environdata, metadata, num_spatial_points)
                flattened_channels.append(feature_map.reshape(num_spatial_points, -1))

            combined_features = torch.cat(flattened_channels, dim=1)
        else:
            wind_u = environdata.get('wind_u')
            wind_v = environdata.get('wind_v')
            bathy = environdata.get('bathymetry')

            spectra_flat = spectra_mapped.view(b * h * w, d * f)
            env_features_flat = torch.stack([
                wind_u.reshape(b * h * w),
                wind_v.reshape(b * h * w),
                bathy.reshape(b * h * w)
            ], dim=-1)

            combined_features = torch.cat([env_features_flat, spectra_flat * 0.01], dim=-1)

        if combined_features.shape[1] != self.mlp_input_dim:
            raise ValueError(
                f"MLP input dimension mismatch: expected {self.mlp_input_dim}, got {combined_features.shape[1]}"
            )
        return combined_features

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.StotSpectra:
        b, h, w, d, f = spectra.shape

        combined_features = self.build_combined_features(spectra, environdata, metadata, mask)

        input_norm = None if environdata is None else environdata.get("input_norm")
        combined_features = self._apply_input_norm(combined_features, input_norm, self.mlp_input_dim)

        stot_flat = self.mlp(combined_features)  # [B*H*W, Features]

        stot = stot_flat.view(b, h, w, d, f)

        stot_final = self.spectra_detransform(stot, mask=mask)
        
        return stot_final * 0.001

@gin.register
class GlobalMLPParameterization(MLPParameterization):
    """
    Inherits from MLPParameterization but adds intra-batch blocking for memory efficiency (For Global Simulations).
    """
    
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Optional[Dict[str, embeddings.BaseFeatureGenerator]] = gin.REQUIRED,
        mlp_input_dim: int = gin.REQUIRED,
        num_intra_batch_block: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            spectra_entransform,
            spectra_detransform,
            spectra_mapper,
            spatial_mapper,
            feature_generators,
            mlp_input_dim,
        )

        self.num_intra_batch_block = num_intra_batch_block


    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.StotSpectra:
        b, h, w, d, f = spectra.shape

        combined_features = self.build_combined_features(spectra, environdata, metadata, mask)

        input_norm = None if environdata is None else environdata.get("input_norm")
        combined_features = self._apply_input_norm(combined_features, input_norm, self.mlp_input_dim)

        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            stot_flat = self.mlp(combined_features)  # [B*H*W, Features]
        else:
            n_points = combined_features.shape[0]
            blocks = int(self.num_intra_batch_block)
            block_size = (n_points + blocks - 1) // blocks
            outputs = []
            for start in range(0, n_points, block_size):
                end = min(start + block_size, n_points)
                outputs.append(self.mlp(combined_features[start:end]))
            stot_flat = torch.cat(outputs, dim=0)

        stot = stot_flat.view(b, h, w, d, f)

        stot_final = self.spectra_detransform(stot, mask=mask)
        
        return stot_final

@gin.register
class SamplingMLPParameterization(MLPParameterization):
    """
    Use a simple MLP to learn source/sink terms from sampling environmental data.
    """
    
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Optional[Dict[str, embeddings.BaseFeatureGenerator]] = gin.REQUIRED,
        mlp_input_dim: int = gin.REQUIRED,
        sample_ratio: Optional[float] = gin.REQUIRED,
        min_samples: Optional[int] = gin.REQUIRED,
        num_intra_batch_block: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            spectra_entransform,
            spectra_detransform,
            spectra_mapper,
            spatial_mapper,
            feature_generators,
            mlp_input_dim,
        )
        self.sample_ratio = sample_ratio if sample_ratio is not None else 0.2
        self.min_samples = min_samples if min_samples is not None else 1
        self.num_intra_batch_block = num_intra_batch_block

    def _run_mlp_blocked(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            return self.mlp(inputs)
        n_points = inputs.shape[0]
        blocks = int(self.num_intra_batch_block)
        block_size = (n_points + blocks - 1) // blocks
        outputs = []
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            outputs.append(self.mlp(inputs[start:end]))
        return torch.cat(outputs, dim=0)

    def forward(
            self,
            spectra: wavetypings.FieldSpectra,
            environdata: Optional[wavetypings.EnvironmentDataDict] = None,
            metadata: Optional[wavetypings.MetadataDict] = None,
            mask: Optional[wavetypings.ValidMask] = None,
        ) -> wavetypings.StotSpectra:
            spectra_masked = self.spectra_entransform(spectra, mask=mask)
            spectra_mapped = self.spectra_mapper(spectra_masked, environdata)
            b, h, w, d, f = spectra_mapped.shape

            combined_features = self.build_combined_features(spectra, environdata, metadata, mask)

            input_norm = None if environdata is None else environdata.get("input_norm")
            combined_features = self._apply_input_norm(combined_features, input_norm, self.mlp_input_dim)

            mask_flat = mask.view(-1)         # [B*H*W]
            valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
            n_valid = valid_idx.numel()
            if n_valid == 0:
                stot = torch.zeros((b, h, w, d, f), device=spectra.device, dtype=spectra.dtype)
                stot_final = self.spectra_detransform(stot, mask=mask)
                return stot_final * 0.001
            n_sample = max(int(n_valid * self.sample_ratio), self.min_samples)
            n_sample = min(n_sample, n_valid)

            sample_idx = valid_idx[torch.randperm(n_valid, device=mask_flat.device)[:n_sample]]
            stot_sample = self.mlp(combined_features[sample_idx])

            with torch.no_grad():
                rest_mask = torch.ones(n_valid, device=mask_flat.device, dtype=torch.bool)
                rest_mask[:n_sample] = False
                rest_idx = valid_idx[rest_mask]
                stot_rest = None
                if rest_idx.numel() > 0:
                    stot_rest = self._run_mlp_blocked(combined_features[rest_idx])

            stot_full = torch.zeros((b * h * w, d * f), device=combined_features.device, dtype=combined_features.dtype)
            stot_full[sample_idx] = stot_sample
            if rest_idx.numel() > 0:
                stot_full[rest_idx] = stot_rest
            stot = stot_full.view(b, h, w, d, f)

            stot_final = self.spectra_detransform(stot, mask=mask)
            return stot_final * 0.001

@gin.register
class CNNFusionParameterization(nn.Module):
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Optional[Dict[str, embeddings.BaseFeatureGenerator]] = gin.REQUIRED,
        env_feature_dim: int = gin.REQUIRED,
        embedding_dim: int = gin.REQUIRED,
        spec_shape: tuple = (24, 30),  # (D, F)
        **kwargs
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper
        self.feature_generators = nn.ModuleDict(feature_generators) if feature_generators is not None else nn.ModuleDict()

        self.spec_D, self.spec_F = spec_shape
        self.n_spec_features = self.spec_D * self.spec_F

        self.env_encoder = nn.Sequential(
            nn.Linear(env_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        self.spectra_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16, self.spec_D, self.spec_F]),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LayerNorm([32, self.spec_D, self.spec_F]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_spec_features)
        )

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.StotSpectra:
        """
        Apply CNN-based parameterization using a compiler-friendly, full-tensor workflow.
        """
        spectra_masked = self.spectra_entransform(spectra, mask=mask)
        wind_u = environdata.get('wind_u')
        wind_v = environdata.get('wind_v')
        bathy = environdata.get('bathymetry')
        
        b, h, w, d, f = spectra_masked.shape
        num_spatial_points = b * h * w
        
        # [B, H, W, D, F] -> [B*H*W, 1, D, F]
        spectra_image = spectra_masked.reshape(num_spatial_points, 1, d, f)
        spectra_mapped = self.spectra_mapper(spectra_masked, environdata)
        # Shape: [B*H*W, 1, D, F] -> [B*H*W, embedding_dim]
        spectra_embedding = self.spectra_encoder(spectra_image)
        
        if len(self.feature_generators) > 0:
            num_spatial_points = b * h * w
            # [B, H, W, D, F] -> [B*H*W, 1, D, F]
            spectra_points = spectra_mapped.reshape(num_spatial_points, 1, d, f)
            input_channels = []
            
            for key, generator in self.feature_generators.items():
                feature_map = generator(spectra_points, environdata, metadata, num_spatial_points)
                input_channels.append(feature_map)
            
            env_features_flat = torch.cat(input_channels, dim=1).reshape(num_spatial_points, -1)
        else:
            # [B, H, W] -> [B*H*W, 3]
            env_features_flat = torch.stack([
                wind_u.reshape(num_spatial_points),
                wind_v.reshape(num_spatial_points),
                bathy.reshape(num_spatial_points)
            ], dim=-1)
        
        # Shape: [B*H*W, 3] -> [B*H*W, embedding_dim]
        env_embedding = self.env_encoder(env_features_flat)

        # Shape: [B*H*W, embedding_dim * 2]
        combined_embedding = torch.cat([spectra_embedding, env_embedding], dim=-1)
        
        # Shape: [B*H*W, embedding_dim * 2] -> [B*H*W, D*F]
        stot_flat = self.decoder(combined_embedding)
        
        # Shape: [B*H*W, D*F] -> [B, H, W, D, F]
        stot = stot_flat.reshape(b, h, w, d, f)
        
        stot_final = self.spectra_detransform(stot, mask=mask)
        
        return stot_final
@gin.register
class UNetParameterization(nn.Module):
    def __init__(
        self, 
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Dict[str, embeddings.BaseFeatureGenerator] = gin.REQUIRED,
        unet: UNet = gin.REQUIRED,
        spec_shape: tuple = (24, 30),  # (D, F)
        norm_feature: Optional[List[str]] = None,  # features to normalize, e.g., ['spectra', 'wind', 'sdn']
        norm_mode: str = "minmax",  # "minmax" or "scale"
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper
        self.spec_D, self.spec_F = spec_shape
        
        self.feature_generators = nn.ModuleDict(feature_generators)

        self.unet = unet
        
        # Normalization settings
        # norm_feature: list of feature names to normalize (including 'spectra' for the spectrum channel)
        # Only features present in feature_generators (or 'spectra') will be normalized; others silently skipped.
        self.norm_feature = set(norm_feature) if norm_feature is not None else set()
        self.norm_mode = norm_mode  # "minmax" or "scale"
        
        # Build ordered list of channel names for normalization indexing
        # Channel order: ['spectra'] + list(feature_generators.keys())
        self._channel_names = ['spectra'] + list(self.feature_generators.keys())

    def compute_input_norm(self, train_loader, model, device):
        """
        Compute per-channel min/max normalization stats for UNet input channels.
        Only computes stats for channels listed in norm_feature (silently skips others).
        
        Returns: tuple (min_vals, scale, mode, features) where:
            - min_vals: per-channel minimum for norm_feature channels
            - scale: per-channel (max - min) for norm_feature channels
            - mode: normalization mode string
            - features: ordered list of normalized feature names
        """
        # Determine which channels to normalize
        channels_to_norm = []
        channel_indices = []
        for i, name in enumerate(self._channel_names):
            if name in self.norm_feature:
                channels_to_norm.append(name)
                channel_indices.append(i)
        
        if len(channels_to_norm) == 0:
            return NotImplemented
        
        num_channels = len(channels_to_norm)
        min_ = torch.full((num_channels,), float("inf"), device=device, dtype=torch.float64)
        max_ = torch.full((num_channels,), -float("inf"), device=device, dtype=torch.float64)

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing UNet input norm"):
                spec_ini, forcings, bathy, spec_label, mask, sparse_mask, _ = batch
                spec_ini = spec_ini.squeeze(1).to(device)
                bathy_step0 = bathy[:, 0].squeeze(-1).to(device)
                forc_step0 = forcings[:, 0].to(device)
                mask_in = mask.squeeze(-1).squeeze(1).to(device)
                spec_ini[torch.isnan(spec_ini)] = 0.0

                environ = {
                    "depth": bathy_step0,
                    "bathymetry": forc_step0[:, :, :, 0],
                    "wind_u": forc_step0[:, :, :, 1],
                    "wind_v": forc_step0[:, :, :, 2],
                }

                freq = model.metadata.get("freq", None)
                depth = environ.get("depth").unsqueeze(-1).unsqueeze(-1)
                wavenum = model.wavenum_solver(freq=freq, depth=depth, metadata=model.metadata)
                group_velocity = model.group_velocity_calculator(freq=freq, wavenum=wavenum, depth=depth)
                metadata = model.metadata.copy()
                metadata.update({"wavenum": wavenum, "Cg": group_velocity})

                # Build UNet input channels
                spectra_masked = self.spectra_entransform(spec_ini, mask=mask_in)
                b, h, w, d, f = spectra_masked.shape
                num_spatial_points = b * h * w
                spectra_points = spectra_masked.reshape(num_spatial_points, 1, d, f)
                
                input_channels = [spectra_points]
                for key, generator in self.feature_generators.items():
                    feature_map = generator(spectra_points, environ, metadata, num_spatial_points)
                    input_channels.append(feature_map)
                
                # [N, C, D, F]
                unet_input = torch.cat(input_channels, dim=1)
                
                # Extract only channels to normalize and compute per-channel stats
                for idx_out, idx_in in enumerate(channel_indices):
                    channel_data = unet_input[:, idx_in, :, :].reshape(-1).double()
                    batch_min = channel_data.min()
                    batch_max = channel_data.max()
                    min_[idx_out] = torch.minimum(min_[idx_out], batch_min)
                    max_[idx_out] = torch.maximum(max_[idx_out], batch_max)

        # Handle zero-scale based on mode
        scale = torch.abs(max_ - min_)
        if self.norm_mode == "minmax":
            global_max = torch.max(max_)
            global_max_safe = torch.clamp(global_max, min=1e-8)
            scale = torch.where(scale == 0, torch.full_like(scale, global_max_safe), scale)
        else:
            # scale mode: zero-scale -> 10
            scale = torch.where(scale == 0, torch.full_like(scale, 10.0), scale)

        return min_.float(), scale.float(), self.norm_mode, channels_to_norm

    def _apply_input_norm(
        self,
        unet_input: torch.Tensor,
        input_norm: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Apply per-channel normalization to UNet input based on mode.
        
        Args:
            unet_input: [N, C, D, F] tensor
            input_norm: dict with 'mode', 'scale', 'min' (optional), 'features'
        
        Returns:
            Normalized unet_input tensor
        """
        if input_norm is None:
            return unet_input
        
        mode = input_norm.get("mode", "scale")
        scale = input_norm.get("scale")
        features = input_norm.get("features", [])
        
        if scale is None or len(features) == 0:
            return unet_input
        
        # Only normalize channels explicitly requested by norm_feature; ignore extra stats in input_norm
        allowed_features = set(features) if len(self.norm_feature) == 0 else set(features).intersection(self.norm_feature)
        if len(allowed_features) == 0:
            return unet_input

        # Build mapping from feature name to norm index
        feature_to_norm_idx = {name: i for i, name in enumerate(features)}
        
        # Apply normalization to each channel that was normalized
        # unet_input = unet_input.clone()
        for ch_idx, ch_name in enumerate(self._channel_names):
            if ch_name not in allowed_features or ch_name not in feature_to_norm_idx:
                continue
            norm_idx = feature_to_norm_idx[ch_name]
            ch_scale = scale[norm_idx]
            
            if mode == "minmax":
                min_vals = input_norm.get("min")
                if min_vals is None:
                    raise ValueError("input_norm mode='minmax' requires 'min'")
                ch_min = min_vals[norm_idx]
                # Handle zero-scale: use global max for minmax mode
                global_max = torch.max(scale)
                global_max_safe = torch.clamp(global_max, min=1e-8)
                ch_scale_safe = ch_scale if ch_scale != 0 else global_max_safe
                unet_input[:, ch_idx, :, :] = (unet_input[:, ch_idx, :, :] - ch_min) / ch_scale_safe
            else:
                # scale mode: zero-scale -> 10
                ch_scale_safe = ch_scale if ch_scale != 0 else 10.0
                unet_input[:, ch_idx, :, :] = unet_input[:, ch_idx, :, :] / ch_scale_safe
        
        return unet_input

    def forward(
        self, 
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None
    ) -> wavetypings.StotSpectra:
        """Run the U-Net parameterization with dynamic feature channels."""
        spectra_masked = self.spectra_entransform(spectra, mask=mask)
        
        b, h, w, d, f = spectra_masked.shape
        num_spatial_points = b * h * w
        
        spectra_points = spectra_masked.reshape(num_spatial_points, 1, d, f)

        input_channels = [spectra_points]

        for key, generator in self.feature_generators.items():
            feature_map = generator(spectra_points, environdata, metadata, num_spatial_points)
            input_channels.append(feature_map)

        unet_input = torch.cat(input_channels, dim=1)
        
        # Apply input normalization if provided
        input_norm = None if environdata is None else environdata.get("input_norm")
        unet_input = self._apply_input_norm(unet_input, input_norm)
        
        stot_image = self.unet(unet_input)
        
        # [B*H*W, 1, D, F] -> [B, H, W, D, F]
        stot = stot_image.reshape(b, h, w, d, f)
        
        stot_final = self.spectra_detransform(stot, mask=mask)
        
        return stot_final * 0.001


@gin.register
class SamplingUNetParameterization(UNetParameterization):
    """
    UNet parameterization with sampling and intra-batch blocking for memory efficiency.
    """

    def __init__(
        self,
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Dict[str, embeddings.BaseFeatureGenerator] = gin.REQUIRED,
        unet: UNet = gin.REQUIRED,
        spec_shape: tuple = (24, 30),
        norm_feature: Optional[List[str]] = None,
        norm_mode: str = "minmax",
        sample_ratio: Optional[float] = gin.REQUIRED,
        min_samples: Optional[int] = gin.REQUIRED,
        num_intra_batch_block: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            spectra_entransform=spectra_entransform,
            spectra_detransform=spectra_detransform,
            spectra_mapper=spectra_mapper,
            spatial_mapper=spatial_mapper,
            feature_generators=feature_generators,
            unet=unet,
            spec_shape=spec_shape,
            norm_feature=norm_feature,
            norm_mode=norm_mode,
        )
        self.sample_ratio = sample_ratio if sample_ratio is not None else 0.2
        self.min_samples = min_samples if min_samples is not None else 1
        self.num_intra_batch_block = num_intra_batch_block

    def _run_unet_blocked(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_intra_batch_block is None or self.num_intra_batch_block <= 1:
            return self.unet(inputs)
        n_points = inputs.shape[0]
        blocks = int(self.num_intra_batch_block)
        block_size = (n_points + blocks - 1) // blocks
        outputs = []
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            outputs.append(self.unet(inputs[start:end]))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None,
    ) -> wavetypings.StotSpectra:
        spectra_masked = self.spectra_entransform(spectra, mask=mask)

        b, h, w, d, f = spectra_masked.shape
        num_spatial_points = b * h * w

        spectra_points = spectra_masked.reshape(num_spatial_points, 1, d, f)

        input_channels = [spectra_points]
        for key, generator in self.feature_generators.items():
            feature_map = generator(spectra_points, environdata, metadata, num_spatial_points)
            input_channels.append(feature_map)

        unet_input = torch.cat(input_channels, dim=1)

        input_norm = None if environdata is None else environdata.get("input_norm")
        unet_input = self._apply_input_norm(unet_input, input_norm)

        if mask is None:
            valid_mask = torch.ones(num_spatial_points, device=unet_input.device, dtype=torch.bool)
        else:
            #mask_flat = mask.view(-1)
            mask_flat = mask.reshape(-1)
            valid_mask = mask_flat > 0

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        n_valid = valid_idx.numel()
        if n_valid == 0:
            stot = torch.zeros((b, h, w, d, f), device=spectra.device, dtype=spectra.dtype)
            stot_final = self.spectra_detransform(stot, mask=mask)
            return stot_final * 0.001

        n_sample = max(int(n_valid * self.sample_ratio), self.min_samples)
        n_sample = min(n_sample, n_valid)

        stot_full = torch.zeros(
            (num_spatial_points, 1, d, f),
            device=unet_input.device,
            dtype=unet_input.dtype,
        )

        if n_sample == n_valid:
            stot_full[valid_idx] = self._run_unet_blocked(unet_input[valid_idx])
        else:
            perm = torch.randperm(n_valid, device=valid_idx.device)
            sample_idx = valid_idx[perm[:n_sample]]
            rest_idx = valid_idx[perm[n_sample:]]

            stot_sample = self._run_unet_blocked(unet_input[sample_idx])

            stot_rest = None
            if rest_idx.numel() > 0:
                with torch.no_grad():
                    stot_rest = self._run_unet_blocked(unet_input[rest_idx])

            stot_full[sample_idx] = stot_sample
            if stot_rest is not None:
                stot_full[rest_idx] = stot_rest

        stot = stot_full.reshape(b, h, w, d, f)
        stot_final = self.spectra_detransform(stot, mask=mask)

        return stot_final * 0.001


@gin.register
class BoostingUNetParameterization(nn.Module):
    """Two-branch U-Net with a main path and a wind-focused high-frequency branch."""

    def __init__(
        self,
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Dict[str, embeddings.BaseFeatureGenerator] = gin.REQUIRED,
        main_unet: UNet = gin.REQUIRED,
        wind_unet: UNet = gin.REQUIRED,
        wind_feature_key: str = "wind",
        high_freq_start: int = 24,
        high_freq_end: int = 30,
        spec_shape: tuple = (24, 30),
        norm_feature: Optional[List[str]] = None,
        norm_mode: str = "minmax",
        main_features: Optional[List[str]] = None,
        wind_features: Optional[List[str]] = None,
        wind_include_spectra: bool = False,
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper
        self.feature_generators = nn.ModuleDict(feature_generators)
        self.main_unet = main_unet
        self.wind_unet = wind_unet
        self.wind_feature_key = wind_feature_key
        self.high_freq_start = high_freq_start
        self.high_freq_end = high_freq_end
        self.spec_D, self.spec_F = spec_shape
        self.norm_feature = set(norm_feature) if norm_feature is not None else set()
        self.norm_mode = norm_mode
        # Resolve feature routing; allow overlaps and optional spectra in wind branch
        default_main = ["spectra"] + [k for k in feature_generators if k != wind_feature_key]
        default_wind = [wind_feature_key]
        self.main_features = main_features if main_features is not None else default_main
        self.wind_features = wind_features if wind_features is not None else default_wind
        self.wind_include_spectra = wind_include_spectra
        if self.wind_include_spectra and "spectra" not in self.wind_features:
            self.wind_features = ["spectra"] + list(self.wind_features)

        self._channel_names_main = list(self.main_features)
        self._channel_names_wind = list(self.wind_features)

        mask = torch.zeros(1, 1, self.spec_D, self.spec_F)
        mask[..., self.high_freq_start:self.high_freq_end] = 1.0
        self.register_buffer("high_freq_mask", mask)
    
    def compute_input_norm(self, train_loader, model, device):
        return NotImplemented, "BoostingUNetParameterization does not support compute_input_norm directly."

    def _build_main_input(
        self,
        spectra_points: torch.Tensor,
        environdata: Optional[wavetypings.EnvironmentDataDict],
        metadata: Optional[wavetypings.MetadataDict],
        num_spatial_points: int,
    ) -> torch.Tensor:
        channels: List[torch.Tensor] = []
        for name in self.main_features:
            if name == "spectra":
                channels.append(spectra_points)
                continue
            if name not in self.feature_generators:
                raise ValueError(f"main_features contains an unregistered feature: {name}")
            feature_map = self.feature_generators[name](
                spectra_points, environdata, metadata, num_spatial_points
            )
            channels.append(feature_map)

        return torch.cat(channels, dim=1)

    def _build_wind_input(
        self,
        spectra_points: torch.Tensor,
        environdata: Optional[wavetypings.EnvironmentDataDict],
        metadata: Optional[wavetypings.MetadataDict],
        num_spatial_points: int,
    ) -> torch.Tensor:
        channels: List[torch.Tensor] = []
        for name in self.wind_features:
            if name == "spectra":
                channels.append(spectra_points)
                continue
            if name in self.feature_generators:
                feature_map = self.feature_generators[name](
                    spectra_points, environdata, metadata, num_spatial_points
                )
                channels.append(feature_map)
                continue

            # Backward-compatible fallback for raw wind magnitude
            if name == self.wind_feature_key:
                if environdata is None:
                    raise ValueError("The wind branch requires environdata")
                wind_u = environdata.get("wind_u")
                wind_v = environdata.get("wind_v")
                if wind_u is None or wind_v is None:
                    raise ValueError("The wind branch requires wind_u and wind_v")
                wind_mag = torch.sqrt(wind_u ** 2 + wind_v ** 2)
                wind_mag_flat = wind_mag.reshape(num_spatial_points, 1, 1, 1)
                channels.append(wind_mag_flat.expand(-1, 1, self.spec_D, self.spec_F))
            else:
                raise ValueError(f"wind_features contains an unregistered feature: {name}")

        return torch.cat(channels, dim=1)

    def _apply_input_norm_channels(
        self,
        unet_input: torch.Tensor,
        channel_names: List[str],
        input_norm: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if input_norm is None:
            return unet_input

        mode = input_norm.get("mode", "scale")
        scale = input_norm.get("scale")
        features = input_norm.get("features", [])

        if scale is None or len(features) == 0:
            return unet_input

        allowed_features = set(features) if len(self.norm_feature) == 0 else set(features).intersection(self.norm_feature)
        if len(allowed_features) == 0:
            return unet_input

        feature_to_norm_idx = {name: i for i, name in enumerate(features)}

        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name not in allowed_features or ch_name not in feature_to_norm_idx:
                continue
            norm_idx = feature_to_norm_idx[ch_name]
            ch_scale = scale[norm_idx]

            if mode == "minmax":
                min_vals = input_norm.get("min")
                if min_vals is None:
                    raise ValueError("input_norm mode='minmax' requires 'min'")
                ch_min = min_vals[norm_idx]
                global_max = torch.max(scale)
                global_max_safe = torch.clamp(global_max, min=1e-8)
                ch_scale_safe = ch_scale if ch_scale != 0 else global_max_safe
                unet_input[:, ch_idx, :, :] = (unet_input[:, ch_idx, :, :] - ch_min) / ch_scale_safe
            else:
                ch_scale_safe = ch_scale if ch_scale != 0 else 10.0
                unet_input[:, ch_idx, :, :] = unet_input[:, ch_idx, :, :] / ch_scale_safe

        return unet_input

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None,
    ) -> wavetypings.StotSpectra:
        spectra_masked = self.spectra_entransform(spectra, mask=mask)

        b, h, w, d, f = spectra_masked.shape
        num_spatial_points = b * h * w

        spectra_points = spectra_masked.reshape(num_spatial_points, 1, d, f)

        main_input = self._build_main_input(spectra_points, environdata, metadata, num_spatial_points)
        wind_input = self._build_wind_input(spectra_points, environdata, metadata, num_spatial_points)

        input_norm = None if environdata is None else environdata.get("input_norm")
        main_input = self._apply_input_norm_channels(main_input, self._channel_names_main, input_norm)
        wind_input = self._apply_input_norm_channels(wind_input, self._channel_names_wind, input_norm)

        stot_main = self.main_unet(main_input).reshape(b, h, w, d, f)
        stot_wind = self.wind_unet(wind_input).reshape(b, h, w, d, f)

        hf_mask = self.high_freq_mask
        if hf_mask.device != stot_main.device:
            hf_mask = hf_mask.to(stot_main.device)
        if hf_mask.dtype != stot_main.dtype:
            hf_mask = hf_mask.to(dtype=stot_main.dtype)

        stot_combined = stot_main + stot_wind * hf_mask
        stot_final = self.spectra_detransform(stot_combined, mask=mask)

        return stot_final * 0.001


@gin.register
class WindFeatMLPUNetParameterization(nn.Module):
    """UNet parameterization with learned wind feature maps via MLP.

    Pipeline:
        1) Raw wind scalars (u, v, etc.) -> MLP -> 1 channel feature map (D x F).
        2) Concatenate spectra + wind feature map + other feature generators.
        3) Run a compact UNet to predict stot on the spectral grid.
    """

    def __init__(
        self,
        spectra_entransform: transforms.BaseEnTransform = gin.REQUIRED,
        spectra_detransform: transforms.BaseDeTransform = gin.REQUIRED,
        spectra_mapper: mappers.BaseSpecMapper = gin.REQUIRED,
        spatial_mapper: mappers.BaseSpatialMapper = gin.REQUIRED,
        feature_generators: Dict[str, embeddings.BaseFeatureGenerator] = gin.REQUIRED,
        windfeat_unet: nn.Module = gin.REQUIRED,
        spec_shape: tuple = (24, 30),
    ):
        super().__init__()
        self.spectra_entransform = spectra_entransform
        self.spectra_detransform = spectra_detransform
        self.spectra_mapper = spectra_mapper
        self.spatial_mapper = spatial_mapper
        self.spec_D, self.spec_F = spec_shape

        self.feature_generators = nn.ModuleDict(feature_generators)
        self.windfeat_unet = windfeat_unet

    def _build_uv_inputs(self, environdata: wavetypings.EnvironmentDataDict, num_spatial_points: int) -> torch.Tensor:
        if environdata is None:
            raise ValueError("environdata is required to extract wind inputs")
        wind_u = environdata.get('wind_u')
        wind_v = environdata.get('wind_v')
        if wind_u is None or wind_v is None:
            raise ValueError("wind_u and wind_v must be present in environdata for WindFeatMLPUNetParameterization")

        wind_u_flat = wind_u.reshape(num_spatial_points)
        wind_v_flat = wind_v.reshape(num_spatial_points)

        # square the wind components to emphasize stronger winds (by Jiang.)
        wind_u_flat = wind_u_flat * wind_u_flat.abs()
        wind_v_flat = wind_v_flat * wind_v_flat.abs()

        uv_inputs = torch.stack([wind_u_flat, wind_v_flat], dim=-1)
        return uv_inputs

    def forward(
        self,
        spectra: wavetypings.FieldSpectra,
        environdata: Optional[wavetypings.EnvironmentDataDict] = None,
        metadata: Optional[wavetypings.MetadataDict] = None,
        mask: Optional[wavetypings.ValidMask] = None,
    ) -> wavetypings.StotSpectra:
        spectra_masked = self.spectra_entransform(spectra, mask=mask)

        b, h, w, d, f = spectra_masked.shape
        num_spatial_points = b * h * w

        spectra_points = spectra_masked.reshape(num_spatial_points, 1, d, f)

        extra_channels = []
        for _, generator in self.feature_generators.items():
            feature_map = generator(spectra_points, environdata, metadata, num_spatial_points)
            extra_channels.append(feature_map)
        extra_channels = torch.cat(extra_channels, dim=1) if len(extra_channels) > 0 else None

        uv_inputs = self._build_uv_inputs(environdata, num_spatial_points)

        stot_image = self.windfeat_unet(spectra_points, uv_inputs, extra_channels)

        stot = stot_image.reshape(b, h, w, d, f)
        stot_final = self.spectra_detransform(stot, mask=mask)

        return stot_final * 0.001

