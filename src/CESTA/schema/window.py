"""Windowing and data split schema definitions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WindowConfig(BaseModel):
    """Configuration for sliding window dataset creation.

    Attributes:
        window_size: Number of timesteps per window.
        train_stride: Stride for training windows (allows overlap).
        test_stride: Stride for validation/testing windows (typically no overlap).
    """

    model_config = ConfigDict(frozen=True)

    window_size: int = Field(default=60, ge=1)
    train_stride: int = Field(default=10, ge=1)
    test_stride: int = Field(default=60, ge=1)


class DataSplitConfig(BaseModel):
    """Configuration for train/validation/test partitioning."""

    model_config = ConfigDict(frozen=True)

    strategy: Literal["chronological", "connectivity-chronological"] = "chronological"
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    tolerance: float = Field(default=0.05, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_split_ratios(self) -> "DataSplitConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            msg = f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.6f}"
            raise ValueError(msg)
        if self.strategy == "connectivity-chronological" and self.val_ratio <= 0.0:
            msg = "connectivity-chronological requires val_ratio > 0.0"
            raise ValueError(msg)
        return self


class DataConfig(BaseModel):
    """Configuration for train-time data preparation."""

    model_config = ConfigDict(frozen=True)

    window: WindowConfig = Field(default_factory=WindowConfig)
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
