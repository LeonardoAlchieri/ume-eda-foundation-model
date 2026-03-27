import numpy as np
import pytest
import torch
from scipy import signal

from edamame.data_augmentation import matton_augment as np_aug
from edamame.data_augmentation import matton_augment_torch as torch_aug


def _band_power(signal: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / fs)
    spectrum = np.fft.rfft(signal - np.mean(signal))
    power = np.abs(spectrum) ** 2
    mask = (freqs >= f_low) & (freqs < f_high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(power[mask]))


def _summary_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y))
    delta = float(np.mean(np.abs(y - x)))
    return mean, std, delta


@pytest.fixture
def base_signal() -> np.ndarray:
    rng = np.random.default_rng(1234)
    fs = 4.0
    n = 240
    t = np.arange(n) / fs

    x = (
        1.2
        + 0.22 * np.sin(2.0 * np.pi * 0.05 * t)
        + 0.14 * np.sin(2.0 * np.pi * 0.25 * t)
        + 0.08 * np.sin(2.0 * np.pi * 0.75 * t)
        + 0.02 * rng.standard_normal(n)
    )
    x = np.clip(x, 0.0, None)
    return x.astype(np.float32)


@pytest.mark.parametrize(
    "aug_cls, kwargs",
    [
        (torch_aug.GaussianNoiseStochastic, dict(sigma_scale_min=0.0, sigma_scale_max=0.5)),
        (torch_aug.Identity, {}),
        (torch_aug.LowPassFilterStochastic, dict(data_freq=4, highcut_hz_min=0.01, highcut_hz_max=1.0, n_steps=200)),
        (torch_aug.HighPassFilterStochastic, dict(data_freq=4, lowcut_hz_min=0.01, lowcut_hz_max=1.0, n_steps=200)),
        (
            torch_aug.HighFrequencyNoiseStochastic,
            dict(sigma_scale_min=0.0, sigma_scale_max=1.0, freq_bin_start_idx=60, freq_bin_stop_idx=120),
        ),
        (torch_aug.BandstopFilterStochastic, dict(data_freq=4, remove_freq_min=0.01, remove_freq_max=1.0, Q=0.707)),
        (torch_aug.BandpassFilterStochastic, dict(data_freq=4, keep_freq_min=0.01, keep_freq_max=1.0, Q=0.707)),
        (torch_aug.TemporalCutoutStochastic, dict(cutout_size_min=20, cutout_size_max=100)),
        (torch_aug.PermuteStochastic, dict(n_splits_max=10)),
        (torch_aug.Flip, {}),
        (torch_aug.TimeWarpingStochastic, dict(sigma_min=0.01, sigma_max=0.1, knot_min=1, knot_max=4)),
        (torch_aug.AmplitudeWarpingStochastic, dict(sigma_min=0.01, sigma_max=0.05, knot_min=0, knot_max=4)),
        (
            torch_aug.LooseSensorArtifactStochastic,
            dict(width_min=4, width_max=120, smooth_width_min=2, smooth_width_max=20),
        ),
        (
            torch_aug.JumpArtifactStochastic,
            dict(max_n_jumps=2, shift_factor_min=0.01, shift_factor_max=0.2, smooth_width_min=2, smooth_width_max=12),
        ),
        (torch_aug.ConstantAmplitudeScalingStochastic, dict(scale_min=0.5, scale_max=1.5)),
    ],
)
def test_torch_augmentation_output_contract(base_signal, aug_cls, kwargs):
    x_t = torch.from_numpy(base_signal.copy())
    aug = aug_cls(**kwargs)

    y_t = aug(x_t)

    assert isinstance(y_t, torch.Tensor)
    assert y_t.shape == x_t.shape
    assert y_t.device == x_t.device
    assert torch.isfinite(y_t).all()


def test_identity_matches_numpy(base_signal):
    np_out = np_aug.Identity()(base_signal.copy())
    torch_out = torch_aug.Identity()(torch.from_numpy(base_signal.copy())).cpu().numpy()
    np.testing.assert_allclose(torch_out, np_out, atol=1e-7, rtol=1e-7)


def test_flip_matches_numpy(base_signal):
    np_out = np_aug.Flip()(base_signal.copy())
    torch_out = torch_aug.Flip()(torch.from_numpy(base_signal.copy())).cpu().numpy()
    np.testing.assert_allclose(torch_out, np_out, atol=1e-6, rtol=1e-6)


def test_constant_scaling_matches_numpy_when_fixed(base_signal):
    scale = 1.23
    np_out = np_aug.ConstantAmplitudeScalingStochastic(scale_min=scale, scale_max=scale)(base_signal.copy())
    torch_out = torch_aug.ConstantAmplitudeScalingStochastic(scale_min=scale, scale_max=scale)(
        torch.from_numpy(base_signal.copy())
    ).cpu().numpy()
    np.testing.assert_allclose(torch_out, np_out, atol=1e-6, rtol=1e-6)


def test_gaussian_noise_statistical_similarity(base_signal):
    np_transform = np_aug.GaussianNoiseStochastic(sigma_scale_min=0.2, sigma_scale_max=0.2)
    torch_transform = torch_aug.GaussianNoiseStochastic(sigma_scale_min=0.2, sigma_scale_max=0.2)

    np_noise_std = []
    torch_noise_std = []
    x_t = torch.from_numpy(base_signal.copy())

    for seed in range(40):
        np.random.seed(seed)
        torch.manual_seed(seed)

        y_np = np_transform(base_signal.copy())
        y_t = torch_transform(x_t.clone()).cpu().numpy()

        np_noise_std.append(float(np.std(y_np - base_signal)))
        torch_noise_std.append(float(np.std(y_t - base_signal)))

    mean_np = float(np.mean(np_noise_std))
    mean_t = float(np.mean(torch_noise_std))
    ratio = mean_t / (mean_np + 1e-12)

    assert 0.6 <= ratio <= 1.4


def test_lowpass_filter_similarity(base_signal):
    fs = 4.0
    torch_transform = torch_aug.LowPassFilterStochastic(data_freq=fs, highcut_hz_min=0.20, highcut_hz_max=0.20, n_steps=8)

    b, a = signal.butter(4, 0.20, btype="lowpass", output="ba", fs=fs)
    np_out = signal.filtfilt(b, a, base_signal.copy())
    torch_out = torch_transform(torch.from_numpy(base_signal.copy())).cpu().numpy()

    in_high = _band_power(base_signal, fs, 0.45, 2.0)
    np_high = _band_power(np_out, fs, 0.45, 2.0)
    torch_high = _band_power(torch_out, fs, 0.45, 2.0)

    np_ratio = np_high / (in_high + 1e-12)
    torch_ratio = torch_high / (in_high + 1e-12)

    assert np_ratio < 0.9
    assert torch_ratio < 0.9
    assert abs(np_ratio - torch_ratio) < 0.5


def test_highpass_filter_similarity(base_signal):
    fs = 4.0
    torch_transform = torch_aug.HighPassFilterStochastic(data_freq=fs, lowcut_hz_min=0.20, lowcut_hz_max=0.20, n_steps=8)

    b, a = signal.butter(4, 0.20, btype="highpass", output="ba", fs=fs)
    np_out = signal.filtfilt(b, a, base_signal.copy())
    torch_out = torch_transform(torch.from_numpy(base_signal.copy())).cpu().numpy()

    in_low = _band_power(base_signal, fs, 0.0, 0.15)
    np_low = _band_power(np_out, fs, 0.0, 0.15)
    torch_low = _band_power(torch_out, fs, 0.0, 0.15)

    np_ratio = np_low / (in_low + 1e-12)
    torch_ratio = torch_low / (in_low + 1e-12)

    assert np_ratio < 0.9
    assert torch_ratio < 0.9
    assert abs(np_ratio - torch_ratio) < 0.5


def test_high_frequency_noise_similarity(base_signal):
    np_transform = np_aug.HighFrequencyNoiseStochastic(
        sigma_scale_min=0.6,
        sigma_scale_max=0.6,
        freq_bin_start_idx=60,
        freq_bin_stop_idx=120,
    )
    torch_transform = torch_aug.HighFrequencyNoiseStochastic(
        sigma_scale_min=0.6,
        sigma_scale_max=0.6,
        freq_bin_start_idx=60,
        freq_bin_stop_idx=120,
    )

    np_out = np_transform(base_signal.copy())
    torch_out = torch_transform(torch.from_numpy(base_signal.copy())).cpu().numpy()

    def _hf_power(x):
        fft_x = np.fft.fft(x)
        return float(np.mean(np.abs(fft_x[60:120]) ** 2))

    p_in = _hf_power(base_signal)
    p_np = _hf_power(np_out)
    p_t = _hf_power(torch_out)

    assert p_np > p_in
    assert p_t > p_in


def test_bandstop_and_bandpass_similarity(base_signal):
    fs = 4.0

    np_stop = np_aug.BandstopFilterStochastic(data_freq=fs, remove_freq_min=0.25, remove_freq_max=0.25, Q=0.707)
    torch_stop = torch_aug.BandstopFilterStochastic(data_freq=fs, remove_freq_min=0.25, remove_freq_max=0.25, Q=0.707)

    out_np_stop = np_stop(base_signal.copy())
    out_t_stop = torch_stop(torch.from_numpy(base_signal.copy())).cpu().numpy()

    in_band = _band_power(base_signal, fs, 0.18, 0.32)
    np_band = _band_power(out_np_stop, fs, 0.18, 0.32)
    t_band = _band_power(out_t_stop, fs, 0.18, 0.32)

    assert np_band / (in_band + 1e-12) < 1.0
    assert t_band / (in_band + 1e-12) < 1.0

    np_pass = np_aug.BandpassFilterStochastic(data_freq=fs, keep_freq_min=0.25, keep_freq_max=0.25, Q=0.707)
    torch_pass = torch_aug.BandpassFilterStochastic(data_freq=fs, keep_freq_min=0.25, keep_freq_max=0.25, Q=0.707)

    out_np_pass = np_pass(base_signal.copy())
    out_t_pass = torch_pass(torch.from_numpy(base_signal.copy())).cpu().numpy()

    def _share(x):
        band = _band_power(x, fs, 0.18, 0.32)
        total = _band_power(x, fs, 0.0, 2.0)
        return band / (total + 1e-12)

    assert _share(out_np_pass) > _share(base_signal)
    assert _share(out_t_pass) > _share(base_signal)


def test_temporal_cutout_similarity(base_signal):
    x = (base_signal + 2.0).astype(np.float32)
    np_transform = np_aug.TemporalCutoutStochastic(cutout_size_min=30, cutout_size_max=30)
    torch_transform = torch_aug.TemporalCutoutStochastic(cutout_size_min=30, cutout_size_max=30)

    np_zero_counts = []
    torch_zero_counts = []

    for seed in range(20):
        np.random.seed(seed)
        torch.manual_seed(seed)

        y_np = np_transform(x.copy())
        y_t = torch_transform(torch.from_numpy(x.copy())).cpu().numpy()

        np_zero_counts.append(int(np.sum(y_np == 0.0)))
        torch_zero_counts.append(int(np.sum(y_t == 0.0)))

    assert all(count == 30 for count in np_zero_counts)
    assert all(count == 30 for count in torch_zero_counts)


def test_permute_similarity(base_signal):
    np.random.seed(21)
    torch.manual_seed(21)

    np_out = np_aug.PermuteStochastic(n_splits_max=8)(base_signal.copy())
    torch_out = torch_aug.PermuteStochastic(n_splits_max=8)(torch.from_numpy(base_signal.copy())).cpu().numpy()

    np.testing.assert_allclose(np.sort(np_out), np.sort(base_signal), atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.sort(torch_out), np.sort(base_signal), atol=1e-7, rtol=1e-7)

    assert not np.allclose(np_out, base_signal)
    assert not np.allclose(torch_out, base_signal)


@pytest.mark.parametrize(
    "np_transform, torch_transform",
    [
        (
            np_aug.TimeWarpingStochastic(sigma_min=0.05, sigma_max=0.05, knot_min=3, knot_max=3),
            torch_aug.TimeWarpingStochastic(sigma_min=0.05, sigma_max=0.05, knot_min=3, knot_max=3),
        ),
        (
            np_aug.AmplitudeWarpingStochastic(sigma_min=0.04, sigma_max=0.04, knot_min=3, knot_max=3),
            torch_aug.AmplitudeWarpingStochastic(sigma_min=0.04, sigma_max=0.04, knot_min=3, knot_max=3),
        ),
    ],
)
def test_warping_similarity(base_signal, np_transform, torch_transform):
    np_stats = []
    torch_stats = []

    x_t = torch.from_numpy(base_signal.copy())

    for seed in range(30):
        np.random.seed(seed)
        torch.manual_seed(seed)

        y_np = np_transform(base_signal.copy())
        y_t = torch_transform(x_t.clone()).cpu().numpy()

        np_stats.append(_summary_stats(base_signal, y_np))
        torch_stats.append(_summary_stats(base_signal, y_t))

    np_stats = np.array(np_stats)
    torch_stats = np.array(torch_stats)

    np_mean = np.mean(np_stats, axis=0)
    torch_mean = np.mean(torch_stats, axis=0)

    assert np.all(np.isfinite(np_mean))
    assert np.all(np.isfinite(torch_mean))

    rel_diff = np.abs(torch_mean - np_mean) / (np.abs(np_mean) + 1e-6)
    assert np.all(rel_diff < 0.6)


@pytest.mark.parametrize(
    "np_transform, torch_transform",
    [
        (
            np_aug.LooseSensorArtifactStochastic(width_min=24, width_max=24, smooth_width_min=2, smooth_width_max=6),
            torch_aug.LooseSensorArtifactStochastic(width_min=24, width_max=24, smooth_width_min=2, smooth_width_max=6),
        ),
        (
            np_aug.JumpArtifactStochastic(
                max_n_jumps=2,
                shift_factor_min=0.05,
                shift_factor_max=0.05,
                smooth_width_min=2,
                smooth_width_max=6,
            ),
            torch_aug.JumpArtifactStochastic(
                max_n_jumps=2,
                shift_factor_min=0.05,
                shift_factor_max=0.05,
                smooth_width_min=2,
                smooth_width_max=6,
            ),
        ),
    ],
)
def test_artifact_similarity(base_signal, np_transform, torch_transform):
    np_stats = []
    torch_stats = []

    x_t = torch.from_numpy(base_signal.copy())

    for seed in range(25):
        np.random.seed(seed)
        torch.manual_seed(seed)

        y_np = np_transform(base_signal.copy())
        y_t = torch_transform(x_t.clone()).cpu().numpy()

        assert np.min(y_np) >= -1e-6
        assert np.min(y_t) >= -1e-6

        np_stats.append(_summary_stats(base_signal, y_np))
        torch_stats.append(_summary_stats(base_signal, y_t))

    np_stats = np.array(np_stats)
    torch_stats = np.array(torch_stats)

    np_delta = float(np.mean(np_stats[:, 2]))
    torch_delta = float(np.mean(torch_stats[:, 2]))

    ratio = torch_delta / (np_delta + 1e-12)
    assert 0.3 < ratio < 3.0
    assert np_delta > 1e-4
    assert torch_delta > 1e-4
