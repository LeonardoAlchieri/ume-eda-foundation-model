import torch


def _prepare_signal_batch(x: torch.Tensor):
    if not torch.is_tensor(x):
        raise TypeError("Input must be a torch.Tensor.")
    if x.ndim not in (1, 2, 3):
        raise ValueError(
            f"Expected shape (T), (B, T), or (B, T, C), got shape {tuple(x.shape)}"
        )

    out_dtype = x.dtype if x.is_floating_point() else torch.float32
    work_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype

    if x.ndim == 1:
        x_bt = x.unsqueeze(0)
        meta = (1, x.shape[0], 1, True)
    elif x.ndim == 2:
        x_bt = x
        meta = (x.shape[0], x.shape[1], 1, False)
    else:
        bsz, t_steps, n_channels = x.shape
        x_bt = x.permute(0, 2, 1).reshape(-1, t_steps)
        meta = (bsz, t_steps, n_channels, False)

    return x_bt.to(dtype=work_dtype), out_dtype, meta


def _restore_signal_batch(x_bt: torch.Tensor, out_dtype: torch.dtype, meta) -> torch.Tensor:
    bsz, t_steps, n_channels, was_1d = meta
    if was_1d:
        return x_bt[0].to(dtype=out_dtype)
    if n_channels == 1:
        return x_bt.to(dtype=out_dtype)
    return (
        x_bt.reshape(bsz, n_channels, t_steps)
        .permute(0, 2, 1)
        .contiguous()
        .to(dtype=out_dtype)
    )


def _randint_inclusive(low: int, high: int, device: torch.device) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), device=device).item())


def _uniform_scalar(low: float, high: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if high <= low:
        return torch.tensor(float(low), device=device, dtype=dtype)
    return torch.empty((), device=device, dtype=dtype).uniform_(low, high)


def _uniform_tensor(
    low: float,
    high: float,
    shape,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if high <= low:
        return torch.full(shape, float(low), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def _linear_interp_1d(query_x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    fp_was_1d = fp.ndim == 1
    if fp_was_1d:
        fp = fp.unsqueeze(0)

    if xp.numel() == 0:
        out = torch.zeros(fp.shape[0], query_x.shape[0], device=fp.device, dtype=fp.dtype)
        return out[0] if fp_was_1d else out
    if xp.numel() == 1:
        out = fp[:, 0:1].expand(fp.shape[0], query_x.shape[0])
        return out[0] if fp_was_1d else out

    sort_idx = torch.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[..., sort_idx]

    keep = torch.ones_like(xp, dtype=torch.bool)
    keep[1:] = xp[1:] > xp[:-1]
    xp = xp[keep]
    fp = fp[..., keep]

    if xp.numel() == 1:
        out = fp[:, 0:1].expand(fp.shape[0], query_x.shape[0])
        return out[0] if fp_was_1d else out

    idx = torch.searchsorted(xp, query_x, right=True)
    idx0 = torch.clamp(idx - 1, 0, xp.numel() - 2)
    idx1 = idx0 + 1

    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[..., idx0]
    y1 = fp[..., idx1]

    w = ((query_x - x0) / torch.clamp(x1 - x0, min=1e-12)).unsqueeze(0)
    out = y0 + w * (y1 - y0)

    out = torch.where(query_x.unsqueeze(0) <= xp[0], fp[:, 0:1], out)
    out = torch.where(query_x.unsqueeze(0) >= xp[-1], fp[:, -1:], out)

    return out[0] if fp_was_1d else out


def _cubic_spline_1d(query_x: torch.Tensor, knot_x: torch.Tensor, knot_y: torch.Tensor) -> torch.Tensor:
    knot_x = knot_x.flatten()
    knot_y = knot_y.flatten()

    if knot_x.numel() < 3:
        return _linear_interp_1d(query_x, knot_x, knot_y)

    sort_idx = torch.argsort(knot_x)
    knot_x = knot_x[sort_idx]
    knot_y = knot_y[sort_idx]

    keep = torch.ones_like(knot_x, dtype=torch.bool)
    keep[1:] = knot_x[1:] > knot_x[:-1]
    knot_x = knot_x[keep]
    knot_y = knot_y[keep]

    n = knot_x.numel()
    if n < 3:
        return _linear_interp_1d(query_x, knot_x, knot_y)

    h = knot_x[1:] - knot_x[:-1]
    if torch.any(h <= 0):
        return _linear_interp_1d(query_x, knot_x, knot_y)

    a = torch.zeros((n, n), device=knot_x.device, dtype=knot_y.dtype)
    rhs = torch.zeros(n, device=knot_x.device, dtype=knot_y.dtype)

    a[0, 0] = 1.0
    a[-1, -1] = 1.0

    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1]
        a[i, i] = 2.0 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i]
        rhs[i] = 6.0 * (
            (knot_y[i + 1] - knot_y[i]) / h[i]
            - (knot_y[i] - knot_y[i - 1]) / h[i - 1]
        )

    try:
        m = torch.linalg.solve(a, rhs)
    except RuntimeError:
        return _linear_interp_1d(query_x, knot_x, knot_y)

    interval_idx = torch.searchsorted(knot_x, query_x, right=True) - 1
    interval_idx = torch.clamp(interval_idx, 0, n - 2)

    x0 = knot_x[interval_idx]
    x1 = knot_x[interval_idx + 1]
    y0 = knot_y[interval_idx]
    y1 = knot_y[interval_idx + 1]
    m0 = m[interval_idx]
    m1 = m[interval_idx + 1]

    hi = x1 - x0
    a_w = (x1 - query_x) / hi
    b_w = (query_x - x0) / hi

    out = (
        a_w * y0
        + b_w * y1
        + (((a_w**3 - a_w) * m0 + (b_w**3 - b_w) * m1) * (hi**2) / 6.0)
    )

    out = torch.where(query_x <= knot_x[0], knot_y[0], out)
    out = torch.where(query_x >= knot_x[-1], knot_y[-1], out)
    return out


def _fft_filter(x: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    x_fft = torch.fft.fft(x, dim=-1)
    y = torch.fft.ifft(x_fft * response.to(dtype=x_fft.dtype), dim=-1).real
    return y


def _fft_lowpass(x: torch.Tensor, cutoff_hz: torch.Tensor, data_freq: float, order: int = 4) -> torch.Tensor:
    cutoff = torch.clamp(cutoff_hz.to(device=x.device, dtype=x.dtype), min=1e-6)
    freqs = torch.fft.fftfreq(x.shape[-1], d=1.0 / data_freq, device=x.device).to(dtype=x.dtype)
    abs_freq = torch.abs(freqs).unsqueeze(0)
    response = 1.0 / torch.sqrt(1.0 + (abs_freq / cutoff) ** (2 * order))
    return _fft_filter(x, response)


def _fft_highpass(x: torch.Tensor, cutoff_hz: torch.Tensor, data_freq: float, order: int = 4) -> torch.Tensor:
    cutoff = torch.clamp(cutoff_hz.to(device=x.device, dtype=x.dtype), min=1e-6)
    freqs = torch.fft.fftfreq(x.shape[-1], d=1.0 / data_freq, device=x.device).to(dtype=x.dtype)
    abs_freq = torch.abs(freqs).unsqueeze(0)
    response = 1.0 / torch.sqrt(
        1.0 + (cutoff / torch.clamp(abs_freq, min=1e-6)) ** (2 * order)
    )
    response = torch.where(abs_freq < 1e-6, torch.zeros_like(response), response)
    return _fft_filter(x, response)


def _fft_peak_response(x: torch.Tensor, center_hz: torch.Tensor, q: float, data_freq: float) -> torch.Tensor:
    center_hz = torch.clamp(center_hz.to(device=x.device, dtype=x.dtype), min=1e-6)
    q = max(float(q), 1e-6)
    bw = torch.clamp(center_hz / q, min=1e-6)

    freqs = torch.fft.fftfreq(x.shape[-1], d=1.0 / data_freq, device=x.device).to(dtype=x.dtype)
    abs_freq = torch.abs(freqs).unsqueeze(0)
    norm_dist = (abs_freq - center_hz) / (bw / 2.0 + 1e-6)
    response = 1.0 / (1.0 + norm_dist**2)
    return response


class GaussianNoiseStochastic:
    def __init__(self, sigma_scale_min=0.0, sigma_scale_max=0.5):
        self.sigma_scale_min = sigma_scale_min
        self.sigma_scale_max = sigma_scale_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]

        sigma_scale = _uniform_tensor(
            self.sigma_scale_min,
            self.sigma_scale_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        mean_power_diff = torch.mean(
            torch.abs(x_bt - torch.mean(x_bt, dim=-1, keepdim=True)),
            dim=-1,
            keepdim=True,
        )
        noise_sigma = mean_power_diff * sigma_scale
        noise = torch.randn_like(x_bt) * noise_sigma
        y_bt = x_bt + noise
        return _restore_signal_batch(y_bt, out_dtype, meta)


class Identity:
    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        return _restore_signal_batch(x_bt, out_dtype, meta)


class LowPassFilterStochastic:
    def __init__(self, data_freq=4, highcut_hz_min=0.01, highcut_hz_max=1, n_steps=1000):
        self.data_freq = data_freq
        self.highcut_hz_min = highcut_hz_min
        self.highcut_hz_max = highcut_hz_max
        self.n_steps = n_steps

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]
        cutoff = _uniform_tensor(
            self.highcut_hz_min,
            self.highcut_hz_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        y_bt = _fft_lowpass(x_bt, cutoff, self.data_freq, order=4)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class HighPassFilterStochastic:
    def __init__(self, data_freq=4, lowcut_hz_min=0.01, lowcut_hz_max=1, n_steps=1000):
        self.data_freq = data_freq
        self.lowcut_hz_min = lowcut_hz_min
        self.lowcut_hz_max = lowcut_hz_max
        self.n_steps = n_steps

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]
        cutoff = _uniform_tensor(
            self.lowcut_hz_min,
            self.lowcut_hz_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        y_bt = _fft_highpass(x_bt, cutoff, self.data_freq, order=4)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class HighFrequencyNoiseStochastic:
    def __init__(self, sigma_scale_min=0.0, sigma_scale_max=1.0, freq_bin_start_idx=60, freq_bin_stop_idx=120):
        self.sigma_scale_min = sigma_scale_min
        self.sigma_scale_max = sigma_scale_max
        self.freq_bin_start_idx = freq_bin_start_idx
        self.req_bin_stop_idx = freq_bin_stop_idx

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series, n = x_bt.shape

        idxs = torch.arange(self.freq_bin_start_idx, self.req_bin_stop_idx, device=x_bt.device)
        idxs = idxs[(idxs > 0) & (idxs < n)]
        if idxs.numel() == 0:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        x_fft = torch.fft.fft(x_bt, dim=-1)
        mean_fft_val = torch.mean(torch.abs(x_fft), dim=-1, keepdim=True).to(dtype=x_bt.dtype)
        sigma_scale = _uniform_tensor(
            self.sigma_scale_min,
            self.sigma_scale_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        sigma = sigma_scale * mean_fft_val
        noise = torch.randn(n_series, idxs.numel(), device=x_bt.device, dtype=x_bt.dtype) * sigma

        x_fft[:, idxs] = x_fft[:, idxs] + noise.to(dtype=x_fft.dtype)

        neg_idxs = torch.remainder(-idxs, n)
        x_fft[:, neg_idxs] = x_fft[:, neg_idxs] + noise.to(dtype=x_fft.dtype)

        y_bt = torch.abs(torch.fft.ifft(x_fft, dim=-1))
        return _restore_signal_batch(y_bt, out_dtype, meta)


class BandstopFilterStochastic:
    def __init__(self, data_freq=4, remove_freq_min=0.01, remove_freq_max=1.0, Q=0.707):
        self.data_freq = data_freq
        self.remove_feq_min = remove_freq_min
        self.remove_freq_max = remove_freq_max
        self.Q = Q

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]
        remove_freq = _uniform_tensor(
            self.remove_feq_min,
            self.remove_freq_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        peak = _fft_peak_response(x_bt, remove_freq, self.Q, self.data_freq)
        response = torch.clamp(1.0 - peak, min=0.0, max=1.0)
        y_bt = _fft_filter(x_bt, response)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class BandpassFilterStochastic:
    def __init__(self, data_freq=4, keep_freq_min=0.01, keep_freq_max=1, Q=0.707):
        self.data_freq = data_freq
        self.keep_freq_min = keep_freq_min
        self.keep_freq_max = keep_freq_max
        self.Q = Q

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]
        keep_freq = _uniform_tensor(
            self.keep_freq_min,
            self.keep_freq_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        response = _fft_peak_response(x_bt, keep_freq, self.Q, self.data_freq)
        y_bt = _fft_filter(x_bt, response)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class TemporalCutoutStochastic:
    def __init__(self, cutout_size_min=1, cutout_size_max=120):
        self.cutout_size_min = cutout_size_min
        self.cutout_size_max = cutout_size_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series, n = x_bt.shape

        cutout_min = max(1, min(self.cutout_size_min, n))
        cutout_max = max(cutout_min, min(self.cutout_size_max, n))

        cutout_size = torch.randint(
            cutout_min,
            cutout_max + 1,
            (n_series,),
            device=x_bt.device,
        )
        start_max = n - cutout_size
        cutout_start = torch.floor(
            torch.rand(n_series, device=x_bt.device) * (start_max.to(dtype=x_bt.dtype) + 1.0)
        ).to(dtype=torch.long)

        time_idx = torch.arange(n, device=x_bt.device).unsqueeze(0)
        mask = (time_idx >= cutout_start.unsqueeze(1)) & (
            time_idx < (cutout_start + cutout_size).unsqueeze(1)
        )

        y_bt = x_bt.clone()
        y_bt[mask] = 0
        return _restore_signal_batch(y_bt, out_dtype, meta)


class PermuteStochastic:
    def __init__(self, n_splits_max=10):
        self.n_splits_max = n_splits_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n = x_bt.shape[1]

        if n < 2:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        high = max(3, self.n_splits_max)
        num_splits = int(torch.randint(2, high, (1,), device=x_bt.device).item())

        orig_steps = torch.arange(n, device=x_bt.device)
        splits = torch.tensor_split(orig_steps, num_splits)
        perm = torch.randperm(len(splits), device=x_bt.device)

        warp_idx = torch.cat([splits[i] for i in perm.tolist()])
        y_bt = x_bt[:, warp_idx]
        return _restore_signal_batch(y_bt, out_dtype, meta)


class Flip:
    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        y_bt = (-1.0 * x_bt) + (2.0 * torch.mean(x_bt, dim=-1, keepdim=True))
        return _restore_signal_batch(y_bt, out_dtype, meta)


class TimeWarpingStochastic:
    def __init__(self, sigma_min=0.01, sigma_max=0.25, knot_min=1, knot_max=4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.knot_min = knot_min
        self.knot_max = knot_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n = x_bt.shape[1]
        if n < 3:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        sigma = _uniform_scalar(self.sigma_min, self.sigma_max, x_bt.device, x_bt.dtype)
        knot = _randint_inclusive(self.knot_min, self.knot_max, x_bt.device)

        orig_steps = torch.arange(n, device=x_bt.device, dtype=x_bt.dtype)
        warp_steps = torch.linspace(0, n - 1, steps=knot + 2, device=x_bt.device, dtype=x_bt.dtype)
        random_warps = torch.randn(knot + 2, device=x_bt.device, dtype=x_bt.dtype) * sigma + 1.0

        time_warp = _cubic_spline_1d(orig_steps, warp_steps, warp_steps * random_warps)
        if not torch.isfinite(time_warp[-1]) or torch.abs(time_warp[-1]) < 1e-6:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        scale = (n - 1) / time_warp[-1]
        warped_steps = torch.clamp(scale * time_warp, 0, n - 1)

        y_bt = _linear_interp_1d(orig_steps, warped_steps, x_bt)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class AmplitudeWarpingStochastic:
    def __init__(self, sigma_min=0.01, sigma_max=0.25, knot_min=0, knot_max=4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.knot_min = knot_min
        self.knot_max = knot_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n = x_bt.shape[1]
        if n < 3:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        sigma = _uniform_scalar(self.sigma_min, self.sigma_max, x_bt.device, x_bt.dtype)
        knot = _randint_inclusive(self.knot_min, self.knot_max, x_bt.device)

        orig_steps = torch.arange(n, device=x_bt.device, dtype=x_bt.dtype)
        warp_steps = torch.linspace(0, n - 1, steps=knot + 2, device=x_bt.device, dtype=x_bt.dtype)
        random_warps = torch.randn(knot + 2, device=x_bt.device, dtype=x_bt.dtype) * sigma + 1.0

        warper = _cubic_spline_1d(orig_steps, warp_steps, random_warps)
        y_bt = x_bt * warper.unsqueeze(0)
        return _restore_signal_batch(y_bt, out_dtype, meta)


class LooseSensorArtifactStochastic:
    def __init__(self, width_min=4, width_max=120, smooth_width_min=2, smooth_width_max=20):
        self.width_min = width_min
        self.width_max = width_max
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n = x_bt.shape[1]
        if n < 4:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        width_min = max(1, min(self.width_min, n))
        width_max = max(width_min, min(self.width_max, n))
        artifact_width = _randint_inclusive(width_min, width_max, x_bt.device)

        artifact_start = _randint_inclusive(0, n - artifact_width, x_bt.device)
        artifact_end = artifact_start + artifact_width - 1

        smooth_left = artifact_start != 0
        smooth_right = artifact_end != n - 1

        smooth_available = (artifact_width - 2) // 2
        smooth_max = min(self.smooth_width_max, smooth_available)
        smooth_min = min(smooth_max, self.smooth_width_min)

        if smooth_left and smooth_max >= smooth_min:
            smooth_width1 = _randint_inclusive(smooth_min, smooth_max, x_bt.device)
        else:
            smooth_width1 = 0

        if smooth_right and smooth_max >= smooth_min:
            smooth_width2 = _randint_inclusive(smooth_min, smooth_max, x_bt.device)
        else:
            smooth_width2 = 0

        y_bt = x_bt.clone()
        drop_start = artifact_start + smooth_width1
        drop_end = artifact_end - smooth_width2

        if drop_end < drop_start:
            return _restore_signal_batch(y_bt, out_dtype, meta)

        mean_amp = torch.mean(y_bt[:, drop_start : drop_end + 1], dim=-1, keepdim=True)
        y_bt[:, drop_start : drop_end + 1] = (
            y_bt[:, drop_start : drop_end + 1] - mean_amp
        )
        y_bt = torch.clamp_min(y_bt, 0.0)

        coord_dtype = x_bt.dtype
        train_x = torch.cat(
            [
                torch.arange(artifact_start, device=x_bt.device, dtype=coord_dtype),
                torch.arange(drop_start, drop_end + 1, device=x_bt.device, dtype=coord_dtype),
                torch.arange(artifact_end + 1, n, device=x_bt.device, dtype=coord_dtype),
            ]
        )
        train_y = torch.cat(
            [
                y_bt[:, :artifact_start],
                y_bt[:, drop_start : drop_end + 1],
                y_bt[:, artifact_end + 1 :],
            ],
            dim=-1,
        )

        if train_x.numel() >= 2:
            if artifact_start != drop_start:
                query = torch.arange(
                    artifact_start,
                    drop_start,
                    device=x_bt.device,
                    dtype=coord_dtype,
                )
                y_bt[:, artifact_start:drop_start] = _linear_interp_1d(query, train_x, train_y)

            if artifact_end != drop_end:
                query = torch.arange(
                    drop_end + 1,
                    artifact_end + 1,
                    device=x_bt.device,
                    dtype=coord_dtype,
                )
                y_bt[:, drop_end + 1 : artifact_end + 1] = _linear_interp_1d(
                    query,
                    train_x,
                    train_y,
                )

        return _restore_signal_batch(y_bt, out_dtype, meta)


class JumpArtifactStochastic:
    def __init__(self, max_n_jumps=2, shift_factor_min=0.01, shift_factor_max=0.2, smooth_width_min=2, smooth_width_max=12):
        self.max_n_jumps = max_n_jumps
        self.shift_factor_min = shift_factor_min
        self.shift_factor_max = shift_factor_max
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n = x_bt.shape[1]
        if n < self.smooth_width_min + 3:
            return _restore_signal_batch(x_bt, out_dtype, meta)

        y_bt = x_bt.clone()

        time_flip = -1 if torch.rand((), device=x_bt.device) < 0.5 else 1
        if time_flip == -1:
            y_bt = torch.flip(y_bt, dims=[1])

        min_start = 1
        max_start = n - self.smooth_width_min - 2
        if max_start < min_start:
            if time_flip == -1:
                y_bt = torch.flip(y_bt, dims=[1])
            return _restore_signal_batch(y_bt, out_dtype, meta)

        candidates = torch.arange(min_start, max_start + 1, device=x_bt.device)
        n_available = candidates.numel()
        n_jumps_max = min(self.max_n_jumps, n_available)
        n_jumps = _randint_inclusive(1, n_jumps_max, x_bt.device)

        perm = torch.randperm(n_available, device=x_bt.device)
        artifact_starts = torch.sort(candidates[perm[:n_jumps]])[0]

        shifts = torch.empty(n_jumps, device=x_bt.device, dtype=x_bt.dtype).uniform_(
            self.shift_factor_min, self.shift_factor_max
        )
        signs = (torch.randint(0, 2, (n_jumps,), device=x_bt.device) * 2 - 1).to(dtype=x_bt.dtype)
        artifact_shift_factors = shifts * signs

        coord_dtype = x_bt.dtype
        for idx, a_start_t in enumerate(artifact_starts):
            a_start = int(a_start_t.item())
            smooth_max_local = min(self.smooth_width_max, n - a_start - 2)
            if smooth_max_local < self.smooth_width_min:
                continue

            a_smooth_win = _randint_inclusive(self.smooth_width_min, smooth_max_local, x_bt.device)
            post_start = a_start + a_smooth_win

            y_bt[:, post_start:] = y_bt[:, post_start:] + artifact_shift_factors[idx] * (
                a_smooth_win / 4.0
            )

            train_x = torch.cat(
                [
                    torch.arange(a_start, device=x_bt.device, dtype=coord_dtype),
                    torch.arange(
                        a_start + a_smooth_win,
                        n,
                        device=x_bt.device,
                        dtype=coord_dtype,
                    ),
                ]
            )
            train_y = torch.cat(
                [
                    y_bt[:, :a_start],
                    y_bt[:, a_start + a_smooth_win :],
                ],
                dim=-1,
            )

            if train_x.numel() >= 2:
                query = torch.arange(
                    a_start,
                    a_start + a_smooth_win,
                    device=x_bt.device,
                    dtype=coord_dtype,
                )
                y_bt[:, a_start : a_start + a_smooth_win] = _linear_interp_1d(
                    query,
                    train_x,
                    train_y,
                )

            y_bt = torch.clamp_min(y_bt, 0.0)

        if time_flip == -1:
            y_bt = torch.flip(y_bt, dims=[1])

        return _restore_signal_batch(y_bt, out_dtype, meta)


class ConstantAmplitudeScalingStochastic:
    def __init__(self, scale_min=0.5, scale_max=1.5):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, x):
        x_bt, out_dtype, meta = _prepare_signal_batch(x)
        n_series = x_bt.shape[0]
        scale = _uniform_tensor(
            self.scale_min,
            self.scale_max,
            (n_series, 1),
            x_bt.device,
            x_bt.dtype,
        )
        y_bt = x_bt * scale
        return _restore_signal_batch(y_bt, out_dtype, meta)


__all__ = [
    "GaussianNoiseStochastic",
    "Identity",
    "LowPassFilterStochastic",
    "HighPassFilterStochastic",
    "HighFrequencyNoiseStochastic",
    "BandstopFilterStochastic",
    "BandpassFilterStochastic",
    "TemporalCutoutStochastic",
    "PermuteStochastic",
    "Flip",
    "TimeWarpingStochastic",
    "AmplitudeWarpingStochastic",
    "LooseSensorArtifactStochastic",
    "JumpArtifactStochastic",
    "ConstantAmplitudeScalingStochastic",
]
