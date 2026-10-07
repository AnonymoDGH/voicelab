pub mod fbank;
pub mod istft;
pub mod resample;

/// `F.interpolate(x, size, mode="linear", align_corners=True)` along time.
/// `x` is `[len, dim]` row-major; returns `[size, dim]`.
pub fn interp_linear(x: &[f32], dim: usize, size: usize) -> Vec<f32> {
    let len = x.len() / dim;
    let mut out = vec![0.0; size * dim];
    for i in 0..size {
        let pos = i as f64 * (len - 1) as f64 / (size - 1) as f64;
        let i0 = (pos.floor() as usize).min(len - 1);
        let i1 = (i0 + 1).min(len - 1);
        let w = (pos - i0 as f64) as f32;
        let (a, b) = (&x[i0 * dim..(i0 + 1) * dim], &x[i1 * dim..(i1 + 1) * dim]);
        for (o, (&a, &b)) in out[i * dim..(i + 1) * dim].iter_mut().zip(a.iter().zip(b)) {
            *o = a * (1.0 - w) + b * w;
        }
    }
    out
}
