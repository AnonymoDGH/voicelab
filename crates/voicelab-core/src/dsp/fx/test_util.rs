//! Helpers shared by the effect tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::OnceLock;

use realfft::RealFftPlanner;

/// Counts allocations per thread, so tests running in parallel do not see each other's.
struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn count_allocation() {
    // `try_with`: the allocator also runs while thread-locals are being torn down.
    let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Number of heap allocations `f` makes on the current thread.
pub fn allocations_during(f: impl FnOnce()) -> u64 {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// Small deterministic PRNG (64-bit LCG, high bits out).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32
    }

    /// Uniform in [-1, 1).
    pub fn signed(&mut self) -> f32 {
        2.0 * self.next_f32() - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        self.next_u32() as usize % n
    }

    /// Chunk sizes from single samples up to 10000, the range the engine may use.
    pub fn chunk_size(&mut self) -> usize {
        let max = [8, 256, 2048, 10000][self.below(4)];
        1 + self.below(max)
    }
}

/// Feed `x` through `f` in random chunk sizes, collecting the appended output.
pub fn chunked(x: &[f32], seed: u64, mut f: impl FnMut(&[f32], &mut Vec<f32>)) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < x.len() {
        let n = rng.chunk_size().min(x.len() - pos);
        f(&x[pos..pos + n], &mut out);
        pos += n;
    }
    out
}

/// Process a copy of `x` in place with `f`, in random chunk sizes.
pub fn chunked_in_place(x: &[f32], seed: u64, mut f: impl FnMut(&mut [f32])) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut y = x.to_vec();
    let mut pos = 0;
    while pos < y.len() {
        let n = rng.chunk_size().min(y.len() - pos);
        f(&mut y[pos..pos + n]);
        pos += n;
    }
    y
}

pub fn power_db(x: &[f32]) -> f32 {
    let p = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    (10.0 * p.max(1e-20).log10()) as f32
}

/// Two seconds of real speech at 16 kHz (the golden-test input).
pub fn speech_16k() -> Vec<f32> {
    static SPEECH: OnceLock<Vec<f32>> = OnceLock::new();
    SPEECH
        .get_or_init(|| {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/golden_120ms.safetensors");
            let bytes = std::fs::read(path).expect("speech fixture");
            let st = safetensors::SafeTensors::deserialize(&bytes).expect("fixture format");
            let data = st.tensor("input").expect("input tensor").data().to_vec();
            data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
        })
        .clone()
}

/// Frequency of the strongest spectral peak (Hann window, parabolic interpolation).
pub fn dominant_freq(x: &[f32], sample_rate: u32) -> f32 {
    let n = x.len().next_power_of_two() / 2;
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let mut input: Vec<f32> = x[..n]
        .iter()
        .enumerate()
        .map(|(i, v)| v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).expect("fft sizes");
    let mag: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
    let k = (1..mag.len() - 1).max_by(|&a, &b| mag[a].total_cmp(&mag[b])).expect("non-empty spectrum");
    let (a, b, c) = (mag[k - 1].ln(), mag[k].ln(), mag[k + 1].ln());
    let offset = 0.5 * (a - c) / (a - 2.0 * b + c);
    (k as f32 + offset) * sample_rate as f32 / n as f32
}
