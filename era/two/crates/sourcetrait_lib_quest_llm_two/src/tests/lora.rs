//! LoraPair units: seeded determinism, the zero-contribution init
//! contract, and the contribution/delta arithmetic.
use crate::*;

#[test]
fn lora_pair_inits_deterministic_zero_contributing_and_scaled() {
    type Back = CpuBack;
    let device = Default::default();

    let build = |seed: u64| {
        lora::LoraPair::<Back>::init(8, 6, 4, 8.0, &mut SplitMix64::new(seed), &device)
    };
    let first = build(42);
    let second = build(42);
    let a_first = first.a.clone().into_data().convert::<f32>().to_vec::<f32>().expect("a");
    let a_second = second.a.clone().into_data().convert::<f32>().to_vec::<f32>().expect("a");
    assert_eq!(a_first, a_second, "same seed must init identically");
    assert_eq!(first.scale, 2.0, "alpha 8 over rank 4");

    // The a draws look like N(0, 0.02): near-zero mean, plausible
    // spread (48 draws; loose bounds).
    let mean = a_first.iter().map(|v| *v as f64).sum::<f64>() / a_first.len() as f64;
    let spread = (a_first.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>()
        / a_first.len() as f64)
        .sqrt();
    assert!(mean.abs() < 0.02, "init mean {mean} implausible for N(0, 0.02)");
    assert!((0.005..0.05).contains(&spread), "init spread {spread} implausible");

    // b starts zero, so an untrained pair contributes exactly nothing.
    let x = burn::tensor::Tensor::<Back, 2>::from_data(
        burn::tensor::TensorData::new((0..16).map(|v| v as f32).collect::<Vec<f32>>(), [2, 8]),
        &device,
    );
    let contribution = first
        .contribution(x)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("contribution");
    assert!(contribution.iter().all(|v| *v == 0.0), "untrained pair must contribute zero");
    let delta = first
        .merged_delta()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("delta");
    assert!(delta.iter().all(|v| *v == 0.0), "untrained delta must be zero");
}
