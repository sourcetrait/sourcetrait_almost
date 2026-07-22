use crate::*;

/// Compare the "logits" tensors of two parity dumps: overall NMSE
/// (sum((a-b)^2)/sum(b^2), b the reference), per-row max-abs, and per-row
/// argmax agreement, with the worst rows listed.
pub(crate) fn diff(candidate_path: &Path, reference_path: &Path, top: usize) -> HeatResult<()> {
    let candidate_bytes = std::fs::read(candidate_path)?;
    let candidate_file = safetensors::SafeTensors::deserialize(&candidate_bytes)?;
    let (rows_a, vocab_a, candidate) = tensor_io::read_f32_matrix(&candidate_file, "logits")?;
    let reference_bytes = std::fs::read(reference_path)?;
    let reference_file = safetensors::SafeTensors::deserialize(&reference_bytes)?;
    let (rows_b, vocab_b, reference) = tensor_io::read_f32_matrix(&reference_file, "logits")?;
    snafu::ensure_whatever!(
        rows_a == rows_b && vocab_a == vocab_b,
        "shape mismatch: ({rows_a}, {vocab_a}) vs ({rows_b}, {vocab_b})"
    );

    let mut diff_sq_total = 0f64;
    let mut reference_sq_total = 0f64;
    let mut row_reports: Vec<(usize, f32, bool)> = Vec::with_capacity(rows_a);
    let mut argmax_matches = 0usize;
    for row in 0..rows_a {
        let range = row * vocab_a..(row + 1) * vocab_a;
        let a = &candidate[range.clone()];
        let b = &reference[range];
        let mut max_abs = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let difference = (x - y).abs();
            if difference > max_abs {
                max_abs = difference;
            }
            diff_sq_total += (difference as f64) * (difference as f64);
            reference_sq_total += (*y as f64) * (*y as f64);
        }
        let agree = argmax(a) == argmax(b);
        if agree {
            argmax_matches += 1;
        }
        row_reports.push((row, max_abs, agree));
    }
    let nmse = if reference_sq_total > 0.0 {
        diff_sq_total / reference_sq_total
    } else {
        0.0
    };

    println!(
        "rows {rows_a} | vocab {vocab_a} | nmse {nmse:.3e} | argmax agreement {}/{} ({:.4})",
        argmax_matches,
        rows_a,
        argmax_matches as f64 / rows_a as f64
    );
    row_reports.sort_by(|left, right| right.1.total_cmp(&left.1));
    println!("worst rows by max_abs:");
    for (row, max_abs, agree) in row_reports.iter().take(top) {
        println!(
            "  row {row} | max_abs {max_abs:.4e} | argmax {}",
            if *agree { "agrees" } else { "DISAGREES" }
        );
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}
