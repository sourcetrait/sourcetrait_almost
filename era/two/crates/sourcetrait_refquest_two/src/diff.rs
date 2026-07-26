//! The diff verb: nmse and argmax over two same-ids logits dumps.
use crate::*;

/// Run `refquest diff <candidate> <reference>`: one JSON payload line.
pub(crate) fn diff(args: &DiffArgs) -> RefquestResult<()> {
    let candidate_bytes = fs::read(&args.candidate)?;
    let reference_bytes = fs::read(&args.reference)?;
    let candidate = load_dump(&candidate_bytes, &args.candidate)?;
    let reference = load_dump(&reference_bytes, &args.reference)?;

    if candidate.rows != reference.rows || candidate.vocab != reference.vocab {
        snafu::whatever!(
            "shape mismatch: candidate {}x{} vs reference {}x{}",
            candidate.rows,
            candidate.vocab,
            reference.rows,
            reference.vocab
        );
    }
    for name in ["prompt_ids", "fed_ids"] {
        let a = id_bytes(&candidate_bytes, name)?;
        let b = id_bytes(&reference_bytes, name)?;
        if a != b {
            snafu::whatever!("{name} differ - a diff is only valid on same-ids dumps");
        }
    }

    let vocab = candidate.vocab;
    let mut num = 0f64;
    let mut den = 0f64;
    let mut argmax_matches = 0usize;
    let mut first_divergence: Option<usize> = None;
    let mut rows: Vec<(usize, f64, usize, usize)> = Vec::with_capacity(candidate.rows);

    for r in 0..candidate.rows {
        let a = candidate.row(r);
        let b = reference.row(r);
        let mut row_num = 0f64;
        let mut row_den = 0f64;
        let mut argmax_a = 0usize;
        let mut argmax_b = 0usize;
        let mut max_a = f32::NEG_INFINITY;
        let mut max_b = f32::NEG_INFINITY;
        for i in 0..vocab {
            let (av, bv) = (a[i], b[i]);
            let d = (av as f64) - (bv as f64);
            row_num += d * d;
            row_den += (bv as f64) * (bv as f64);
            if av > max_a {
                max_a = av;
                argmax_a = i;
            }
            if bv > max_b {
                max_b = bv;
                argmax_b = i;
            }
        }
        num += row_num;
        den += row_den;
        if argmax_a == argmax_b {
            argmax_matches += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(r);
        }
        let row_nmse = if row_den > 0.0 { row_num / row_den } else { 0.0 };
        rows.push((r, row_nmse, argmax_a, argmax_b));
    }

    rows.sort_by(|x, y| y.1.total_cmp(&x.1));
    let worst: Vec<serde_json::Value> = rows
        .iter()
        .take(args.top)
        .map(|(r, nmse, a, b)| {
            serde_json::json!({ "row": r, "nmse": nmse, "argmax_candidate": a, "argmax_reference": b })
        })
        .collect();

    let payload = serde_json::json!({
        "event": "diff",
        "candidate": args.candidate.display().to_string(),
        "reference": args.reference.display().to_string(),
        "rows": candidate.rows,
        "vocab": vocab,
        "nmse": if den > 0.0 { num / den } else { 0.0 },
        "argmax_matches": argmax_matches,
        "argmax_agreement": (argmax_matches as f64) / (candidate.rows as f64),
        "first_argmax_divergence": first_divergence,
        "worst_rows": worst,
    });
    println!("{payload}");
    Ok(())
}

/// A dump's logits view: f32 rows over the raw file buffer.
struct DumpView<'bytes> {
    data: &'bytes [u8],
    rows: usize,
    vocab: usize,
}

impl DumpView<'_> {
    /// Row as an f32 vec (the buffer carries little-endian f32).
    fn row(&self, r: usize) -> Vec<f32> {
        let stride = self.vocab * 4;
        let start = r * stride;
        self.data[start..start + stride]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn load_dump<'bytes>(bytes: &'bytes [u8], path: &Path) -> RefquestResult<DumpView<'bytes>> {
    let tensors = match safetensors::SafeTensors::deserialize(bytes) {
        Ok(tensors) => tensors,
        Err(e) => snafu::whatever!("{}: not a safetensors file: {e}", path.display()),
    };
    let view = match tensors.tensor("logits") {
        Ok(view) => view,
        Err(e) => snafu::whatever!("{}: no logits tensor: {e}", path.display()),
    };
    if view.dtype() != safetensors::Dtype::F32 {
        snafu::whatever!("{}: logits dtype is {:?}, expected F32", path.display(), view.dtype());
    }
    let shape = view.shape();
    if shape.len() != 2 {
        snafu::whatever!("{}: logits shape {:?}, expected 2-d", path.display(), shape);
    }
    let offset = view.data().as_ptr() as usize - bytes.as_ptr() as usize;
    let len = view.data().len();
    Ok(DumpView {
        data: &bytes[offset..offset + len],
        rows: shape[0],
        vocab: shape[1],
    })
}

fn id_bytes<'bytes>(bytes: &'bytes [u8], name: &str) -> RefquestResult<&'bytes [u8]> {
    let tensors = match safetensors::SafeTensors::deserialize(bytes) {
        Ok(tensors) => tensors,
        Err(e) => snafu::whatever!("not a safetensors file: {e}"),
    };
    let view = match tensors.tensor(name) {
        Ok(view) => view,
        Err(e) => snafu::whatever!("no {name} tensor: {e}"),
    };
    let offset = view.data().as_ptr() as usize - bytes.as_ptr() as usize;
    Ok(&bytes[offset..offset + view.data().len()])
}
