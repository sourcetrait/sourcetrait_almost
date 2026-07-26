//! Instruction-shaped training examples: the on-disk shapes, their
//! ingest, and the packing that turns them into trainable rows.
//!
//! The continued-pretraining path packs a DOCUMENT stream - join every
//! document, cut fixed windows, shuffle. That is wrong for an
//! instruction example, because the cut falls wherever it falls and
//! would split a prompt from its response into two independently
//! shuffled rows. So these pack ONE EXAMPLE PER ROW: render, encode,
//! mark the assistant positions, pad to the window, and keep the
//! example whole or drop it.
//!
//! ## DEV
//! The loss mask rides a SECOND tensor beside the ids, which is why
//! `train::load_chunks` had to start rejecting tensors it does not
//! recognise: it previously read one named tensor and ignored the
//! rest, so a mask written beside the ids would have been skipped in
//! silence and the run would have completed as an unmasked one.
//! ##
// The stage verbs (cfg train) and the rollout verb are the consumers;
// a non-train build sees the ingest side alone. The allow retires
// when every stage is wired.
#![allow(dead_code)]
use crate::*;

/// A supervised example: one conversation whose assistant turns are
/// what the model must learn to produce.
pub(crate) const SFT_TYPEDEF: &str =
    "table<messages: table<role: string, content: string>>";

/// A preference pair: a shared prompt and two candidate assistant
/// replies, the first preferred.
pub(crate) const DPO_TYPEDEF: &str = "table<prompt: table<role: string, content: string>, \
     chosen: string, rejected: string>";

/// A verifiable prompt: what to ask, which verifier grades the reply,
/// and the reference that verifier compares against.
pub(crate) const RLVR_TYPEDEF: &str = "table<prompt: table<role: string, content: string>, \
     verifier: string, reference: string>";

/// The pack artifact's tensors.
pub(crate) const TENSOR_IDS: &str = "chunks";
pub(crate) const TENSOR_LOSS_MASK: &str = "loss_mask";

/// One preference pair, messages already parsed.
pub(crate) struct DpoExample {
    pub(crate) prompt: Vec<lib::ChatMessage>,
    pub(crate) chosen: String,
    pub(crate) rejected: String,
}

/// One verifiable prompt, messages already parsed.
pub(crate) struct RlvrExample {
    pub(crate) prompt: Vec<lib::ChatMessage>,
    pub(crate) verifier: String,
    pub(crate) reference: String,
}

/// One packed row: the window's ids and the per-position loss mask
/// (1 where the model must produce that token).
pub(crate) struct PackedRow {
    pub(crate) ids: Vec<u32>,
    pub(crate) mask: Vec<u8>,
}

/// Read a message table into chat messages, rejecting any role the
/// renderer does not know rather than dropping it.
fn read_messages(rows: &[&lib::nu::Record]) -> BquestResult<Vec<lib::ChatMessage>> {
    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
        let role = lib::ChatRole::parse(&field_str(row, "role")?)?;
        messages.push(lib::ChatMessage::new(role, &field_str(row, "content")?));
    }
    Ok(messages)
}

fn load_table(path: &Path, typedef: &str) -> BquestResult<Vec<lib::nu::Value>> {
    let value = lib::nu::load_value(path)?;
    lib::nu::conform(&value, &lib::nu::parse_typedef(typedef)?)?;
    let rows = match value.as_list() {
        Ok(rows) => rows.to_vec(),
        Err(e) => snafu::whatever!("{}: not a table: {e}", path.display()),
    };
    snafu::ensure_whatever!(!rows.is_empty(), "{}: no examples", path.display());
    Ok(rows)
}

fn as_record(value: &lib::nu::Value) -> BquestResult<&lib::nu::Record> {
    match value.as_record() {
        Ok(record) => Ok(record),
        Err(e) => snafu::whatever!("example row: {e}"),
    }
}

/// Load supervised examples: one message list per row.
pub(crate) fn load_sft(path: &Path) -> BquestResult<Vec<Vec<lib::ChatMessage>>> {
    let mut examples = Vec::new();
    for row in load_table(path, SFT_TYPEDEF)? {
        let record = as_record(&row)?;
        let messages = read_messages(&field_rows(record, "messages")?)?;
        snafu::ensure_whatever!(
            messages.iter().any(|m| m.role == Some(lib::ChatRole::Assistant)),
            "a supervised example carries no assistant turn, so it would train nothing"
        );
        examples.push(messages);
    }
    Ok(examples)
}

/// Load preference pairs.
pub(crate) fn load_dpo(path: &Path) -> BquestResult<Vec<DpoExample>> {
    let mut examples = Vec::new();
    for row in load_table(path, DPO_TYPEDEF)? {
        let record = as_record(&row)?;
        examples.push(DpoExample {
            prompt: read_messages(&field_rows(record, "prompt")?)?,
            chosen: field_str(record, "chosen")?,
            rejected: field_str(record, "rejected")?,
        });
    }
    Ok(examples)
}

/// Load verifiable prompts.
pub(crate) fn load_rlvr(path: &Path) -> BquestResult<Vec<RlvrExample>> {
    let mut examples = Vec::new();
    for row in load_table(path, RLVR_TYPEDEF)? {
        let record = as_record(&row)?;
        let example = RlvrExample {
            prompt: read_messages(&field_rows(record, "prompt")?)?,
            verifier: field_str(record, "verifier")?,
            reference: field_str(record, "reference")?,
        };
        verifier_known(&example.verifier)?;
        examples.push(example);
    }
    Ok(examples)
}

/// Render + encode a conversation, marking every assistant position.
/// The span boundaries come from the renderer rather than from
/// re-measuring a prefix, so they are exact by construction.
pub(crate) fn encode_supervised(
    tokenizer: &tokenizers::Tokenizer,
    messages: &[lib::ChatMessage],
) -> BquestResult<(Vec<u32>, Vec<u8>)> {
    let render = lib::chat_render(messages, false);
    let encoded = lib::encode_render(tokenizer, &render)?;
    let mut mask = vec![0u8; encoded.ids.len()];
    for span in &encoded.assistant_spans {
        for slot in mask[span.start..span.end].iter_mut() {
            *slot = 1;
        }
    }
    Ok((encoded.ids, mask))
}

/// Render a prompt with the assistant opener appended - what a
/// rollout is fed and what a preference candidate continues from.
pub(crate) fn encode_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &[lib::ChatMessage],
) -> BquestResult<Vec<u32>> {
    let render = lib::chat_render(prompt, true);
    Ok(lib::encode_render(tokenizer, &render)?.ids)
}

/// Pad one encoded example to `width`, or report it as too long.
/// Truncation is deliberately not offered: a clipped response trains
/// the model to stop mid-answer, and dropping is visible where
/// clipping is not.
pub(crate) fn pack_row(
    ids: Vec<u32>,
    mask: Vec<u8>,
    width: usize,
    pad_id: u32,
) -> Option<PackedRow> {
    if ids.len() > width || ids.is_empty() {
        return None;
    }
    let mut ids = ids;
    let mut mask = mask;
    ids.resize(width, pad_id);
    mask.resize(width, 0);
    Some(PackedRow { ids, mask })
}

/// Write a packed set as the two-tensor artifact plus its provenance
/// sidecar. Rows are `width` wide; the trainer takes ids[..width-1]
/// as inputs and ids[1..] as targets under mask[1..].
pub(crate) fn write_pack(
    out: &Path,
    rows: &[PackedRow],
    width: usize,
    provenance: lib::nu::Record,
) -> BquestResult<()> {
    snafu::ensure_whatever!(!rows.is_empty(), "no rows survived packing");
    let mut id_bytes: Vec<u8> = Vec::with_capacity(rows.len() * width * 4);
    let mut mask_bytes: Vec<u8> = Vec::with_capacity(rows.len() * width);
    for row in rows {
        snafu::ensure_whatever!(
            row.ids.len() == width && row.mask.len() == width,
            "packed row width mismatch"
        );
        for id in &row.ids {
            id_bytes.extend_from_slice(&id.to_le_bytes());
        }
        mask_bytes.extend_from_slice(&row.mask);
    }
    let id_view = match safetensors::tensor::TensorView::new(
        safetensors::Dtype::U32,
        vec![rows.len(), width],
        &id_bytes,
    ) {
        Ok(view) => view,
        Err(e) => snafu::whatever!("ids view failed: {e}"),
    };
    let mask_view = match safetensors::tensor::TensorView::new(
        safetensors::Dtype::U8,
        vec![rows.len(), width],
        &mask_bytes,
    ) {
        Ok(view) => view,
        Err(e) => snafu::whatever!("mask view failed: {e}"),
    };
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    match safetensors::serialize_to_file(
        vec![
            (String::from(TENSOR_IDS), id_view),
            (String::from(TENSOR_LOSS_MASK), mask_view),
        ],
        None,
        out,
    ) {
        Ok(()) => {}
        Err(e) => snafu::whatever!("pack write failed: {e}"),
    }
    lib::nu::save_value(
        &out.with_extension("nuon"),
        &lib::nu::Value::record(provenance, span()),
    )?;
    Ok(())
}

// ---- Verifiers ----------------------------------------------------

/// The verifier names a reinforcement prompt may select. Both are
/// mechanical: nothing here asks a model whether an answer is good.
pub(crate) const VERIFIER_NUON: &str = "nuon_equals";
pub(crate) const VERIFIER_EXACT: &str = "exact";

fn verifier_known(name: &str) -> BquestResult<()> {
    snafu::ensure_whatever!(
        name == VERIFIER_NUON || name == VERIFIER_EXACT,
        "unknown verifier {name:?} (known: {VERIFIER_NUON}, {VERIFIER_EXACT})"
    );
    Ok(())
}

/// Grade one response against its reference. 1.0 is a pass, 0.0 a
/// fail - the reward a group-relative advantage is computed from.
///
/// `nuon_equals` parses BOTH sides and compares VALUES, never text:
/// NUON renders the same value differently depending on content, so a
/// text comparison would score formatting rather than correctness.
pub(crate) fn verify(verifier: &str, response: &str, reference: &str) -> BquestResult<f32> {
    verifier_known(verifier)?;
    Ok(match verifier {
        VERIFIER_NUON => {
            let expected = lib::nu::from_nuon_text(reference)?;
            match lib::nu::from_nuon_text(response.trim()) {
                Ok(actual) if actual == expected => 1.0,
                _ => 0.0,
            }
        }
        _ => {
            if response.trim() == reference.trim() {
                1.0
            } else {
                0.0
            }
        }
    })
}
