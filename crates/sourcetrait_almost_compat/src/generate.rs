use crate::*;

/// Sampling and length knobs for one generation run.
#[derive(Debug, Clone)]
pub(crate) struct GenerateOptions {
    pub(crate) greedy: bool,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) sample_len: usize,
    pub(crate) seed: u64,
}

/// Prefill + decode loop. Generated text streams to stdout; timing stats go
/// to stderr so stdout stays pipeable. No repeat penalty, by design: the
/// baseline must keep logits unmodified for parity comparisons.
pub(crate) fn generate(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    opts: &GenerateOptions,
    device: &candle_core::Device,
) -> CompatResult<()> {
    let encoding = match tokenizer.encode(text, true) {
        Ok(encoding) => encoding,
        Err(error) => snafu::whatever!("prompt tokenization failed: {error}"),
    };
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    snafu::ensure_whatever!(!prompt_ids.is_empty(), "the prompt tokenized to zero tokens");
    let stop_ids = resolve_stop_ids(tokenizer);

    let sampling = if opts.greedy {
        r::candle::Sampling::ArgMax
    } else {
        r::candle::Sampling::TopP {
            p: opts.top_p,
            temperature: opts.temperature,
        }
    };
    let mut processor = r::candle::LogitsProcessor::from_sampling(opts.seed, sampling);
    let mut stream = TokenStream::new(tokenizer);

    let prefill_start = Instant::now();
    let input = candle_core::Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let mut next_token = processor.sample(&logits)?;
    let mut offset = prompt_ids.len();
    let mut generated = 0usize;
    let mut hit_stop = false;
    let decode_start = Instant::now();
    loop {
        if stop_ids.contains(&next_token) {
            hit_stop = true;
            break;
        }
        if let Some(chunk) = stream.next_token(next_token)? {
            print!("{chunk}");
            io::stdout().flush()?;
        }
        generated += 1;
        if generated >= opts.sample_len {
            break;
        }
        let input = candle_core::Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        offset += 1;
        next_token = processor.sample(&logits)?;
    }
    if let Some(rest) = stream.decode_rest()? {
        print!("{rest}");
    }
    println!();
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    eprintln!(
        "lmst: prefill {} tokens in {:.2}s ({:.1} tok/s); decode {} tokens in {:.2}s ({:.1} tok/s); stopped by {}",
        prompt_ids.len(),
        prefill_seconds,
        prompt_ids.len() as f64 / prefill_seconds.max(f64::EPSILON),
        generated,
        decode_seconds,
        generated as f64 / decode_seconds.max(f64::EPSILON),
        if hit_stop { "stop token" } else { "sample_len" },
    );
    Ok(())
}
