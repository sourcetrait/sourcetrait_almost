use crate::graph::{bucket_for, pad_mask_values, SlotWrite, BUCKET_GRAIN};
use crate::{
    chat_continue, chat_wrap, consts, GenerateOptions, RestoredContext, Settings,
};

/// The E4 building-block contract: a raw-launched slot_write at a
/// device-resident index lands exactly where a host-offset slice_set
/// would. Skips quietly off-gpu (the cuda feature compiles on any box).
#[test]
fn slot_write_matches_slice_set() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let device = candle_core::Device::new_cuda(0).unwrap();
    for dtype in [candle_core::DType::F32, candle_core::DType::BF16] {
        let dst = candle_core::Tensor::zeros((1, 4, 8, 16), dtype, &device).unwrap();
        let reference = candle_core::Tensor::zeros((1, 4, 8, 16), dtype, &device).unwrap();
        let src = candle_core::Tensor::arange(1f32, (4 * 16 + 1) as f32, &device)
            .unwrap()
            .reshape((1, 4, 1, 16))
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let slot = candle_core::Tensor::new(&[5u32], &device).unwrap();

        dst.inplace_op3(&src, &slot, &SlotWrite { dtype }).unwrap();
        reference.slice_set(&src, 2, 5).unwrap();

        let written = dst
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = reference
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(written, expected, "dtype {dtype:?}");
    }
}

/// Re-staging the slot buffer moves later writes without touching the
/// kernel or its arguments - the staged-buffer indirection the decode
/// graph replays against.
#[test]
fn slot_write_follows_restaged_index() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let device = candle_core::Device::new_cuda(0).unwrap();
    let dst = candle_core::Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &device).unwrap();
    let src_one = candle_core::Tensor::full(1f32, (1, 2, 1, 8), &device).unwrap();
    let src_two = candle_core::Tensor::full(2f32, (1, 2, 1, 8), &device).unwrap();
    let slot = candle_core::Tensor::new(&[0u32], &device).unwrap();

    let op = SlotWrite {
        dtype: candle_core::DType::F32,
    };
    dst.inplace_op3(&src_one, &slot, &op).unwrap();
    // Re-stage the index in place (what the host does between replays).
    slot.slice_set(&candle_core::Tensor::new(&[3u32], &device).unwrap(), 0, 0)
        .unwrap();
    dst.inplace_op3(&src_two, &slot, &op).unwrap();

    let sums = dst
        .sum(3)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Per head: slot 0 sums to 8 (ones), slot 3 sums to 16 (twos).
    assert_eq!(sums, vec![8.0, 0.0, 0.0, 16.0, 8.0, 0.0, 0.0, 16.0]);
}

/// Phase C capture contract: an attention-shaped candle op chain over
/// persistent buffers - slot_write appends, bucket narrows, matmul,
/// additive mask, f32 softmax, matmul, a persistent-output write -
/// must replay exactly what the uncaptured chain computes. Pins the
/// RECIPE: candle's param-cache guard enabled BEFORE a warmup run
/// (strided/broadcast kernels upload dims/strides from temporary host
/// Vecs - uncached, a captured copy replays dead memory; a cache miss
/// during active capture is a designed hard error), then capture with
/// intermediates dropping in-recording (free nodes). Diagnostic env
/// knobs: QUEST_CAPTURE_MODE=thread_local|relaxed|global,
/// QUEST_CAPTURE_HOLD=inside|1|0 ("0" = the known-bad drop-after-
/// capture shape), QUEST_CAPTURE_DOT=<dot dump path>. Run variant
/// sweeps one per process - a broken capture can wedge the CUDA
/// context.
#[test]
fn capture_replays_attention_shaped_chain() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let device = candle_core::Device::new_cuda(0).unwrap();
    let candle_core::Device::Cuda(cuda_device) = &device else {
        unreachable!()
    };
    let dtype = candle_core::DType::BF16;
    let (heads, dim, capacity, bucket) = (4usize, 16usize, 64usize, 32usize);

    let q = candle_core::Tensor::randn(0f32, 1f32, (1, heads, 1, dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k_new = candle_core::Tensor::randn(0f32, 1f32, (1, heads, 1, dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let v_new = candle_core::Tensor::randn(0f32, 1f32, (1, heads, 1, dim), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let key_buffer = candle_core::Tensor::zeros((1, heads, capacity, dim), dtype, &device).unwrap();
    let value_buffer =
        candle_core::Tensor::zeros((1, heads, capacity, dim), dtype, &device).unwrap();
    let slot = candle_core::Tensor::new(&[5u32], &device).unwrap();
    let mask_values = pad_mask_values(bucket, 6);
    let mask = candle_core::Tensor::from_vec(mask_values, (1, 1, 1, bucket), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let out = candle_core::Tensor::zeros((1, heads, 1, dim), candle_core::DType::F32, &device)
        .unwrap();

    let write = SlotWrite { dtype };
    // Returns every intermediate so a variant can HOLD them across the
    // capture (drops during recording are one breakage hypothesis).
    let chain = || -> candle_core::Result<Vec<candle_core::Tensor>> {
        key_buffer.inplace_op3(&k_new, &slot, &write)?;
        value_buffer.inplace_op3(&v_new, &slot, &write)?;
        let keys = key_buffer.narrow(2, 0, bucket)?;
        let values = value_buffer.narrow(2, 0, bucket)?;
        let logits = (q.matmul(&keys.transpose(2, 3)?)? * 0.25)?;
        let masked = logits.broadcast_add(&mask)?;
        let masked_f32 = masked.to_dtype(candle_core::DType::F32)?;
        let weights_f32 = candle_nn::ops::softmax_last_dim(&masked_f32)?;
        let weights = weights_f32.to_dtype(dtype)?;
        let attn = weights.matmul(&values)?;
        let attn_f32 = attn.to_dtype(candle_core::DType::F32)?;
        out.slice_set(&attn_f32, 0, 0)?;
        Ok(vec![
            keys, values, logits, masked, masked_f32, weights_f32, weights, attn, attn_f32,
        ])
    };

    fn snap(tensor: &candle_core::Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    // candle's capture contract: enable the param cache FIRST, then a
    // WARMUP run populates it (content-keyed dims/strides vectors);
    // only then capture - a cache miss during active capture is a
    // hard error by design.
    let _htod_cache = cuda_device.enable_cuda_graph_htod_cache();
    let _ = chain().unwrap();
    let expected = snap(&out);

    // One variant per PROCESS (a broken capture can wedge the CUDA
    // context): the diagnostic matrix drives this via env; the default
    // is the pinned production recipe.
    use cudarc::driver::sys::CUstreamCaptureMode;
    let mode_name =
        std::env::var("QUEST_CAPTURE_MODE").unwrap_or_else(|_| String::from("thread_local"));
    // "inside" = drop intermediates WHILE STILL CAPTURING (recorded
    // free nodes - the production sequence's shape, the default);
    // "1" = hold them for the graph's lifetime; "0" = drop after
    // end_capture (live frees of graph-epoch VAs - the known-bad
    // shape, expected to fail).
    let hold = std::env::var("QUEST_CAPTURE_HOLD").unwrap_or_else(|_| String::from("inside"));
    let mode = match mode_name.as_str() {
        "thread_local" => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        "relaxed" => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        "global" => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        other => panic!("unknown QUEST_CAPTURE_MODE {other}"),
    };
    let label = format!("{mode_name}/hold={hold}");
    eprintln!("{label}: begin");

    let stream = cuda_device.cuda_stream();
    stream.begin_capture(mode).unwrap();
    let chain_result = chain();
    // "inside" drops while the stream is still capturing (frees record
    // as in-graph free nodes - the production sequence's shape);
    // everything else stays alive across end_capture.
    let mut held: Option<Vec<candle_core::Tensor>> = match chain_result {
        Ok(intermediates) => {
            if hold == "inside" {
                drop(intermediates);
                None
            } else {
                Some(intermediates)
            }
        }
        Err(error) => panic!("chain failed during capture: {error}"),
    };
    let end_result = stream.end_capture(
        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if hold == "0" {
        // Known-bad shape: live frees of graph-epoch VAs after the
        // capture ended.
        held = None;
    }
    let graph = end_result.unwrap().unwrap();
    eprintln!("{label}: captured");

    // Forensics: QUEST_CAPTURE_DOT=<path> dumps the captured graph
    // (verbose: mem-node VAs + kernel params) before launching.
    if let Ok(dot_path) = std::env::var("QUEST_CAPTURE_DOT") {
        let c_path = std::ffi::CString::new(dot_path).unwrap();
        unsafe {
            cudarc::driver::sys::cuGraphDebugDotPrint(graph.cu_graph(), c_path.as_ptr(), 1)
                .result()
                .unwrap();
        }
        eprintln!("{label}: dot dumped");
    }

    for launch_index in 0..2 {
        graph.launch().unwrap();
        eprintln!("{label}: launched {launch_index}");
        let replayed = snap(&out);
        assert_eq!(
            replayed, expected,
            "{label}: launch {launch_index} diverged"
        );
    }
    eprintln!("{label}: ok");
    drop(held);
}

#[test]
fn bucket_math_rounds_up_by_grain() {
    assert_eq!(bucket_for(0), BUCKET_GRAIN);
    assert_eq!(bucket_for(1), BUCKET_GRAIN);
    assert_eq!(bucket_for(BUCKET_GRAIN), BUCKET_GRAIN);
    assert_eq!(bucket_for(BUCKET_GRAIN + 1), 2 * BUCKET_GRAIN);
    assert_eq!(bucket_for(32769), 34816);
}

#[test]
fn pad_mask_hides_exactly_the_tail() {
    let values = pad_mask_values(8, 3);
    assert_eq!(values.len(), 8);
    assert!(values[..3].iter().all(|v| *v == 0.0));
    assert!(values[3..].iter().all(|v| *v == f32::NEG_INFINITY));
}

/// The quest-chat flow: generate() then generate_from() chained in ONE
/// process (RestoredContext seeded from each report), decode graph-armed.
/// Needs the pulled checkpoint + a cuda device; run one variant per
/// process:
///   cargo test --release --features flash-attn chained_generate_from \
///     -- --ignored --nocapture
/// Variants: QUEST_PROBE_GRAPH=0 (classic decode), QUEST_PROBE_CAPTURE=0
/// (staged decode without capture+replay from turn 2 on).
#[test]
#[ignore]
fn chained_generate_from_under_graph() {
    if !crate::r::candle::cuda_is_available() {
        return;
    }
    let graph = std::env::var("QUEST_PROBE_GRAPH").map(|v| v != "0").unwrap_or(true);
    let capture = std::env::var("QUEST_PROBE_CAPTURE").map(|v| v != "0").unwrap_or(true);
    let model_dir = crate::hub::default_model_dir(consts::DEFAULT_MODEL_ID);
    let paths = crate::hub::ensure_model(consts::DEFAULT_MODEL_ID, &model_dir).unwrap();
    let device = candle_core::Device::new_cuda(0).unwrap();
    let settings = Settings {
        use_flash_attn: cfg!(feature = "flash-attn"),
        profile_attn: false,
        eviction: None,
        graph,
    };
    let loaded = crate::load::load_model(&paths, &device, candle_core::DType::BF16, settings).unwrap();
    let mut model = loaded.model;
    // Mirror Config::default().generation (the chat-session defaults)
    // unless the env overrides pieces for bisection.
    let greedy = std::env::var("QUEST_PROBE_GREEDY").map(|v| v != "0").unwrap_or(false);
    let sample_len: usize = std::env::var("QUEST_PROBE_SAMPLE_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let options = GenerateOptions {
        greedy,
        temperature: consts::DEFAULT_TEMPERATURE,
        top_p: consts::DEFAULT_TOP_P,
        sample_len,
        seed: 299_792_458,
        speculate: false,
        dump_logits: None,
    };
    // A big first turn (QUEST_PROBE_PROMPT_FILE's contents) mirrors the
    // failing chat session's paste; absent, a short built-in.
    let pasted = std::env::var("QUEST_PROBE_PROMPT_FILE")
        .ok()
        .map(|path| std::fs::read_to_string(path).unwrap());
    let turns: usize = std::env::var("QUEST_PROBE_TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let mut restored: Option<RestoredContext> = None;
    for turn in 0..turns {
        if turn == 1 && !capture {
            model.set_graph_capture(false).unwrap();
        }
        // QUEST_PROBE_CLEAR=1: replace the generate_from continuation
        // with the needle-shaped flow - clear + a LONGER fresh prompt,
        // so the regrow happens at length 0 with graphs alive.
        let clear_flow = std::env::var("QUEST_PROBE_CLEAR").map(|v| v == "1").unwrap_or(false);
        if clear_flow && turn > 0 {
            restored = None;
        }
        let text = match (&restored, turn) {
            (None, 0) => match &pasted {
                Some(pasted) => {
                    chat_wrap(&format!("{pasted}\n\nDo you understand NUON now?"))
                }
                None => chat_wrap("Say hi."),
            },
            (None, _) => match &pasted {
                Some(pasted) => chat_wrap(&format!(
                    "{pasted}\n{pasted}\n\nDo you understand NUON now?"
                )),
                None => chat_wrap("Say hi again."),
            },
            (Some(_), 1) => chat_continue(
                "good job. let's try it out. output a nuon table that represents name and age. bob is 42 years old. cindy is 3 months old. todd is 12 years old.",
            ),
            (Some(_), _) => chat_continue("test"),
        };
        let mut generation = match &restored {
            Some(context) => model
                .generate_from(&loaded.tokenizer, context, &text, &options)
                .unwrap(),
            None => model.generate(&loaded.tokenizer, &text, &options).unwrap(),
        };
        let mut generated = 0usize;
        for step in &mut generation {
            let step = step.unwrap_or_else(|error| panic!("turn {turn} step failed: {error}"));
            let _ = step;
            generated += 1;
        }
        // Deferred-error checkpoints: cudarc destructors record failures
        // on the context and the NEXT fallible call delivers them; a
        // failing sync here localizes which teardown recorded.
        if let Err(error) = device.synchronize() {
            eprintln!("turn {turn} checkpoint A (post-loop, pre-finish): {error}");
        }
        let report = generation.finish().unwrap();
        if let Err(error) = device.synchronize() {
            eprintln!("turn {turn} checkpoint B (post-finish): {error}");
        }
        let (free_bytes, _total) = cudarc::driver::result::mem_get_info().unwrap();
        eprintln!(
            "turn {turn}: {generated} steps, ctx {} ({:?}), vram free {} MiB",
            report.context_ids.len(),
            report.finish_reason,
            free_bytes / (1024 * 1024)
        );
        restored = Some(RestoredContext {
            context_len: report.context_ids.len(),
            context_ids: report.context_ids,
        });
        if let Err(error) = device.synchronize() {
            eprintln!("turn {turn} checkpoint C (pre-next-turn): {error}");
        }
    }
}
