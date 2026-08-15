//! The page chunker: the global record stream packed into
//! training-sized page tables, each measured through the real
//! assembler against the training-chunk budget.
use crate::*;

use std::io::BufRead;
use std::io::Write;

use crate::assembler::Assembler;
use crate::assembler::SyntaxTable;
use crate::bucket::BucketTable;
use crate::dictionary::read_words_ordered;
use crate::lexer::Segmenter;
use crate::ucd::CharacterTable;

/// The framed TRAIN assembly for one page: META (source, page) then
/// the table as the INPUT serialization, at the enforced two-space
/// indentation. The framing is what trains; the page file keeps the
/// bare table.
fn framed_page(table_text: &str, page_number: usize) -> String {
    let mut out = String::new();
    out.push_str("OPEN TRAIN\n");
    out.push_str("  OPEN PAGE\n");
    out.push_str("    OPEN META\n");
    out.push_str("      BEGIN NUON\n");
    out.push_str(&format!(
        "        {{source: \"Wiktionary\", page: {page_number}}}\n"
    ));
    out.push_str("      END NUON\n");
    out.push_str("    CLOSE META\n");
    out.push_str("    OPEN INPUT\n");
    out.push_str("      BEGIN NUON\n");
    for line in table_text.lines() {
        out.push_str("        ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("      END NUON\n");
    out.push_str("    CLOSE INPUT\n");
    out.push_str("  CLOSE PAGE\n");
    out.push_str("CLOSE TRAIN\n");
    out
}

/// The bare page table: the record lines bracketed as one NUON list.
fn page_table(rows: &[String]) -> String {
    let mut out = String::with_capacity(rows.iter().map(String::len).sum::<usize>() + 8);
    out.push_str("[\n");
    for row in rows {
        out.push_str(row);
        out.push('\n');
    }
    out.push_str("]\n");
    out
}

/// One packed page's outcome.
struct PackedPage {
    rows: usize,
    wire_tokens: usize,
}

/// `biquest wikimedia pages`: pack the record stream into page tables
/// greedily - as many entries per page as fit the budget once the
/// whole framed page assembles - writing pages/page_NNNNNN.nuon plus
/// provenance. A single entry larger than the budget lands alone on
/// its own oversized page, counted and surfaced, never split or
/// silently dropped.
pub(crate) fn wikimedia_pages(args: &WikimediaPagesArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    snafu::ensure_whatever!(args.seq_len >= 64, "seq_len below any usable page");
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let admitted = match &args.words {
        Some(path) => read_words_ordered(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let syntax = match &args.syntax {
        Some(path) => SyntaxTable::load(path)?,
        None => SyntaxTable::embedded()?,
    };
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let assembler = Assembler::new(&syntax, &segmenter, &table, &buckets, &admitted);

    let pages_dir = args.out.join("pages");
    fs::create_dir_all(&pages_dir)?;

    // The framing's own wire cost, measured once on an empty table at
    // a wide page number; per-row costs are content tokenization of
    // the row line, additive across newline boundaries. The greedy
    // estimate overfills at most marginally and the exact assembly of
    // each flushed page is the check that trims it.
    let overhead = assembler.encode(&framed_page("[\n]\n", 999_999))?.len();

    let file = fs::File::open(&args.nuonl)?;
    let reader = io::BufReader::with_capacity(1 << 20, file);

    let mut current_rows: Vec<String> = Vec::new();
    let mut current_estimate = overhead;
    let mut packed: Vec<PackedPage> = Vec::new();
    let mut entries = 0usize;
    let mut oversized_pages = 0usize;
    let mut max_wire = 0usize;

    let flush = |rows: &mut Vec<String>,
                     packed: &mut Vec<PackedPage>,
                     oversized: &mut usize,
                     max_wire: &mut usize|
     -> BiquestResult<()> {
        while !rows.is_empty() {
            let page_number = packed.len() + 1;
            let mut kept = rows.clone();
            let mut spill: Vec<String> = Vec::new();
            let wire_tokens = loop {
                let framed = framed_page(&page_table(&kept), page_number);
                let wire = assembler.encode(&framed)?.len();
                if wire <= args.seq_len || kept.len() == 1 {
                    break wire;
                }
                spill.insert(0, kept.pop().expect("non-empty"));
            };
            if wire_tokens > args.seq_len {
                *oversized += 1;
            }
            *max_wire = (*max_wire).max(wire_tokens);
            fs::write(
                pages_dir.join(format!("page_{page_number:06}.nuon")),
                page_table(&kept),
            )?;
            packed.push(PackedPage { rows: kept.len(), wire_tokens });
            *rows = spill;
        }
        Ok(())
    };

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        entries += 1;
        let row_cost = segmenter.segment(&format!("{line}\n"))?.len();
        if !current_rows.is_empty() && current_estimate + row_cost > args.seq_len {
            flush(&mut current_rows, &mut packed, &mut oversized_pages, &mut max_wire)?;
            current_estimate = overhead;
        }
        current_rows.push(line);
        current_estimate += row_cost;
        if entries.is_multiple_of(50_000) {
            eprintln!(
                "pages: {entries} entries, {} pages, {:.0} s",
                packed.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    flush(&mut current_rows, &mut packed, &mut oversized_pages, &mut max_wire)?;

    snafu::ensure_whatever!(!packed.is_empty(), "the record stream packs no page");
    let total_wire: usize = packed.iter().map(|page| page.wire_tokens).sum();
    let min_rows = packed.iter().map(|page| page.rows).min().unwrap_or(0);
    let max_rows = packed.iter().map(|page| page.rows).max().unwrap_or(0);

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "nuonl" => v_str(&args.nuonl.display().to_string()),
            "seq_len" => v_int(args.seq_len as i64),
            "entries" => v_int(entries as i64),
            "pages" => v_int(packed.len() as i64),
            "oversized_pages" => v_int(oversized_pages as i64),
            "max_page_wire" => v_int(max_wire as i64),
            "mean_page_wire" => v_int((total_wire / packed.len()) as i64),
            "frame_overhead_wire" => v_int(overhead as i64),
            "min_page_rows" => v_int(min_rows as i64),
            "max_page_rows" => v_int(max_rows as i64),
            "source" => v_str("Wiktionary"),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.join("provenance.nuon"), &provenance)?;

    let engine_state = nu_protocol::engine::EngineState::new();
    let mut page_file = io::BufWriter::new(fs::File::create(
        args.out.join("pages.nuonl"),
    )?);
    for (index, page) in packed.iter().enumerate() {
        let row = harness::nu::Value::record(
            harness::nu::record! {
                "page" => v_int((index + 1) as i64),
                "rows" => v_int(page.rows as i64),
                "wire_tokens" => v_int(page.wire_tokens as i64),
            },
            span(),
        );
        writeln!(
            page_file,
            "{}",
            crate::associations::condensed_line(&engine_state, &row)?
        )?;
    }
    page_file.flush()?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "entries" => v_int(entries as i64),
            "pages" => v_int(packed.len() as i64),
            "oversized_pages" => v_int(oversized_pages as i64),
            "max_page_wire" => v_int(max_wire as i64),
            "mean_page_wire" => v_int((total_wire / packed.len()) as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
