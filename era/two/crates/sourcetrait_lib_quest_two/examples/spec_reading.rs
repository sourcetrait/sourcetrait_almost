// SpecReadingEarly: what the supervised checkpoint does with the era-two
// channel grammar, with no transport and no daemon involved. Emits NUON
// lines so a partial run stays readable.
use sourcetrait_lib_quest_two as lib;
use lib::channel::{Aliasing, Tag, parse_blocks};

const PREAMBLE: &str = "\
You answer using typed channels. An output block opens with the marker \
<|extra_id_2|> at the start of a line, carries a nushell type on that same \
line, then the NUON value on its own lines, then <|extra_id_6|> alone on a \
line. Emit nothing else.

Worked example. Asked for the first two even numbers as a list of ints:
<|extra_id_2|> list<int>
[2, 4]
<|extra_id_6|>";

/// One probe cell: what it asks for, and the type it should declare.
const CELLS: [(&str, &str, &str); 6] = [
    ("flat_record", "record<name: string, age: int>",
     "Give a record for a person named Ada who is 36."),
    ("nested_record", "record<user: record<name: string>, active: bool>",
     "Give a record whose user field is a record with name Ada, and active true."),
    ("list_of_string", "list<string>",
     "Give the three primary colours as a list of strings."),
    ("table", "table<name: string, age: int>",
     "Give a table of two people: Ada aged 36 and Bob aged 41."),
    ("cell_path", "cell-path",
     "Give the cell path addressing the name field of row 0."),
    ("sugar_scalar", "duration",
     "Give a duration of ninety seconds."),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = lib::LibConfig::load(Some("r2_sft"))?;
    let settings = lib::LibSettings::load(Some("default"))?;
    let model_dir = config.model_dir();
    let hybrid = lib::load_config(&model_dir)?;
    let tokenizer = lib::load_tokenizer(&model_dir)?;
    lib::verify_token_map(&tokenizer)?;

    let device = candle_core::Device::new_cuda(0)?;
    let weights = lib::load_weights(&config, candle_core::DType::BF16, &device)?;
    let mut model = lib::OlmoHybrid::new(&hybrid, settings, weights)?;

    let span = lib::nu::Span::unknown();
    let mut rows: Vec<lib::nu::Value> = Vec::new();
    for (name, typedef, task) in CELLS {
        let prompt = format!("{PREAMBLE}\n\nNow: {task}\nDeclare the type {typedef}.");
        let options = lib::GenerateOptions::greedy(192);
        let mut generation = model.generate(&tokenizer, &prompt, &options)?;
        let mut text = String::new();
        for step in generation.by_ref() {
            text.push_str(&step?.chunk);
        }
        let report = generation.finish();
        text.push_str(&report.rest);

        let opened_marked = text.trim_start().starts_with(Tag::Output.spelling());
        let opened_attractor = text.trim_start().starts_with("<function_calls>");
        let strict = parse_blocks(&text, Aliasing::Strict);
        let aliased = parse_blocks(&text, Aliasing::ToolMarkers);
        let payload_is_nuon = aliased
            .as_ref()
            .ok()
            .and_then(|blocks| blocks.first())
            .map(|block| lib::nu::from_nuon_text(block.content.trim()).is_ok())
            .unwrap_or(false);
        let declared_matches = aliased
            .as_ref()
            .ok()
            .and_then(|blocks| blocks.first())
            .map(|block| block.header == typedef)
            .unwrap_or(false);

        rows.push(lib::nu::Value::record(
            lib::nu::record! {
                "cell" => lib::nu::Value::string(name, span),
                "declared" => lib::nu::Value::string(typedef, span),
                "opened_marked" => lib::nu::Value::bool(opened_marked, span),
                "opened_attractor" => lib::nu::Value::bool(opened_attractor, span),
                "strict_parses" => lib::nu::Value::bool(strict.is_ok(), span),
                "aliased_parses" => lib::nu::Value::bool(aliased.is_ok(), span),
                "payload_is_nuon" => lib::nu::Value::bool(payload_is_nuon, span),
                "declared_matches" => lib::nu::Value::bool(declared_matches, span),
                "generated" => lib::nu::Value::int(report.generated_token_count as i64, span),
                "response" => lib::nu::Value::string(text.clone(), span),
            },
            span,
        ));
        println!("{}", lib::nu::to_nuon_text(rows.last().expect("just pushed"))?);
    }

    let out = std::env::args().nth(1).unwrap_or_default();
    if !out.is_empty() {
        lib::nu::save_value(std::path::Path::new(&out), &lib::nu::Value::list(rows, span))?;
    }
    Ok(())
}
