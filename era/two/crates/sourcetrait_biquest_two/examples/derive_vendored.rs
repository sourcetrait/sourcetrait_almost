//! Derive the vendored third-party data files from the pinned
//! enwiktionary dump: the language-code map, the inflection-tag map,
//! and the place data. Run on a dump bump, then re-vendor data/.
//!
//! Usage:
//!   cargo run --release --example derive_vendored -- \
//!     <multistream.xml.bz2> <index.txt.bz2> <out_dir> <dump_date>
//!
//! The out_dir gains languages/<date>/languages.nuon,
//! form_of/<date>/tags.nuon, and place/<date>/place.nuon. Copy them
//! into data/ (or point out_dir at data/ directly).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::Seek;

const LANGUAGES_JSON: &str = "Module:languages/code to canonical name.json";
const ETYMOLOGY_JSON: &str = "Module:etymology languages/code to canonical name.json";
const FAMILIES_JSON: &str = "Module:families/code to canonical name.json";
const FORM_OF_DATA_1: &str = "Module:form of/data/1";
const FORM_OF_DATA_2: &str = "Module:form of/data/2";
const FORM_OF_LANG_EN: &str = "Module:form of/lang-data/en";
const PLACE_PLACETYPES: &str = "Module:place/placetypes";
const PLACE_LOCATIONS: &str = "Module:place/locations";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: derive_vendored <dump.xml.bz2> <index.txt.bz2> <out_dir> <date>");
        std::process::exit(2);
    }
    let dump = std::path::Path::new(&args[1]);
    let index = std::path::Path::new(&args[2]);
    let out_dir = std::path::Path::new(&args[3]);
    let date = &args[4];

    let titles = [
        LANGUAGES_JSON,
        ETYMOLOGY_JSON,
        FAMILIES_JSON,
        FORM_OF_DATA_1,
        FORM_OF_DATA_2,
        FORM_OF_LANG_EN,
        PLACE_PLACETYPES,
        PLACE_LOCATIONS,
    ];
    let pages = fetch_pages(dump, index, &titles);
    for title in titles {
        assert!(pages.contains_key(title), "page {title:?} not found in the dump");
    }

    let languages = render_languages(
        &pages[LANGUAGES_JSON],
        &pages[ETYMOLOGY_JSON],
        &pages[FAMILIES_JSON],
        date,
    );
    let tags = render_tags(
        &pages[FORM_OF_DATA_1],
        &pages[FORM_OF_DATA_2],
        &pages[FORM_OF_LANG_EN],
        date,
    );
    let place = render_place(&pages[PLACE_PLACETYPES], &pages[PLACE_LOCATIONS], date);
    let form_of = render_form_of_aliases(dump, index, date);

    for (family, file, payload) in [
        ("languages", "languages.nuon", languages),
        ("form_of", "tags.nuon", tags),
        ("form_of", "form_of.nuon", form_of),
        ("place", "place.nuon", place),
    ] {
        let dir = out_dir.join(family).join(date);
        std::fs::create_dir_all(&dir).expect("create out dir");
        let path = dir.join(file);
        std::fs::write(&path, payload).expect("write artifact");
        println!("wrote {}", path.display());
    }
}

/// Stream the multistream index, find the wanted titles' block
/// offsets, decode each block once, and return title -> wikitext.
fn fetch_pages(
    dump: &std::path::Path,
    index: &std::path::Path,
    titles: &[&str],
) -> HashMap<String, String> {
    let wanted: std::collections::HashSet<&str> = titles.iter().copied().collect();
    let index_file = std::fs::File::open(index).expect("open index");
    let index_reader: Box<dyn BufRead> = if index.extension().is_some_and(|e| e == "bz2") {
        Box::new(std::io::BufReader::with_capacity(
            1 << 20,
            bzip2::read::MultiBzDecoder::new(std::io::BufReader::with_capacity(
                1 << 20,
                index_file,
            )),
        ))
    } else {
        Box::new(std::io::BufReader::with_capacity(1 << 20, index_file))
    };
    let mut offsets: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for line in index_reader.lines() {
        let line = line.expect("index line");
        let mut parts = line.splitn(3, ':');
        let (Some(offset), Some(_), Some(title)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if wanted.contains(title) {
            let offset: u64 = offset.parse().expect("index offset");
            offsets.entry(offset).or_default().push(title.to_string());
        }
    }
    let mut pages = HashMap::new();
    for (offset, block_titles) in offsets {
        let mut file = std::fs::File::open(dump).expect("open dump");
        file.seek(std::io::SeekFrom::Start(offset)).expect("seek");
        let decoder =
            bzip2::read::BzDecoder::new(std::io::BufReader::with_capacity(1 << 20, file));
        let mut reader = quick_xml::Reader::from_reader(std::io::BufReader::with_capacity(
            1 << 20,
            decoder,
        ));
        reader.config_mut().check_end_names = false;
        let mut buf = Vec::new();
        let mut path: Vec<Vec<u8>> = Vec::new();
        let mut title = String::new();
        let mut text = String::new();
        loop {
            buf.clear();
            let event = match reader.read_event_into(&mut buf) {
                Ok(event) => event,
                Err(e) => panic!("xml parse failed at offset {offset}: {e}"),
            };
            use quick_xml::events::Event;
            match event {
                Event::Eof => break,
                Event::Start(start) => {
                    let name = start.name().as_ref().to_vec();
                    if name == b"page" {
                        title.clear();
                        text.clear();
                        path.clear();
                    }
                    path.push(name);
                }
                Event::Text(data) => {
                    let decoded = data.decode().expect("xml text");
                    absorb(&path, &decoded, &mut title, &mut text);
                }
                Event::CData(data) => {
                    let decoded = data.decode().expect("xml cdata");
                    absorb(&path, &decoded, &mut title, &mut text);
                }
                Event::GeneralRef(general_ref) => {
                    let resolved = match general_ref.resolve_char_ref() {
                        Ok(Some(c)) => c.to_string(),
                        _ => match general_ref.decode().expect("entity").as_ref() {
                            "amp" => String::from("&"),
                            "lt" => String::from("<"),
                            "gt" => String::from(">"),
                            "quot" => String::from("\""),
                            "apos" => String::from("'"),
                            other => panic!("unresolvable entity &{other};"),
                        },
                    };
                    absorb(&path, &resolved, &mut title, &mut text);
                }
                Event::End(end) => {
                    if end.name().as_ref() == b"page"
                        && block_titles.iter().any(|wanted| wanted == &title)
                    {
                        pages.insert(title.clone(), text.clone());
                    }
                    path.pop();
                }
                _ => {}
            }
        }
    }
    pages
}

fn absorb(path: &[Vec<u8>], decoded: &str, title: &mut String, text: &mut String) {
    match path {
        [page, field] if page == b"page" && field == b"title" => title.push_str(decoded),
        [page, revision, field]
            if page == b"page" && revision == b"revision" && field == b"text" =>
        {
            text.push_str(decoded)
        }
        _ => {}
    }
}

/// A string as an always-quoted NUON literal.
fn nuon_str(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn provenance_header(subject: &str, sources: &str, date: &str) -> String {
    format!(
        "# Third-party data: English Wiktionary's {subject}, derived\n\
         # from {sources}\n\
         # at the {date} dump (CC BY-SA; attribution at the\n\
         # repository's docs/licenses/wiktionary/). Re-derive on a dump\n\
         # bump via `cargo run --release --example derive_vendored`.\n"
    )
}

// ------------------------------------------------------------ form-of aliases

/// The complete form-of alias vocabulary, from the dump's own
/// Template-namespace redirects: every redirect whose transitively
/// resolved target ends in " of". This is the whole alias surface -
/// the hand-grown list this replaces missed members by construction.
fn render_form_of_aliases(dump: &std::path::Path, index: &std::path::Path, date: &str) -> String {
    // Index pass: the blocks holding Template-namespace pages.
    let index_file = std::fs::File::open(index).expect("open index");
    let index_reader: Box<dyn BufRead> = if index.extension().is_some_and(|e| e == "bz2") {
        Box::new(std::io::BufReader::with_capacity(
            1 << 20,
            bzip2::read::MultiBzDecoder::new(std::io::BufReader::with_capacity(
                1 << 20,
                index_file,
            )),
        ))
    } else {
        Box::new(std::io::BufReader::with_capacity(1 << 20, index_file))
    };
    let mut offsets: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for line in index_reader.lines() {
        let line = line.expect("index line");
        let mut parts = line.splitn(3, ':');
        let (Some(offset), Some(_), Some(title)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if title.starts_with("Template:") {
            offsets.insert(offset.parse().expect("index offset"));
        }
    }

    // Block pass: template redirect pairs, source -> target, both
    // without the namespace prefix.
    let mut redirects: HashMap<String, String> = HashMap::new();
    for offset in offsets {
        for (title, target) in block_template_redirects(dump, offset) {
            redirects.insert(title, target);
        }
    }

    // Transitive resolution, then the " of" filter. Targets in the
    // en- prefixed family stay out: those templates bake their
    // language and have their own arms.
    let mut aliases: Vec<(String, String)> = Vec::new();
    for source in redirects.keys() {
        let mut target = source.clone();
        for _ in 0..4 {
            match redirects.get(&target) {
                Some(next) => target = next.clone(),
                None => break,
            }
        }
        if target.ends_with(" of")
            && target != "form of"
            && !target.starts_with("en-")
            && source != &target
        {
            aliases.push((source.clone(), target));
        }
    }
    aliases.sort();
    aliases.dedup();

    let mut out = provenance_header(
        "form-of alias vocabulary",
        "the Template-namespace redirects whose resolved targets\n\
         # end in \" of\"",
        date,
    );
    out.push_str("{\n    aliases: [\n        [ name full ];\n");
    for (name, full) in &aliases {
        out.push_str(&format!(
            "        [ {} {} ]\n",
            nuon_str(name),
            nuon_str(full)
        ));
    }
    out.push_str("    ]\n}\n");
    out
}

/// One multistream block's Template-namespace redirect pairs.
fn block_template_redirects(dump: &std::path::Path, offset: u64) -> Vec<(String, String)> {
    let mut file = std::fs::File::open(dump).expect("open dump");
    file.seek(std::io::SeekFrom::Start(offset)).expect("seek");
    let decoder = bzip2::read::BzDecoder::new(std::io::BufReader::with_capacity(1 << 20, file));
    let mut reader =
        quick_xml::Reader::from_reader(std::io::BufReader::with_capacity(1 << 20, decoder));
    reader.config_mut().check_end_names = false;
    let mut buf = Vec::new();
    let mut path: Vec<Vec<u8>> = Vec::new();
    let mut title = String::new();
    let mut redirect: Option<String> = None;
    let mut pairs = Vec::new();
    loop {
        buf.clear();
        let event = match reader.read_event_into(&mut buf) {
            Ok(event) => event,
            Err(e) => panic!("xml parse failed at offset {offset}: {e}"),
        };
        use quick_xml::events::Event;
        match event {
            Event::Eof => break,
            Event::Start(start) => {
                let name = start.name().as_ref().to_vec();
                if name == b"page" {
                    title.clear();
                    redirect = None;
                    path.clear();
                }
                path.push(name);
            }
            Event::Empty(empty) => {
                if empty.name().as_ref() == b"redirect" {
                    for attribute in empty.attributes().flatten() {
                        if attribute.key.as_ref() == b"title"
                            && let Ok(value) = attribute.decoded_and_normalized_value(
                                quick_xml::XmlVersion::Explicit1_0,
                                reader.decoder(),
                            )
                        {
                            redirect = Some(value.into_owned());
                        }
                    }
                }
            }
            Event::Text(data) => {
                if let [page, field] = path.as_slice()
                    && page == b"page"
                    && field == b"title"
                {
                    title.push_str(&data.decode().expect("xml text"));
                }
            }
            Event::End(end) => {
                if end.name().as_ref() == b"page"
                    && let (Some(source), Some(target)) = (
                        title.strip_prefix("Template:"),
                        redirect.as_deref().and_then(|t| t.strip_prefix("Template:")),
                    )
                {
                    pairs.push((source.to_string(), target.to_string()));
                }
                path.pop();
            }
            _ => {}
        }
    }
    pairs
}

// ---------------------------------------------------------------- languages

/// The three code-to-canonical-name JSON maps, merged with a kind
/// column: language, etymology (etymology-only languages), family.
fn render_languages(languages: &str, etymology: &str, families: &str, date: &str) -> String {
    let mut out = provenance_header(
        "language-code vocabulary",
        "the code-to-canonical-name JSON modules (languages,\n\
         # etymology languages, families)",
        date,
    );
    out.push_str("{\n    codes: [\n        [ code kind name ];\n");
    for (kind, payload) in [
        ("language", languages),
        ("etymology", etymology),
        ("family", families),
    ] {
        let map: BTreeMap<String, String> =
            serde_json::from_str(payload).expect("code-to-name JSON parses");
        for (code, name) in map {
            out.push_str(&format!(
                "        [ {} {} {} ]\n",
                nuon_str(&code),
                nuon_str(kind),
                nuon_str(&name)
            ));
        }
    }
    out.push_str("    ]\n}\n");
    out
}

// ---------------------------------------------------------------- lua scan

/// One top-level value inside a Lua table constructor.
#[derive(Debug, Clone)]
enum LuaValue {
    Str(String),
    Bool(bool),
    Number,
    Nil,
    Ident(String),
    List(Vec<LuaValue>),
}

/// Split a Lua table body into top-level items, respecting strings,
/// nested braces, and comments. Returns (positional, named) values.
fn parse_lua_body(body: &str) -> (Vec<LuaValue>, Vec<(String, LuaValue)>) {
    let items = split_top_level(body);
    let mut positional = Vec::new();
    let mut named = Vec::new();
    for item in items {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        match top_level_assign(trimmed) {
            Some((key, value)) => {
                let key = key
                    .trim()
                    .trim_start_matches("[\"")
                    .trim_end_matches("\"]")
                    .trim()
                    .to_string();
                named.push((key, parse_lua_value(value.trim())));
            }
            None => positional.push(parse_lua_value(trimmed)),
        }
    }
    (positional, named)
}

/// The position of a top-level `=` (outside strings and braces).
fn top_level_assign(item: &str) -> Option<(&str, &str)> {
    let bytes = item.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    let mut in_string: Option<u8> = None;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(quote) = in_string {
            if byte == b'\\' {
                index += 2;
                continue;
            }
            if byte == quote {
                in_string = None;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' | b'\'' => in_string = Some(byte),
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b'=' if depth == 0 => {
                // `==` is comparison, not assignment.
                if index + 1 < bytes.len() && bytes[index + 1] == b'=' {
                    index += 2;
                    continue;
                }
                return Some((&item[..index], &item[index + 1..]));
            }
            _ => {}
        }
        index += 1;
    }
    None
}

/// Truncate a scalar line at a top-level `--` comment, then drop the
/// trailing comma a one-line entry carries.
fn scalar_cleanup(text: &str) -> &str {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut in_string: Option<u8> = None;
    let mut end = text.len();
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(quote) = in_string {
            if byte == b'\\' {
                index += 2;
                continue;
            }
            if byte == quote {
                in_string = None;
            }
            index += 1;
            continue;
        }
        if byte == b'"' || byte == b'\'' {
            in_string = Some(byte);
        } else if byte == b'-' && index + 1 < bytes.len() && bytes[index + 1] == b'-' {
            end = index;
            break;
        }
        index += 1;
    }
    text[..end].trim().trim_end_matches(',').trim()
}

fn parse_lua_value(text: &str) -> LuaValue {
    if text.trim().starts_with('{') {
        let trimmed = text.trim();
        let inner = &trimmed[1..trimmed.rfind('}').unwrap_or(trimmed.len())];
        let (positional, _) = parse_lua_body(inner);
        return LuaValue::List(positional);
    }
    let trimmed = scalar_cleanup(text);
    if trimmed == "true" {
        return LuaValue::Bool(true);
    }
    if trimmed == "false" {
        return LuaValue::Bool(false);
    }
    if trimmed == "nil" {
        return LuaValue::Nil;
    }
    if trimmed.starts_with('"') || trimmed.starts_with('\'') {
        return LuaValue::Str(lua_string(trimmed));
    }
    if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-') {
        return LuaValue::Number;
    }
    LuaValue::Ident(trimmed.to_string())
}

/// A Lua quoted string's content (escapes resolved, concatenation
/// of adjacent literals via `..` joined).
fn lua_string(text: &str) -> String {
    let mut out = String::new();
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'"' || byte == b'\'' {
            let quote = byte;
            index += 1;
            while index < bytes.len() && bytes[index] != quote {
                if bytes[index] == b'\\' && index + 1 < bytes.len() {
                    let escaped = bytes[index + 1];
                    match escaped {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        other => out.push(other as char),
                    }
                    index += 2;
                    continue;
                }
                let rest = &text[index..];
                let c = rest.chars().next().expect("in-bounds");
                out.push(c);
                index += c.len_utf8();
            }
            index += 1;
        } else {
            index += 1;
        }
    }
    out
}

/// Split a table body on top-level commas; strings, braces, and
/// line comments respected.
fn split_top_level(body: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let bytes = body.as_bytes();
    let mut index = 0usize;
    let mut depth = 0usize;
    let mut in_string: Option<u8> = None;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(quote) = in_string {
            if byte == b'\\' && index + 1 < bytes.len() {
                current.push_str(&body[index..index + 2]);
                index += 2;
                continue;
            }
            if byte == quote {
                in_string = None;
            }
            let c = body[index..].chars().next().expect("in-bounds");
            current.push(c);
            index += c.len_utf8();
            continue;
        }
        if byte == b'-' && index + 1 < bytes.len() && bytes[index + 1] == b'-' {
            // A comment runs to end of line (long comments do not
            // appear inside the scanned tables).
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        match byte {
            b'"' | b'\'' => {
                in_string = Some(byte);
                current.push(byte as char);
                index += 1;
            }
            b'{' => {
                depth += 1;
                current.push('{');
                index += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                current.push('}');
                index += 1;
            }
            b',' if depth == 0 => {
                items.push(std::mem::take(&mut current));
                index += 1;
            }
            _ => {
                let c = body[index..].chars().next().expect("in-bounds");
                current.push(c);
                index += c.len_utf8();
            }
        }
    }
    if !current.trim().is_empty() {
        items.push(current);
    }
    items
}

/// Every `PREFIX["KEY"] = VALUE` entry in a Lua source, where VALUE
/// is a balanced `{...}` block or a one-line scalar. Returns
/// (key, raw value text) pairs in order.
fn lua_entries<'a>(source: &'a str, prefix: &str) -> Vec<(String, &'a str)> {
    let mut entries = Vec::new();
    let opener = format!("{prefix}[\"");
    let mut rest_start = 0usize;
    while let Some(found) = source[rest_start..].find(&opener) {
        let key_start = rest_start + found + opener.len();
        // Only line-initial entries: the prefix must follow a newline
        // (plus indentation) so prose mentions do not match.
        let line_ok = source[..rest_start + found]
            .rfind('\n')
            .map(|nl| source[nl + 1..rest_start + found].trim().is_empty())
            .unwrap_or(true);
        let Some(key_end) = source[key_start..].find("\"]").map(|o| key_start + o) else {
            break;
        };
        let key = &source[key_start..key_end];
        let after = &source[key_end + 2..];
        let Some(eq) = after.find('=') else {
            rest_start = key_end;
            continue;
        };
        if !after[..eq].trim().is_empty() || !line_ok {
            rest_start = key_end;
            continue;
        }
        let value_text = after[eq + 1..].trim_start();
        let value_offset = value_text.as_ptr() as usize - source.as_ptr() as usize;
        let value_end = if value_text.starts_with('{') {
            balanced_block_end(value_text)
        } else {
            value_text.find('\n').unwrap_or(value_text.len())
        };
        entries.push((
            key.to_string(),
            &source[value_offset..value_offset + value_end],
        ));
        rest_start = value_offset + value_end;
    }
    entries
}

/// The byte length of the balanced `{...}` block at the text's head.
fn balanced_block_end(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    let mut in_string: Option<u8> = None;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(quote) = in_string {
            if byte == b'\\' {
                index += 2;
                continue;
            }
            if byte == quote {
                in_string = None;
            }
            index += 1;
            continue;
        }
        if byte == b'-' && index + 1 < bytes.len() && bytes[index + 1] == b'-' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        match byte {
            b'"' | b'\'' => in_string = Some(byte),
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return index + 1;
                }
            }
            _ => {}
        }
        index += 1;
    }
    text.len()
}

/// The interior of a `{...}` block (the outer braces stripped).
fn block_interior(value: &str) -> &str {
    let trimmed = value.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

/// Flatten `[[target|display]]` and `[[target]]` wikilinks to their
/// display text.
fn flatten_wikilinks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(open) = rest.find("[[") {
        out.push_str(&rest[..open]);
        let Some(close) = rest[open..].find("]]").map(|o| open + o) else {
            out.push_str(&rest[open..]);
            return out;
        };
        let inner = &rest[open + 2..close];
        let display = match inner.rsplit_once('|') {
            Some((_, display)) => display,
            None => match inner.rsplit_once('#') {
                Some((page, _)) if !page.is_empty() => page,
                _ => inner,
            },
        };
        out.push_str(display);
        rest = &rest[close + 2..];
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------- form of

/// The inflection-tag map: canonical tags with display forms, and
/// shortcut expansions (a shortcut expands to one or more tags; a
/// multipart value like "1//2" stays one element).
fn render_tags(data_1: &str, data_2: &str, lang_en: &str, date: &str) -> String {
    let mut tag_rows: Vec<(String, String)> = Vec::new();
    let mut shortcut_rows: Vec<(String, Vec<String>)> = Vec::new();
    for source in [data_1, data_2, lang_en] {
        for (name, value) in lua_entries(source, "tags") {
            let (positional, named) = parse_lua_body(block_interior(value));
            let display = named
                .iter()
                .find(|(key, _)| key == "display")
                .and_then(|(_, value)| match value {
                    LuaValue::Str(text) => Some(flatten_wikilinks(text)),
                    _ => None,
                })
                .unwrap_or_else(|| name.clone());
            tag_rows.push((name.clone(), display));
            // Positional slot 3 carries the tag's shortcut aliases.
            if let Some(slot) = positional.get(2) {
                match slot {
                    LuaValue::Str(alias) => {
                        shortcut_rows.push((alias.clone(), vec![name.clone()]));
                    }
                    LuaValue::List(aliases) => {
                        for alias in aliases {
                            if let LuaValue::Str(alias) = alias {
                                shortcut_rows.push((alias.clone(), vec![name.clone()]));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        for (shortcut, value) in lua_entries(source, "shortcuts") {
            match parse_lua_value(value.trim()) {
                LuaValue::Str(expansion) => {
                    shortcut_rows.push((shortcut, vec![expansion]));
                }
                LuaValue::List(items) => {
                    let expansion: Vec<String> = items
                        .into_iter()
                        .filter_map(|item| match item {
                            LuaValue::Str(text) => Some(text),
                            _ => None,
                        })
                        .collect();
                    if !expansion.is_empty() {
                        shortcut_rows.push((shortcut, expansion));
                    }
                }
                _ => {}
            }
        }
    }
    tag_rows.sort();
    tag_rows.dedup();
    shortcut_rows.sort();
    shortcut_rows.dedup();

    let mut out = provenance_header(
        "inflection-tag vocabulary",
        "Module:form of/data/1, /data/2, and /lang-data/en",
        date,
    );
    out.push_str("{\n    tags: [\n        [ name display ];\n");
    for (name, display) in &tag_rows {
        out.push_str(&format!(
            "        [ {} {} ]\n",
            nuon_str(name),
            nuon_str(display)
        ));
    }
    out.push_str("    ],\n    shortcuts: [\n        [ shortcut expansion ];\n");
    for (shortcut, expansion) in &shortcut_rows {
        let list: Vec<String> = expansion.iter().map(|item| nuon_str(item)).collect();
        out.push_str(&format!(
            "        [ {} [{}] ]\n",
            nuon_str(shortcut),
            list.join(" ")
        ));
    }
    out.push_str("    ]\n}\n");
    out
}

// ---------------------------------------------------------------- place

fn named_str(named: &[(String, LuaValue)], key: &str) -> Option<String> {
    named.iter().find(|(k, _)| k == key).and_then(|(_, value)| match value {
        LuaValue::Str(text) => Some(text.clone()),
        _ => None,
    })
}

fn named_flag(named: &[(String, LuaValue)], key: &str) -> bool {
    named
        .iter()
        .any(|(k, value)| k == key && matches!(value, LuaValue::Bool(true)))
}

/// The place data: placetype aliases and qualifiers, per-placetype
/// render keys (preposition, affix, plural, fallback,
/// holonym_use_the), placename article rows and patterns, and the
/// locations that carry article or alias data.
fn render_place(placetypes_src: &str, locations_src: &str, date: &str) -> String {
    // The section between two export markers, so entry scans stay
    // inside their own table.
    let section = |src: &str, start: &str, end: &str| -> String {
        let from = src.find(start).unwrap_or_else(|| panic!("{start} not found"));
        let to = src[from..]
            .find(end)
            .map(|o| from + o)
            .unwrap_or(src.len());
        src[from..to].to_string()
    };

    let aliases_src = section(
        placetypes_src,
        "export.placetype_aliases = {",
        "\nexport.placetype_qualifiers",
    );
    let mut aliases: Vec<(String, String)> = Vec::new();
    for (alias, value) in lua_entries(&aliases_src, "") {
        if let LuaValue::Str(full) = parse_lua_value(value.trim()) {
            aliases.push((alias, full));
        }
    }

    let qualifiers_src = section(
        placetypes_src,
        "export.placetype_qualifiers = {",
        "\nexport.former_qualifiers",
    );
    let mut qualifiers: Vec<(String, String, String)> = Vec::new();
    for (name, value) in lua_entries(&qualifiers_src, "") {
        let (display, article) = match parse_lua_value(value.trim()) {
            LuaValue::Bool(_) => (name.clone(), "default"),
            LuaValue::Str(text) => (flatten_wikilinks(&text), "default"),
            LuaValue::Ident(ident) => match ident.as_str() {
                "no_link_def_article" => (name.clone(), "the"),
                "no_link_no_article" => (name.clone(), "none"),
                _ => (name.clone(), "default"),
            },
            _ => (name.clone(), "default"),
        };
        qualifiers.push((name, display, article.to_string()));
    }

    let data_src = section(
        placetypes_src,
        "export.placetype_data = {",
        "\nexport.plural_placetype_to_singular",
    );
    let mut placetype_rows: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for (name, value) in lua_entries(&data_src, "") {
        let (_, named) = parse_lua_body(block_interior(value));
        let mut keys: Vec<(String, String)> = Vec::new();
        for key in ["preposition", "fallback", "affix_type", "suffix", "prefix", "plural", "article"] {
            if let Some(text) = named_str(&named, key) {
                keys.push((key.to_string(), flatten_wikilinks(&text)));
            }
        }
        if named_flag(&named, "holonym_use_the") {
            keys.push((String::from("holonym_use_the"), String::from("true")));
        }
        if !keys.is_empty() {
            placetype_rows.push((name, keys));
        }
    }

    let article_src = section(
        placetypes_src,
        "export.placename_article = {",
        "\nexport.placename_the_re",
    );
    let mut placename_articles: Vec<(String, String)> = Vec::new();
    for (placetype, value) in lua_entries(&article_src, "") {
        for (placename, article_value) in lua_entries(value, "") {
            if let LuaValue::Str(article) = parse_lua_value(article_value.trim())
                && article == "the"
            {
                placename_articles.push((placetype.clone(), placename));
            }
        }
    }

    let the_re_src = section(
        placetypes_src,
        "export.placename_the_re = {",
        "\nexport.cat_implications",
    );
    let mut the_patterns: Vec<(String, String, String)> = Vec::new();
    for (placetype, value) in lua_entries(&the_re_src, "") {
        if let LuaValue::List(patterns) = parse_lua_value(value.trim()) {
            for pattern in patterns {
                let LuaValue::Str(pattern) = pattern else { continue };
                for (kind, text) in translate_lua_pattern(&pattern) {
                    the_patterns.push((placetype.clone(), kind, text));
                }
            }
        }
    }

    // Locations: any ["Name"] = {...} row carrying the/alias_of/
    // display, across every table in the module. display = true
    // canonicalizes the display to the alias target; a string value
    // displays that string; absent means the alias categorizes only.
    let mut locations: Vec<(String, bool, String, bool, String)> = Vec::new();
    for (name, value) in lua_entries(locations_src, "") {
        if !value.trim_start().starts_with('{') {
            continue;
        }
        let (_, named) = parse_lua_body(block_interior(value));
        let the = named_flag(&named, "the");
        let alias_of = named_str(&named, "alias_of").unwrap_or_default();
        let display_expand = named_flag(&named, "display");
        let display_as = named_str(&named, "display").unwrap_or_default();
        if the || !alias_of.is_empty() || display_expand || !display_as.is_empty() {
            locations.push((name, the, alias_of, display_expand, display_as));
        }
    }
    locations.sort();
    locations.dedup();

    let mut out = provenance_header(
        "place vocabulary",
        "Module:place/placetypes and Module:place/locations",
        date,
    );
    out.push_str("{\n    aliases: [\n        [ alias full ];\n");
    for (alias, full) in &aliases {
        out.push_str(&format!(
            "        [ {} {} ]\n",
            nuon_str(alias),
            nuon_str(full)
        ));
    }
    out.push_str("    ],\n    qualifiers: [\n        [ name display article ];\n");
    for (name, display, article) in &qualifiers {
        out.push_str(&format!(
            "        [ {} {} {} ]\n",
            nuon_str(name),
            nuon_str(display),
            nuon_str(article)
        ));
    }
    out.push_str(
        "    ],\n    placetypes: [\n        [ name preposition fallback affix_type \
         affix plural article holonym_use_the ];\n",
    );
    for (name, keys) in &placetype_rows {
        let get = |key: &str| -> String {
            keys.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        // suffix and prefix collapse into one affix column beside
        // affix_type; an explicit suffix/prefix key implies its type
        // when affix_type is absent.
        let mut affix_type = get("affix_type");
        let mut affix = String::new();
        let suffix = get("suffix");
        let prefix = get("prefix");
        if !suffix.is_empty() {
            affix = suffix;
            if affix_type.is_empty() {
                affix_type = String::from("suf");
            }
        } else if !prefix.is_empty() {
            affix = prefix;
            if affix_type.is_empty() {
                affix_type = String::from("pref");
            }
        }
        out.push_str(&format!(
            "        [ {} {} {} {} {} {} {} {} ]\n",
            nuon_str(name),
            nuon_str(&get("preposition")),
            nuon_str(&get("fallback")),
            nuon_str(&affix_type),
            nuon_str(&affix),
            nuon_str(&get("plural")),
            nuon_str(&get("article")),
            if keys.iter().any(|(k, _)| k == "holonym_use_the") {
                "true"
            } else {
                "false"
            }
        ));
    }
    out.push_str("    ],\n    placename_articles: [\n        [ placetype name ];\n");
    for (placetype, name) in &placename_articles {
        out.push_str(&format!(
            "        [ {} {} ]\n",
            nuon_str(placetype),
            nuon_str(name)
        ));
    }
    out.push_str("    ],\n    the_patterns: [\n        [ placetype kind text ];\n");
    for (placetype, kind, text) in &the_patterns {
        out.push_str(&format!(
            "        [ {} {} {} ]\n",
            nuon_str(placetype),
            nuon_str(kind),
            nuon_str(text)
        ));
    }
    out.push_str(
        "    ],\n    locations: [\n        [ name the alias_of display_expand \
         display_as ];\n",
    );
    for (name, the, alias_of, display_expand, display_as) in &locations {
        out.push_str(&format!(
            "        [ {} {} {} {} {} ]\n",
            nuon_str(name),
            if *the { "true" } else { "false" },
            nuon_str(alias_of),
            if *display_expand { "true" } else { "false" },
            nuon_str(display_as)
        ));
    }
    out.push_str("    ]\n}\n");
    out
}

/// Translate a Lua holonym pattern to prefix/suffix/contains rows; a
/// two-letter `[Xx]` class expands to both variants.
fn translate_lua_pattern(pattern: &str) -> Vec<(String, String)> {
    let expanded: Vec<String> = if let (Some(open), Some(close)) =
        (pattern.find('['), pattern.find(']'))
    {
        let class: Vec<char> = pattern[open + 1..close].chars().collect();
        class
            .iter()
            .map(|&c| format!("{}{}{}", &pattern[..open], c, &pattern[close + 1..]))
            .collect()
    } else {
        vec![pattern.to_string()]
    };
    expanded
        .into_iter()
        .map(|p| {
            if let Some(rest) = p.strip_prefix('^') {
                (String::from("prefix"), rest.to_string())
            } else if let Some(rest) = p.strip_suffix('$') {
                (String::from("suffix"), rest.to_string())
            } else {
                (String::from("contains"), p)
            }
        })
        .collect()
}
