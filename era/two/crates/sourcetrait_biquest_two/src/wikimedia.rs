//! The WikimediaDumpTool verbs: raw-source page access, the parse
//! test surface, and the ruled typography normalization filter.
use crate::*;

use crate::wikitext::normalize_typography;
use crate::wikitext::parse_blocks;
use crate::wikitext::Block;
use crate::wikitext::Inline;
use crate::wikixml::index_find_title;
use crate::wikixml::open_pages;
use crate::wikixml::read_block;
use crate::wikixml::WikiPage;

/// Find one exact title: index-seeked when an index is given, else a
/// whole-source stream stopping at the first match.
fn find_page(
    title: &str,
    source: &Path,
    index: Option<&PathBuf>,
) -> BiquestResult<Option<WikiPage>> {
    match index {
        Some(index) => {
            let Some(row) = index_find_title(index, title)? else {
                return Ok(None);
            };
            let pages = read_block(source, row.offset)?;
            Ok(pages.into_iter().find(|page| page.title == title))
        }
        None => {
            let mut reader = open_pages(source)?;
            while let Some(page) = reader.next_page()? {
                if page.title == title {
                    return Ok(Some(page));
                }
            }
            Ok(None)
        }
    }
}

/// A found page, or the case-sensitivity-aware error.
fn require_page(
    title: &str,
    source: &Path,
    index: Option<&PathBuf>,
) -> BiquestResult<WikiPage> {
    match find_page(title, source, index)? {
        Some(page) => Ok(page),
        None => snafu::whatever!(
            "title {title:?} is not in {} (enwiktionary ns0 titles are case-sensitive)",
            source.display()
        ),
    }
}

/// `biquest wikimedia page`: one page's wikitext (or its metadata
/// record with --meta) on stdout.
pub(crate) fn wikimedia_page(args: &WikimediaPageArgs) -> BiquestResult<()> {
    let page = require_page(&args.title, &args.source, args.index.as_ref())?;
    if args.meta {
        let record = harness::nu::Value::record(
            harness::nu::record! {
                "title" => v_str(&page.title),
                "ns" => v_int(page.ns),
                "id" => v_int(page.id),
                "redirect" => match &page.redirect {
                    Some(target) => v_str(target),
                    None => harness::nu::Value::nothing(span()),
                },
                "text_bytes" => v_int(page.text.len() as i64),
            },
            span(),
        );
        println!("{}", harness::nu::to_nuon_text(&record)?);
    } else {
        print!("{}", page.text);
    }
    Ok(())
}

/// Reconstruct inlines as display text: near-source, comments gone,
/// entities resolved - the parse's visual round trip.
fn render_inlines(inlines: &[Inline]) -> String {
    let mut out = String::new();
    for inline in inlines {
        match inline {
            Inline::Text(text) => out.push_str(text),
            Inline::Template(template) => {
                out.push_str("{{");
                out.push_str(&template.name);
                for positional in &template.positional {
                    out.push('|');
                    out.push_str(positional);
                }
                for (key, value) in &template.named {
                    out.push('|');
                    out.push_str(key);
                    out.push('=');
                    out.push_str(value);
                }
                out.push_str("}}");
            }
            Inline::Link(link) => {
                out.push_str("[[");
                out.push_str(&link.target);
                if let Some(display) = &link.display {
                    out.push('|');
                    out.push_str(display);
                }
                out.push_str("]]");
                out.push_str(&link.trail);
            }
            Inline::ExternalLink { url, label } => {
                out.push('[');
                out.push_str(url);
                if let Some(label) = label {
                    out.push(' ');
                    out.push_str(label);
                }
                out.push(']');
            }
            Inline::Emphasis(emphasis) => out.push_str(match emphasis {
                crate::wikitext::Emphasis::Italic => "''",
                crate::wikitext::Emphasis::Bold => "'''",
                crate::wikitext::Emphasis::BoldItalic => "'''''",
            }),
            Inline::Ref { attrs, content } => {
                out.push_str("<ref");
                if !attrs.is_empty() {
                    out.push(' ');
                    out.push_str(attrs);
                }
                match content {
                    Some(content) => {
                        out.push('>');
                        out.push_str(content);
                        out.push_str("</ref>");
                    }
                    None => out.push_str("/>"),
                }
            }
            Inline::Nowiki(content) => {
                out.push_str("<nowiki>");
                out.push_str(content);
                out.push_str("</nowiki>");
            }
            Inline::Html { tag, attrs, closing, void } => {
                out.push('<');
                if *closing {
                    out.push('/');
                }
                out.push_str(tag);
                if !attrs.is_empty() {
                    out.push(' ');
                    out.push_str(attrs);
                }
                if *void {
                    out.push('/');
                }
                out.push('>');
            }
        }
    }
    out
}

/// `biquest wikimedia parse`: a page's block tree as a NUON table -
/// kind, level, markers, and the near-source content rendering.
pub(crate) fn wikimedia_parse(args: &WikimediaParseArgs) -> BiquestResult<()> {
    let page = require_page(&args.title, &args.source, args.index.as_ref())?;
    let blocks = parse_blocks(&page.text)?;
    let rows: Vec<harness::nu::Value> = blocks
        .iter()
        .map(|block| {
            let (kind, level, markers, content) = match block {
                Block::Heading { level, content } => {
                    ("heading", Some(*level as i64), String::new(), render_inlines(content))
                }
                Block::ListItem { markers, content } => {
                    ("list_item", None, markers.clone(), render_inlines(content))
                }
                Block::Paragraph { content } => {
                    ("paragraph", None, String::new(), render_inlines(content))
                }
                Block::Table { source } => {
                    ("table", None, String::new(), source.clone())
                }
                Block::HorizontalRule => {
                    ("horizontal_rule", None, String::new(), String::new())
                }
                Block::Blank => ("blank", None, String::new(), String::new()),
            };
            harness::nu::Value::record(
                harness::nu::record! {
                    "kind" => v_str(kind),
                    "level" => match level {
                        Some(level) => v_int(level),
                        None => harness::nu::Value::nothing(span()),
                    },
                    "markers" => v_str(&markers),
                    "content" => v_str(&content),
                },
                span(),
            )
        })
        .collect();
    let rendered =
        harness::nu::to_nuon_pretty(&harness::nu::Value::list(rows, span()))?;
    println!("{rendered}");
    Ok(())
}

/// `biquest wikimedia normalize`: the ruled typography normalization
/// over text or a file, on stdout - the renderer's transform as a
/// standalone test surface.
pub(crate) fn wikimedia_normalize(args: &WikimediaNormalizeArgs) -> BiquestResult<()> {
    let text = match &args.file {
        Some(path) => fs::read_to_string(path)?,
        None => match &args.text {
            Some(text) => text.clone(),
            None => snafu::whatever!("normalize wants TEXT or --file"),
        },
    };
    print!("{}", normalize_typography(&text));
    Ok(())
}
