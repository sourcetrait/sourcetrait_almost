//! The WikimediaDumpTool verbs: raw-source page access over export
//! saves and the pinned wikimedia dumps.
use crate::*;

use crate::wikixml::index_find_title;
use crate::wikixml::open_pages;
use crate::wikixml::read_block;
use crate::wikixml::WikiPage;

/// Find one exact title: index-seeked when --index is given, else a
/// whole-source stream stopping at the first match.
fn find_page(args: &WikimediaPageArgs) -> BiquestResult<Option<WikiPage>> {
    match &args.index {
        Some(index) => {
            let Some(row) = index_find_title(index, &args.title)? else {
                return Ok(None);
            };
            let pages = read_block(&args.source, row.offset)?;
            Ok(pages.into_iter().find(|page| page.title == args.title))
        }
        None => {
            let mut reader = open_pages(&args.source)?;
            while let Some(page) = reader.next_page()? {
                if page.title == args.title {
                    return Ok(Some(page));
                }
            }
            Ok(None)
        }
    }
}

/// `biquest wikimedia page`: one page's wikitext (or its metadata
/// record with --meta) on stdout.
pub(crate) fn wikimedia_page(args: &WikimediaPageArgs) -> BiquestResult<()> {
    let Some(page) = find_page(args)? else {
        snafu::whatever!(
            "title {:?} is not in {} (enwiktionary ns0 titles are case-sensitive)",
            args.title,
            args.source.display()
        );
    };
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
