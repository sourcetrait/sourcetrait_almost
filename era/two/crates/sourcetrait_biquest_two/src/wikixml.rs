//! The wikimedia XML page layer: export saves and the pinned dumps,
//! streamed whole or seeked to one multistream block.
use crate::*;

use std::io::BufRead;
use std::io::Seek;

/// One page as the export and dump XML carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WikiPage {
    pub(crate) title: String,
    pub(crate) ns: i64,
    pub(crate) id: i64,
    /// The redirect target, when the page is one.
    pub(crate) redirect: Option<String>,
    /// The latest revision's wikitext, entities resolved.
    pub(crate) text: String,
}

/// A page mid-parse; finish() checks and converts the numeric fields.
#[derive(Default)]
struct PartialPage {
    title: String,
    ns: String,
    id: String,
    redirect: Option<String>,
    text: String,
}

/// Append decoded character data to the field the element stack
/// addresses: title/ns/id directly under page, text under
/// page/revision (the revision's own id sits deeper and never
/// collides).
fn absorb_text(partial: &mut PartialPage, stack: &[Vec<u8>], decoded: &str) {
    match stack {
        [page_el, field] if page_el == b"page" => match field.as_slice() {
            b"title" => partial.title.push_str(decoded),
            b"ns" => partial.ns.push_str(decoded),
            b"id" => partial.id.push_str(decoded),
            _ => {}
        },
        [page_el, revision, text_el]
            if page_el == b"page" && revision == b"revision" && text_el == b"text" =>
        {
            partial.text.push_str(decoded);
        }
        _ => {}
    }
}

impl PartialPage {
    fn finish(self) -> BiquestResult<WikiPage> {
        let ns = match self.ns.trim().parse::<i64>() {
            Ok(value) => value,
            Err(e) => snafu::whatever!("page ns {:?}: {e}", self.ns),
        };
        let id = match self.id.trim().parse::<i64>() {
            Ok(value) => value,
            Err(e) => snafu::whatever!("page id {:?}: {e}", self.id),
        };
        Ok(WikiPage {
            title: self.title,
            ns,
            id,
            redirect: self.redirect,
            text: self.text,
        })
    }
}

/// A streaming page reader over any XML page source: the export
/// schema, the dump, and a rootless multistream block all carry the
/// same page element.
pub(crate) struct PageReader<R: BufRead> {
    reader: quick_xml::Reader<R>,
    buf: Vec<u8>,
}

impl<R: BufRead> PageReader<R> {
    pub(crate) fn new(source: R) -> Self {
        let mut reader = quick_xml::Reader::from_reader(source);
        // A multistream block is a rootless page sequence; nothing
        // read here needs well-formedness past the elements taken.
        reader.config_mut().check_end_names = false;
        Self { reader, buf: Vec::new() }
    }

    /// The next page, or None when the source ends. Fields read by
    /// element position (absorb_text); the redirect target rides its
    /// empty element's title attribute, and entity references arrive
    /// as their own events, resolved as XML 1.0 content.
    pub(crate) fn next_page(&mut self) -> BiquestResult<Option<WikiPage>> {
        use quick_xml::events::Event;
        let mut stack: Vec<Vec<u8>> = Vec::new();
        let mut page: Option<PartialPage> = None;
        loop {
            self.buf.clear();
            let event = match self.reader.read_event_into(&mut self.buf) {
                Ok(event) => event,
                Err(e) => snafu::whatever!("xml parse failed: {e}"),
            };
            match event {
                Event::Eof => {
                    snafu::ensure_whatever!(
                        page.is_none(),
                        "the source ends inside a page"
                    );
                    return Ok(None);
                }
                Event::Start(start) => {
                    let name = start.name().as_ref().to_vec();
                    if name == b"page" {
                        page = Some(PartialPage::default());
                        stack.clear();
                    }
                    stack.push(name);
                }
                Event::Empty(empty) => {
                    if empty.name().as_ref() == b"redirect"
                        && let Some(partial) = page.as_mut()
                    {
                        for attribute in empty.attributes() {
                            let attribute = match attribute {
                                Ok(attribute) => attribute,
                                Err(e) => snafu::whatever!("redirect attribute: {e}"),
                            };
                            if attribute.key.as_ref() == b"title" {
                                let value = match attribute.decoded_and_normalized_value(
                                    quick_xml::XmlVersion::Explicit1_0,
                                    self.reader.decoder(),
                                ) {
                                    Ok(value) => value,
                                    Err(e) => snafu::whatever!("redirect title: {e}"),
                                };
                                partial.redirect = Some(value.into_owned());
                            }
                        }
                    }
                }
                Event::Text(text) => {
                    let Some(partial) = page.as_mut() else { continue };
                    let decoded = match text.decode() {
                        Ok(decoded) => decoded,
                        Err(e) => snafu::whatever!("xml text: {e}"),
                    };
                    absorb_text(partial, &stack, &decoded);
                }
                Event::CData(data) => {
                    let Some(partial) = page.as_mut() else { continue };
                    let decoded = match data.decode() {
                        Ok(decoded) => decoded,
                        Err(e) => snafu::whatever!("xml cdata: {e}"),
                    };
                    absorb_text(partial, &stack, &decoded);
                }
                Event::GeneralRef(general_ref) => {
                    let Some(partial) = page.as_mut() else { continue };
                    let resolved = resolve_general_ref(&general_ref)?;
                    absorb_text(partial, &stack, &resolved);
                }
                Event::End(end) => {
                    if end.name().as_ref() == b"page"
                        && let Some(partial) = page.take()
                    {
                        return Ok(Some(partial.finish()?));
                    }
                    stack.pop();
                }
                _ => {}
            }
        }
    }
}

/// Resolve one general reference: a numeric character reference, or
/// one of the five predefined entities - the complete set for the
/// DTD-less export and dump XML.
fn resolve_general_ref(
    general_ref: &quick_xml::events::BytesRef<'_>,
) -> BiquestResult<String> {
    match general_ref.resolve_char_ref() {
        Ok(Some(c)) => return Ok(c.to_string()),
        Ok(None) => {}
        Err(e) => snafu::whatever!("xml character reference: {e}"),
    }
    let name = match general_ref.decode() {
        Ok(name) => name,
        Err(e) => snafu::whatever!("xml entity reference: {e}"),
    };
    match name.as_ref() {
        "amp" => Ok(String::from("&")),
        "lt" => Ok(String::from("<")),
        "gt" => Ok(String::from(">")),
        "quot" => Ok(String::from("\"")),
        "apos" => Ok(String::from("'")),
        other => snafu::whatever!("unresolvable entity reference &{other};"),
    }
}

/// A buffered reader over a page file, bz2-decoded by extension (a
/// multistream dump decodes across every stream).
fn open_source(path: &Path) -> BiquestResult<Box<dyn BufRead>> {
    let file = fs::File::open(path)?;
    let source: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "bz2") {
        Box::new(io::BufReader::with_capacity(
            1 << 20,
            bzip2::read::MultiBzDecoder::new(io::BufReader::with_capacity(1 << 20, file)),
        ))
    } else {
        Box::new(io::BufReader::with_capacity(1 << 20, file))
    };
    Ok(source)
}

/// Open a page source whole: an export .xml, or a dump .xml.bz2
/// streamed across all of its bz2 streams.
pub(crate) fn open_pages(path: &Path) -> BiquestResult<PageReader<Box<dyn BufRead>>> {
    Ok(PageReader::new(open_source(path)?))
}

/// One multistream index row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexRow {
    /// The bz2 block's byte offset into the multistream dump.
    pub(crate) offset: u64,
    pub(crate) page_id: i64,
    pub(crate) title: String,
}

/// Parse one `offset:pageid:title` index line; colons in the title
/// survive because only the two leading fields split.
pub(crate) fn parse_index_line(line: &str) -> BiquestResult<IndexRow> {
    let mut parts = line.splitn(3, ':');
    let (Some(offset), Some(page_id), Some(title)) =
        (parts.next(), parts.next(), parts.next())
    else {
        snafu::whatever!("index line wants offset:pageid:title, got {line:?}");
    };
    let offset = match offset.parse::<u64>() {
        Ok(value) => value,
        Err(e) => snafu::whatever!("index offset {offset:?}: {e}"),
    };
    let page_id = match page_id.parse::<i64>() {
        Ok(value) => value,
        Err(e) => snafu::whatever!("index page id {page_id:?}: {e}"),
    };
    Ok(IndexRow { offset, page_id, title: title.to_string() })
}

/// Find a title's row by streaming the (bz2 or plain) index file.
pub(crate) fn index_find_title(
    path: &Path,
    title: &str,
) -> BiquestResult<Option<IndexRow>> {
    let source = open_source(path)?;
    for line in source.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let row = parse_index_line(&line)?;
        if row.title == title {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

/// The pages of one multistream block: seek to the index offset and
/// decode exactly one bz2 stream (about a hundred pages).
pub(crate) fn read_block(dump: &Path, offset: u64) -> BiquestResult<Vec<WikiPage>> {
    let mut file = fs::File::open(dump)?;
    file.seek(io::SeekFrom::Start(offset))?;
    let decoder = bzip2::read::BzDecoder::new(io::BufReader::with_capacity(1 << 20, file));
    let mut reader = PageReader::new(io::BufReader::with_capacity(1 << 20, decoder));
    let mut pages = Vec::new();
    while let Some(page) = reader.next_page()? {
        pages.push(page);
    }
    Ok(pages)
}
