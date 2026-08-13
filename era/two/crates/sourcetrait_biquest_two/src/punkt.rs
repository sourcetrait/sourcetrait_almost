//! The nltk Punkt sentence tokenizer, inference path only.
use crate::*;

const ORTHO_BEG_UC: u32 = 1 << 1;
const ORTHO_MID_UC: u32 = 1 << 2;
const ORTHO_UNK_UC: u32 = 1 << 3;
const ORTHO_BEG_LC: u32 = 1 << 4;
const ORTHO_MID_LC: u32 = 1 << 5;
const ORTHO_UNK_LC: u32 = 1 << 6;
const ORTHO_UC: u32 = ORTHO_BEG_UC + ORTHO_MID_UC + ORTHO_UNK_UC;
const ORTHO_LC: u32 = ORTHO_BEG_LC + ORTHO_MID_LC + ORTHO_UNK_LC;

/// PunktLanguageVars._re_non_word_chars (single-character class).
const NON_WORD_CHARS: &str = ")\";}]*:@'({[!?";
/// PunktLanguageVars._re_word_start exclusions.
const WORD_START_EXCLUDED: &str = "(\"`{[:;&#*@)}]-,";
/// PunktSentenceTokenizer.PUNCTUATION.
const SENT_PUNCTUATION: &str = ";:,.!?";
/// Boundary-realignment quote class.
const REALIGN_CHARS: &str = "\"')]}";

pub(crate) struct PunktParams {
    pub(crate) abbrev_types: HashSet<String>,
    pub(crate) collocations: HashSet<(String, String)>,
    pub(crate) sent_starters: HashSet<String>,
    pub(crate) ortho_context: HashMap<String, u32>,
}

impl PunktParams {
    /// Load the punkt_tab english parameter files (nltk tabdata forms).
    pub(crate) fn load(dir: &Path) -> BquestResult<Self> {
        fn data_lines(path: &Path) -> BquestResult<Vec<String>> {
            let text = fs::read_to_string(path)?;
            let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
            if lines.last().is_some_and(|last| last.is_empty()) {
                lines.pop();
            }
            Ok(lines)
        }
        let abbrev_types = data_lines(&dir.join("abbrev_types.txt"))?.into_iter().collect();
        let sent_starters = data_lines(&dir.join("sent_starters.txt"))?.into_iter().collect();
        let mut collocations = HashSet::new();
        for line in data_lines(&dir.join("collocations.tab"))? {
            let Some((first, second)) = line.split_once('\t') else {
                snafu::whatever!("collocations.tab line without a tab: {line:?}");
            };
            collocations.insert((first.to_string(), second.to_string()));
        }
        let mut ortho_context = HashMap::new();
        for line in data_lines(&dir.join("ortho_context.tab"))? {
            let Some((typ, value)) = line.split_once('\t') else {
                snafu::whatever!("ortho_context.tab line without a tab: {line:?}");
            };
            let Ok(flags) = value.parse::<u32>() else {
                snafu::whatever!("ortho_context.tab value not an int: {line:?}");
            };
            ortho_context.insert(typ.to_string(), flags);
        }
        Ok(Self { abbrev_types, collocations, sent_starters, ortho_context })
    }
}

/// A PunktToken: the token text plus its first-pass/second-pass flags.
struct PunktTok {
    tok: String,
    typ: String,
    period_final: bool,
    sentbreak: bool,
    abbr: bool,
    ellipsis: bool,
}

impl PunktTok {
    fn new(tok: String) -> Self {
        let lowered: String = tok.to_lowercase();
        let typ = if numeric_type_matches(&lowered) {
            String::from("##number##")
        } else {
            lowered
        };
        let period_final = tok.ends_with('.');
        Self { tok, typ, period_final, sentbreak: false, abbr: false, ellipsis: false }
    }

    fn type_no_period(&self) -> &str {
        if self.typ.chars().count() > 1 && self.typ.ends_with('.') {
            &self.typ[..self.typ.len() - 1]
        } else {
            &self.typ
        }
    }

    fn type_no_sentperiod(&self) -> &str {
        if self.sentbreak {
            self.type_no_period()
        } else {
            &self.typ
        }
    }

    fn first_char(&self) -> Option<char> {
        self.tok.chars().next()
    }

    fn first_upper(&self) -> bool {
        self.first_char().is_some_and(char::is_uppercase)
    }

    fn first_lower(&self) -> bool {
        self.first_char().is_some_and(char::is_lowercase)
    }

    /// _RE_ELLIPSIS: the token is two or more periods.
    fn is_ellipsis_tok(&self) -> bool {
        self.tok.chars().count() >= 2 && self.tok.chars().all(|c| c == '.')
    }

    /// _RE_INITIAL: one letter-ish word character plus a period.
    fn is_initial(&self) -> bool {
        let chars: Vec<char> = self.tok.chars().collect();
        chars.len() == 2 && py_is_word_nondigit(chars[0]) && chars[1] == '.'
    }
}

/// _RE_NUMERIC on a lowercased token; the pattern is fully anchored.
fn numeric_type_matches(lowered: &str) -> bool {
    let chars: Vec<char> = lowered.chars().collect();
    let mut i = 0usize;
    if i < chars.len() && chars[i] == '-' {
        i += 1;
    }
    if i < chars.len()
        && (chars[i] == '.' || chars[i] == ',')
        && i + 1 < chars.len()
        && py_is_decimal(chars[i + 1])
    {
        i += 1;
    }
    if i >= chars.len() || !py_is_decimal(chars[i]) {
        return false;
    }
    i += 1;
    chars[i..].iter().all(|&c| py_is_decimal(c) || c == ',' || c == '.' || c == '-')
}

/// The multi-char punctuation match at a position, as its length.
fn multi_char_len(chars: &[char], i: usize) -> Option<usize> {
    if i >= chars.len() {
        return None;
    }
    if chars[i] == '-' {
        let mut j = i;
        while j < chars.len() && chars[j] == '-' {
            j += 1;
        }
        if j - i >= 2 {
            return Some(j - i);
        }
        return None;
    }
    if chars[i] == '.' {
        let mut j = i;
        while j < chars.len() && chars[j] == '.' {
            j += 1;
        }
        if j - i >= 2 {
            return Some(j - i);
        }
        let mut reps = 0usize;
        let mut k = i;
        while k < chars.len() && chars[k] == '.' && k + 1 < chars.len() && py_is_space(chars[k + 1])
        {
            k += 2;
            reps += 1;
        }
        while reps >= 2 {
            let end = i + 2 * reps;
            if end < chars.len() && chars[end] == '.' {
                return Some(2 * reps + 1);
            }
            reps -= 1;
        }
        return None;
    }
    None
}

fn is_non_word_char(c: char) -> bool {
    NON_WORD_CHARS.contains(c)
}

/// The word-end lookahead of the punkt word tokenizer at position j.
fn word_end_lookahead(chars: &[char], j: usize) -> bool {
    if j >= chars.len() {
        return true;
    }
    let c = chars[j];
    if py_is_space(c) || is_non_word_char(c) || multi_char_len(chars, j).is_some() {
        return true;
    }
    if c == ',' {
        let k = j + 1;
        return k >= chars.len()
            || py_is_space(chars[k])
            || is_non_word_char(chars[k])
            || multi_char_len(chars, k).is_some();
    }
    false
}

/// PunktLanguageVars.word_tokenize: the findall scan.
fn punkt_word_tokenize(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if let Some(len) = multi_char_len(&chars, i) {
            tokens.push(chars[i..i + len].iter().collect());
            i += len;
            continue;
        }
        let c = chars[i];
        if py_is_space(c) {
            i += 1;
            continue;
        }
        if !WORD_START_EXCLUDED.contains(c) {
            let mut j = i + 1;
            while !word_end_lookahead(&chars, j) {
                j += 1;
            }
            tokens.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        tokens.push(c.to_string());
        i += 1;
    }
    tokens
}

/// PunktBaseClass._tokenize_words, inference shape.
fn tokenize_words(text: &str) -> Vec<PunktTok> {
    let mut tokens = Vec::new();
    for line in text.split('\n') {
        if py_strip_ws(line).is_empty() {
            continue;
        }
        for tok in punkt_word_tokenize(line) {
            tokens.push(PunktTok::new(tok));
        }
    }
    tokens
}

fn first_pass_annotation(params: &PunktParams, tok: &mut PunktTok) {
    if tok.tok == "." || tok.tok == "?" || tok.tok == "!" {
        tok.sentbreak = true;
    } else if tok.is_ellipsis_tok() {
        tok.ellipsis = true;
    } else if tok.period_final && !tok.tok.ends_with("..") {
        let head: String = {
            let chars: Vec<char> = tok.tok.chars().collect();
            chars[..chars.len() - 1].iter().collect::<String>().to_lowercase()
        };
        let hyphen_tail = head.rsplit('-').next().unwrap_or(&head).to_string();
        if params.abbrev_types.contains(&head) || params.abbrev_types.contains(&hyphen_tail) {
            tok.abbr = true;
        } else {
            tok.sentbreak = true;
        }
    }
}

/// PunktSentenceTokenizer._ortho_heuristic: Some(true/false) or None
/// for "unknown".
fn ortho_heuristic(params: &PunktParams, tok: &PunktTok) -> Option<bool> {
    if tok.tok.chars().count() == 1
        && tok.tok.chars().next().is_some_and(|c| SENT_PUNCTUATION.contains(c))
    {
        return Some(false);
    }
    let ortho = *params.ortho_context.get(tok.type_no_sentperiod()).unwrap_or(&0);
    if tok.first_upper() && (ortho & ORTHO_LC != 0) && (ortho & ORTHO_MID_UC == 0) {
        return Some(true);
    }
    if tok.first_lower() && ((ortho & ORTHO_UC != 0) || (ortho & ORTHO_BEG_LC == 0)) {
        return Some(false);
    }
    None
}

/// PunktSentenceTokenizer._second_pass_annotation over (tok1, tok2).
fn second_pass_annotation(params: &PunktParams, tokens: &mut [PunktTok], index: usize) {
    if index + 1 >= tokens.len() {
        return;
    }
    if !tokens[index].period_final {
        return;
    }
    let typ = tokens[index].type_no_period().to_string();
    let next_typ = tokens[index + 1].type_no_sentperiod().to_string();
    let tok_is_initial = tokens[index].is_initial();

    if params.collocations.contains(&(typ.clone(), next_typ.clone())) {
        tokens[index].sentbreak = false;
        tokens[index].abbr = true;
        return;
    }

    if (tokens[index].abbr || tokens[index].ellipsis) && !tok_is_initial {
        match ortho_heuristic(params, &tokens[index + 1]) {
            Some(true) => {
                tokens[index].sentbreak = true;
                return;
            }
            _ => {
                if tokens[index + 1].first_upper()
                    && params.sent_starters.contains(&next_typ)
                {
                    tokens[index].sentbreak = true;
                    return;
                }
            }
        }
    }

    if tok_is_initial || typ == "##number##" {
        match ortho_heuristic(params, &tokens[index + 1]) {
            Some(false) => {
                tokens[index].sentbreak = false;
                tokens[index].abbr = true;
            }
            None => {
                if tok_is_initial
                    && tokens[index + 1].first_upper()
                    && (params.ortho_context.get(&next_typ).unwrap_or(&0) & ORTHO_LC == 0)
                {
                    tokens[index].sentbreak = false;
                    tokens[index].abbr = true;
                }
            }
            Some(true) => {}
        }
    }
}

/// text_contains_sentbreak: any non-final annotated token breaking.
fn text_contains_sentbreak(params: &PunktParams, context: &str) -> bool {
    let mut tokens = tokenize_words(context);
    for tok in tokens.iter_mut() {
        first_pass_annotation(params, tok);
    }
    for index in 0..tokens.len() {
        second_pass_annotation(params, &mut tokens, index);
    }
    let mut found = false;
    for tok in &tokens {
        if found {
            return true;
        }
        if tok.sentbreak {
            found = true;
        }
    }
    false
}

/// One period-context match, as spans into the character vector.
struct PeriodMatch {
    start: usize,
    after_end: usize,
    next_tok: Option<(usize, usize)>,
}

fn period_context_matches(chars: &[char]) -> Vec<PeriodMatch> {
    let mut matches = Vec::new();
    for i in 0..chars.len() {
        let c = chars[i];
        if c != '.' && c != '?' && c != '!' {
            continue;
        }
        let j = i + 1;
        if j < chars.len() && is_non_word_char(chars[j]) {
            matches.push(PeriodMatch { start: i, after_end: j + 1, next_tok: None });
            continue;
        }
        let mut k = j;
        while k < chars.len() && py_is_space(chars[k]) {
            k += 1;
        }
        if k > j && k < chars.len() {
            let tok_start = k;
            let mut tok_end = k;
            while tok_end < chars.len() && !py_is_space(chars[tok_end]) {
                tok_end += 1;
            }
            matches.push(PeriodMatch {
                start: i,
                after_end: tok_end,
                next_tok: Some((tok_start, tok_end)),
            });
        }
    }
    matches
}

fn slice_text(chars: &[char], start: usize, stop: usize) -> String {
    chars[start..stop].iter().collect()
}

/// _get_last_whitespace_index, over ASCII string.whitespace.
fn last_ascii_whitespace_index(chars: &[char], start: usize, stop: usize) -> usize {
    for i in (start..stop).rev() {
        if PY_ASCII_WHITESPACE.contains(chars[i]) {
            return i - start;
        }
    }
    0
}

struct EndContext {
    match_end: usize,
    next_tok_start: Option<usize>,
    context: String,
}

/// _match_potential_end_contexts: overlap-pruned match/context pairs.
fn match_potential_end_contexts(chars: &[char]) -> Vec<EndContext> {
    let mut out = Vec::new();
    let mut previous_slice = (0usize, 0usize);
    let mut previous: Option<PeriodMatch> = None;
    for m in period_context_matches(chars) {
        let mut index_after_last_space =
            last_ascii_whitespace_index(chars, previous_slice.1, m.start);
        if index_after_last_space > 0 {
            index_after_last_space += previous_slice.1 + 1;
        } else {
            index_after_last_space = previous_slice.0;
        }
        let prev_word_slice = (index_after_last_space, m.start);
        if let Some(prev) = previous.take()
            && previous_slice.1 <= prev_word_slice.0 {
                let context = slice_text(chars, previous_slice.0, previous_slice.1)
                    + &slice_text(chars, prev.start, prev.start + 1)
                    + &slice_text(chars, prev.start + 1, prev.after_end);
                out.push(EndContext {
                    match_end: prev.start + 1,
                    next_tok_start: prev.next_tok.map(|(s, _)| s),
                    context,
                });
            }
        previous = Some(m);
        previous_slice = prev_word_slice;
    }
    if let Some(prev) = previous {
        let context = slice_text(chars, previous_slice.0, previous_slice.1)
            + &slice_text(chars, prev.start, prev.start + 1)
            + &slice_text(chars, prev.start + 1, prev.after_end);
        out.push(EndContext {
            match_end: prev.start + 1,
            next_tok_start: prev.next_tok.map(|(s, _)| s),
            context,
        });
    }
    out
}

/// _slices_from_text: sentence spans before realignment.
fn slices_from_text(params: &PunktParams, chars: &[char]) -> Vec<(usize, usize)> {
    let mut slices = Vec::new();
    let mut last_break = 0usize;
    for context in match_potential_end_contexts(chars) {
        if text_contains_sentbreak(params, &context.context) {
            slices.push((last_break, context.match_end));
            last_break = context.next_tok_start.unwrap_or(context.match_end);
        }
    }
    let mut rstripped = chars.len();
    while rstripped > 0 && py_is_space(chars[rstripped - 1]) {
        rstripped -= 1;
    }
    slices.push((last_break, rstripped));
    slices
}

/// re_boundary_realignment's lazy match, as (group_len, match_end).
fn realign_match(chars: &[char], start: usize, stop: usize) -> Option<(usize, usize)> {
    let mut max_run = 0usize;
    while start + max_run < stop && REALIGN_CHARS.contains(chars[start + max_run]) {
        max_run += 1;
    }
    for k in 1..=max_run {
        let tail = start + k;
        if tail < stop && py_is_space(chars[tail]) {
            let mut ws_end = tail;
            while ws_end < stop && py_is_space(chars[ws_end]) {
                ws_end += 1;
            }
            return Some((k, ws_end - start));
        }
        if tail + 1 < stop && chars[tail] == '-' && chars[tail + 1] == '-' {
            return Some((k, k));
        }
        if tail == stop || chars[tail] == '\n' {
            return Some((k, k));
        }
    }
    None
}

/// _realign_boundaries over the sentence slices.
fn realign_boundaries(chars: &[char], slices: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut realign = 0usize;
    for index in 0..slices.len() {
        let sentence1 = (slices[index].0 + realign, slices[index].1);
        let sentence2 = slices.get(index + 1).copied();
        let Some(sentence2) = sentence2 else {
            if sentence1.0 < sentence1.1 {
                out.push(sentence1);
            }
            continue;
        };
        if let Some((group_len, match_end)) = realign_match(chars, sentence2.0, sentence2.1) {
            out.push((sentence1.0, sentence2.0 + group_len));
            realign = match_end;
        } else {
            realign = 0;
            if sentence1.0 < sentence1.1 {
                out.push(sentence1);
            }
        }
    }
    out
}

/// nltk.sent_tokenize with the loaded english parameters.
pub(crate) fn sent_tokenize(params: &PunktParams, text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let slices = slices_from_text(params, &chars);
    realign_boundaries(&chars, slices)
        .into_iter()
        .map(|(start, stop)| slice_text(&chars, start, stop))
        .collect()
}
