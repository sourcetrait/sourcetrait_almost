pub(crate) mod chat;
pub mod run;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
}

pub(crate) use std::{
    collections::hash_map::RandomState,
    hash::{
        BuildHasher,
        Hasher,
    },
    io::{
        self,
    },
    path::PathBuf,
    time::Duration,
};

pub(crate) use sourcetrait_quest_lib as lib;

pub(crate) mod r {
    pub(crate) mod term {
        pub(crate) use ratatui::crossterm::{
            event::{
                self,
                DisableBracketedPaste,
                EnableBracketedPaste,
                Event,
                KeyCode,
                KeyEventKind,
                KeyModifiers,
                KeyboardEnhancementFlags,
                PopKeyboardEnhancementFlags,
                PushKeyboardEnhancementFlags,
            },
            execute,
            terminal::supports_keyboard_enhancement,
        };
    }
}
