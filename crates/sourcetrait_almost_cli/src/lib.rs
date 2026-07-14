pub(crate) mod chat;
pub(crate) mod cli;
pub mod run;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
}

pub(crate) use std::{
    io::{
        self,
        Write,
    },
    path::PathBuf,
    time::Duration,
};

pub(crate) use sourcetrait_almost_lib as lib;

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

#[allow(unused_imports)]
pub(crate) use crate::cli::{
    Cli,
    Command,
    SnapshotAction,
};
