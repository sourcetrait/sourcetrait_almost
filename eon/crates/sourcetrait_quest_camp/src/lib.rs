pub(crate) mod chat;
pub(crate) mod error;
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
};

pub(crate) use sourcetrait_quest_bridge as bridge;
pub(crate) use sourcetrait_quest_bridge::all::Era;
pub(crate) use sourcetrait_quest_bridge_two as bridge_two;

pub(crate) use crate::error::CampResult;

pub(crate) mod r {
    pub(crate) mod mpsc {
        pub(crate) use tokio::sync::mpsc::{
            Receiver,
            Sender,
            channel,
        };
    }
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
