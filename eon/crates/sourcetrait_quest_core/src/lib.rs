pub mod channel;
pub mod config;
pub(crate) mod error;
pub mod nu;
pub mod template;

#[allow(unused_imports)]
pub(crate) use std::{
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
};

pub(crate) mod r {
    pub(crate) mod nu {
        pub(crate) use nu_parser::parse;
        pub(crate) use nu_protocol::{
            ast::Expr,
            engine::{
                EngineState,
                StateWorkingSet,
            },
        };
    }
}

pub use crate::error::{
    QuestCoreError,
    QuestCoreResult,
};

#[cfg(test)]
mod tests {
    mod channel;
    mod config;
    mod nu;
    mod template;
}
