pub mod channel;
#[cfg(feature = "evaluate")]
pub mod contract;
pub(crate) mod error;
#[cfg(feature = "evaluate")]
pub mod evaluate;
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

#[cfg(feature = "evaluate")]
pub(crate) use nu_protocol::CompareTypes;

#[cfg(feature = "evaluate")]
pub(crate) use crate::channel::{
    Block,
    Envelope,
    Tag,
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
        #[cfg(feature = "evaluate")]
        pub(crate) use nu_protocol::{
            debugger::WithoutDebug,
            engine::Stack,
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
    #[cfg(feature = "evaluate")]
    mod contract;
    #[cfg(feature = "evaluate")]
    mod evaluate;
    mod nu;
    mod template;
}
