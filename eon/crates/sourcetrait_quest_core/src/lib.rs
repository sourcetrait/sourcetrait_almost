pub mod channel;
pub(crate) mod error;
#[cfg(feature = "evaluate")]
pub mod evaluate;
pub mod nu;

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
    mod evaluate;
    mod nu;
}
