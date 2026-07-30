pub mod channel;
pub mod consts;
pub(crate) mod error;
pub mod nu;
pub(crate) mod questness {
    pub mod arbitrate;
    pub mod config;
    pub mod contract;
    pub mod evaluate;
    pub mod shape;
    pub mod turn;
}
pub mod railroad;
pub mod session;
pub mod syllabus;
pub mod template;

#[allow(unused_imports)]
pub(crate) use std::{
    collections::HashMap,
    env,
    fs,
    io,
    path::{
        Path,
        PathBuf,
    },
    process,
    thread,
    time,
};

pub(crate) use nu_protocol::CompareTypes;
pub(crate) use sourcetrait_lib_quest_llm_two as llm;
pub(crate) use sourcetrait_quest_bridge as bridge;

pub(crate) use crate::channel::{
    Block,
    Envelope,
    Tag,
};
pub(crate) use crate::questness::contract::{
    NuSignature,
    check_agreements,
};
pub(crate) use crate::questness::evaluate::QuestnessEvaluator;
pub(crate) use crate::questness::turn;

pub(crate) mod r {
    pub(crate) mod hash {
        pub(crate) use xxhash_rust::xxh3::xxh3_64;
    }
    pub(crate) mod nu {
        pub(crate) use nu_parser::parse;
        pub(crate) use nu_protocol::{
            ast::Expr,
            debugger::WithoutDebug,
            engine::{
                EngineState,
                Stack,
                StateWorkingSet,
            },
        };
    }
}

pub use crate::error::{
    HarnessQuestError,
    HarnessQuestResult,
};
pub use crate::questness::arbitrate::{
    Questness,
    Step,
};
pub use crate::questness::shape::{
    Shape,
    ShapeMember,
};
pub use crate::questness::turn::{
    Answer,
    Assembled,
    Binding,
    Request,
    thought_turn,
};

#[cfg(test)]
mod tests {
    mod channel;
    mod consts;
    mod nu;
    mod questness {
        mod arbitrate;
        mod config;
        mod contract;
        mod evaluate;
        mod shape;
        mod turn;
    }
    mod railroad;
    mod session;
    mod syllabus;
    mod template;
}
