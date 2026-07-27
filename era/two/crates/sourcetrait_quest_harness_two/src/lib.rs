pub(crate) mod error;
pub mod questness {
    pub mod contract;
    pub mod evaluate;
}

#[allow(unused_imports)]
pub(crate) use std::io;

pub(crate) use nu_protocol::CompareTypes;

pub(crate) use sourcetrait_quest_core::channel::{
    Block,
    Envelope,
    Tag,
};
pub(crate) use sourcetrait_quest_core::nu;

pub(crate) mod r {
    pub(crate) mod nu {
        pub(crate) use nu_parser::parse;
        pub(crate) use nu_protocol::{
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
    QuestHarnessError,
    QuestHarnessResult,
};
pub use crate::questness::contract::{
    NuContract,
    check_agreements,
};
pub use crate::questness::evaluate::QuestnessEvaluator;

#[cfg(test)]
mod tests {
    mod contract;
    mod evaluate;
}
