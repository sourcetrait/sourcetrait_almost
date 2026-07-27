pub(crate) mod error;
pub(crate) mod questness {
    pub mod arbitrate;
    pub mod contract;
    pub mod evaluate;
    pub mod turn;
}

#[allow(unused_imports)]
pub(crate) use std::io;

pub(crate) use nu_protocol::CompareTypes;

pub(crate) use sourcetrait_quest_core::channel::{
    Block,
    Envelope,
    Tag,
};
pub(crate) use sourcetrait_quest_core::harness;
pub(crate) use sourcetrait_quest_core::nu;
pub(crate) use crate::questness::turn;
pub(crate) use crate::questness::contract::{
    NuContract,
    check_agreements,
};
pub(crate) use crate::questness::evaluate::QuestnessEvaluator;

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
pub use crate::questness::arbitrate::{
    Questness,
    Step,
};
pub use crate::questness::turn::{
    Answer,
    Assembled,
    Binding,
    Request,
};

#[cfg(test)]
mod tests {
    mod arbitrate;
    mod contract;
    mod evaluate;
    mod turn;
}
