pub(crate) mod chat;
pub(crate) mod era;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod chat;
}

pub(crate) use std::thread;

pub(crate) use sourcetrait_lib_quest as lib;
pub(crate) use sourcetrait_quest_bridge as bridge;

pub(crate) mod r {
    pub(crate) mod mpsc {
        pub(crate) use tokio::sync::mpsc::{
            Receiver,
            Sender,
            channel,
            error::TryRecvError,
        };
    }
}

pub use crate::era::BridgeTwo;
