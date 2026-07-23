pub mod all;
pub mod cli;
pub mod srvc;
pub mod tui;

mod error;

pub use error::{
    BridgeError,
    BridgeResult,
};
