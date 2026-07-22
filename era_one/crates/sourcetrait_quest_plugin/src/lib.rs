pub(crate) mod channel;
pub(crate) mod convert;
pub(crate) mod error;
pub(crate) mod gate;
pub(crate) mod typedef;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod channel;
    pub(crate) mod convert;
    pub(crate) mod gate;
    pub(crate) mod typedef;
}

pub(crate) use std::{
    io::{
        self,
        Read,
    },
};

pub(crate) use nu_protocol::CompareTypes;

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

pub(crate) use crate::error::{
    NuonRenderSnafu,
    NuonSnafu,
    TypedefSnafu,
};

pub use crate::{
    channel::{
        extract_output,
        frame_channel,
        HARDCODED_CONFIGURATION,
        INPUT_DATA,
        INPUT_DEFINITION,
        INSUFFICIENCY,
        OUTPUT_DATA,
        OUTPUT_DEFINITION,
        PROMPT_CONFIGURATION_DATA,
        PROMPT_CONFIGURATION_DEFINITION,
    },
    convert::{
        parse_nuon,
        render_nuon,
    },
    error::{
        PluginError,
        PluginResult,
    },
    gate::{
        check,
        check_main,
        CheckReport,
        CheckStage,
    },
    typedef::{
        conforms,
        derive_type,
        parse_typedef,
        render_typedef,
    },
};
pub use nu_protocol::{
    Type,
    Value,
};
