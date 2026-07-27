//! The singleton model container: one owner, one turn at a time.
//!
//! Reached only from `serve` today, which `run` does not yet wire; see
//! that module's note.
#![allow(dead_code)]
use crate::*;

/// Requests are few and each is answered before the next is taken.
const REQUEST_CAPACITY: usize = 8;

/// What runs a session's work against a loaded model.
///
/// Sync and `?Send` BY DESIGN. The era-two model holds `Rc` handles, so
/// it is not `Send` and one thread must own it for its life - which is
/// why the container takes a FACTORY rather than an engine: the closure
/// crosses to the thread, and what it builds never leaves.
pub(crate) trait Engine {
    fn open(&mut self, options: &bridge::all::ChatOptions) -> Result<bridge::all::EraInfo, String>;

    /// Emit through `chunk` as the answer forms; return the accounting.
    fn turn(
        &mut self,
        text: &str,
        chunk: &mut dyn FnMut(bridge::TurnChunk) -> bool,
    ) -> Result<bridge::all::TurnReport, String>;

    fn reset(&mut self) -> Result<(), String>;
}

/// One unit of work, and where its answer goes.
pub(crate) enum Work {
    Open {
        options: bridge::all::ChatOptions,
        answer: r::tokio::OneShot<Result<bridge::all::EraInfo, String>>,
    },
    Turn {
        text: String,
        chunks: r::tokio::Sender<bridge::TurnChunk>,
        answer: r::tokio::OneShot<Result<bridge::all::TurnReport, String>>,
    },
    Reset {
        answer: r::tokio::OneShot<Result<(), String>>,
    },
}

/// The way to the container. Cloning it does not clone the model.
#[derive(Clone)]
pub(crate) struct ContainerHandle {
    work: r::tokio::Sender<Work>,
}

impl ContainerHandle {
    /// Put the engine on its own thread and hand back the way in.
    ///
    /// THE SERIALISATION POINT IS HERE rather than layered over it: the
    /// thread takes one unit of work at a time, so queueing is the
    /// container's own behaviour and no caller has to arrange it.
    pub(crate) fn spawn<E, F>(make: F) -> Self
    where
        E: Engine,
        F: FnOnce() -> E + Send + 'static,
    {
        let (work, mut inbox) = r::tokio::channel::<Work>(REQUEST_CAPACITY);
        std::thread::Builder::new()
            .name(String::from("quest-model"))
            .spawn(move || {
                let mut engine = make();
                while let Some(unit) = inbox.blocking_recv() {
                    serve_one(&mut engine, unit);
                }
            })
            .expect("the model thread starts");
        Self { work }
    }

    pub(crate) async fn open(
        &self,
        options: bridge::all::ChatOptions,
    ) -> Result<bridge::all::EraInfo, String> {
        let (answer, wait) = r::tokio::oneshot();
        self.dispatch(Work::Open { options, answer }, wait).await
    }

    pub(crate) async fn turn(
        &self,
        text: String,
        chunks: r::tokio::Sender<bridge::TurnChunk>,
    ) -> Result<bridge::all::TurnReport, String> {
        let (answer, wait) = r::tokio::oneshot();
        self.dispatch(
            Work::Turn {
                text,
                chunks,
                answer,
            },
            wait,
        )
        .await
    }

    pub(crate) async fn reset(&self) -> Result<(), String> {
        let (answer, wait) = r::tokio::oneshot();
        self.dispatch(Work::Reset { answer }, wait).await
    }

    /// Hand work over and wait for its answer; a gone container is an
    /// error rather than a hang.
    async fn dispatch<T>(
        &self,
        unit: Work,
        wait: r::tokio::OneShotWait<Result<T, String>>,
    ) -> Result<T, String> {
        if self.work.send(unit).await.is_err() {
            return Err(String::from("the model container is gone"));
        }
        match wait.await {
            Ok(answer) => answer,
            Err(_) => Err(String::from("the model container dropped the work")),
        }
    }
}

/// Run one unit on the engine and answer it.
///
/// A dropped answer channel is ignored: the session that asked has gone,
/// and the work is already done.
fn serve_one<E: Engine>(engine: &mut E, unit: Work) {
    match unit {
        Work::Open { options, answer } => {
            let _ = answer.send(engine.open(&options));
        }
        Work::Turn {
            text,
            chunks,
            answer,
        } => {
            let mut emit = |chunk: bridge::TurnChunk| chunks.blocking_send(chunk).is_ok();
            let report = engine.turn(&text, &mut emit);
            let _ = answer.send(report);
        }
        Work::Reset { answer } => {
            let _ = answer.send(engine.reset());
        }
    }
}
