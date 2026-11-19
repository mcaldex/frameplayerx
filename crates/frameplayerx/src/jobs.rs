//! Background work: anything that may block (network listings, opening
//! media, scans, update checks) runs on a thread and reports back here.

use crossbeam_channel::{Receiver, Sender, unbounded};

pub struct Jobs<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T: Send + 'static> Jobs<T> {
    pub fn new() -> Jobs<T> {
        let (tx, rx) = unbounded();
        Jobs { tx, rx }
    }

    /// Runs `f` on a new thread; its result is delivered by [`Jobs::poll`].
    pub fn spawn(&self, name: &str, f: impl FnOnce() -> T + Send + 'static) {
        if let Err(e) = self.try_spawn(name, f) {
            log::error!("cannot start job thread: {e}");
        }
    }

    /// Starts a job whose caller needs to handle a thread creation failure.
    pub fn try_spawn(
        &self,
        name: &str,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> std::io::Result<()> {
        let tx = self.tx.clone();
        std::thread::Builder::new()
            .name(format!("fp-job-{name}"))
            .spawn(move || {
                let _ = tx.send(f());
            })
            .map(drop)
    }

    /// Finished results, without blocking.
    pub fn poll(&self) -> Vec<T> {
        self.rx.try_iter().collect()
    }

    /// A sender for long-running workers that report more than once.
    pub fn sender(&self) -> Sender<T> {
        self.tx.clone()
    }
}

impl<T: Send + 'static> Default for Jobs<T> {
    fn default() -> Self {
        Jobs::new()
    }
}
