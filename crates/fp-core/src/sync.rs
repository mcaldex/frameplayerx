//! Locks that keep working after a thread panicked while holding them.
//!
//! The data FramePlayerX guards stays consistent between statements, so one
//! thread's panic must not take down every other thread sharing it.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub fn lock<T: ?Sized>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn read<T: ?Sized>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(PoisonError::into_inner)
}

pub fn write<T: ?Sized>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_locks_still_lock() {
        let m = Mutex::new(1);
        let l = RwLock::new(2);
        std::thread::scope(|s| {
            let _ = s
                .spawn(|| {
                    let _g = (m.lock(), l.write());
                    panic!("poison both");
                })
                .join();
        });
        assert!(m.is_poisoned() && l.is_poisoned());
        *lock(&m) += 1;
        *write(&l) += 1;
        assert_eq!((*lock(&m), *read(&l)), (2, 3));
    }
}
