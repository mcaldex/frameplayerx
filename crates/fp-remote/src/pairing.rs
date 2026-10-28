//! Pairing a phone with the web remote: the headset shows a short code,
//! and the phone that enters it on the pairing page is given the token.

use crate::RemoteError;
use crate::token::constant_time_eq;
use std::time::{Duration, Instant};

/// Digits in a code.
const DIGITS: u32 = 6;
/// A code stops pairing this long after the headset last showed it, so one
/// still works while you take the headset off to type it.
const LIFE: Duration = Duration::from_secs(10 * 60);
/// Wrong codes in a row after which pairing stops until the headset asks
/// for a new code: a guesser gets five tries, not a million.
const MAX_WRONG: u32 = 5;

/// What entering a code did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entered {
    Paired,
    Wrong,
    /// The headset isn't showing a code, or hasn't for [`LIFE`].
    NoCode,
    /// Too many wrong codes: nothing pairs until the headset asks for a
    /// new code.
    Stopped,
}

#[derive(Debug, Default)]
pub(crate) struct Pairing {
    code: Option<String>,
    /// When the headset last showed the code.
    shown: Option<Instant>,
    wrong: u32,
    stopped: bool,
}

impl Pairing {
    /// The code for the headset to show, made when there is none or it
    /// lapsed; `None` once too many wrong codes stopped pairing.
    pub(crate) fn show(&mut self) -> Result<Option<String>, RemoteError> {
        if self.stopped {
            return Ok(None);
        }
        if !self.live() {
            self.code = Some(new_code()?);
            self.wrong = 0;
        }
        self.shown = Some(Instant::now());
        Ok(self.code.clone())
    }

    /// Pairing again after it stopped, with a new code.
    pub(crate) fn restart(&mut self) {
        *self = Pairing::default();
    }

    fn live(&self) -> bool {
        self.code.is_some() && self.shown.is_some_and(|t| t.elapsed() <= LIFE)
    }

    /// Checks a code typed on a phone (spaces allowed). The right one is
    /// used up: the next phone gets a new code.
    pub(crate) fn enter(&mut self, typed: &str) -> Entered {
        if self.stopped {
            return Entered::Stopped;
        }
        let Some(code) = self.code.as_deref().filter(|_| self.live()) else {
            return Entered::NoCode;
        };
        let digits: String = typed.chars().filter(|c| !c.is_whitespace()).collect();
        if constant_time_eq(digits.as_bytes(), code.as_bytes()) {
            self.code = None;
            self.wrong = 0;
            return Entered::Paired;
        }
        self.wrong += 1;
        if self.wrong >= MAX_WRONG {
            self.stopped = true;
            self.code = None;
        }
        Entered::Wrong
    }
}

/// [`DIGITS`] random digits, each code as likely as any other.
fn new_code() -> Result<String, RemoteError> {
    const RANGE: u32 = 10u32.pow(DIGITS);
    // Below the largest multiple of RANGE that fits, so no code is likelier.
    const LIMIT: u32 = u32::MAX - u32::MAX % RANGE;
    loop {
        let mut b = [0u8; 4];
        getrandom::fill(&mut b).map_err(|e| RemoteError::Random(e.to_string()))?;
        let n = u32::from_le_bytes(b);
        if n < LIMIT {
            return Ok(format!("{:0width$}", n % RANGE, width = DIGITS as usize));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrong_for(code: &str) -> &'static str {
        if code == "000000" { "111111" } else { "000000" }
    }

    #[test]
    fn a_code_pairs_one_phone() {
        let mut p = Pairing::default();
        assert_eq!(p.enter("123456"), Entered::NoCode, "nothing shown yet");
        let code = p.show().unwrap().unwrap();
        assert_eq!(code.len(), 6);
        assert!(code.bytes().all(|b| b.is_ascii_digit()));
        assert_eq!(
            p.show().unwrap().as_deref(),
            Some(code.as_str()),
            "kept while shown"
        );
        let spaced = format!("{} {}", &code[..3], &code[3..]);
        assert_eq!(p.enter(&spaced), Entered::Paired);
        assert_eq!(p.enter(&code), Entered::NoCode, "used up");
        assert!(p.show().unwrap().is_some(), "the next phone gets a new one");
    }

    #[test]
    fn wrong_codes_stop_pairing() {
        let mut p = Pairing::default();
        let code = p.show().unwrap().unwrap();
        for _ in 0..MAX_WRONG {
            assert_eq!(p.enter(wrong_for(&code)), Entered::Wrong);
        }
        assert_eq!(p.enter(&code), Entered::Stopped, "even the right one");
        assert_eq!(p.show().unwrap(), None);
        p.restart();
        let code = p.show().unwrap().unwrap();
        assert_eq!(p.enter(&code), Entered::Paired);
    }

    #[test]
    fn a_code_lapses_when_no_longer_shown() {
        let mut p = Pairing::default();
        let code = p.show().unwrap().unwrap();
        let Some(long_ago) = Instant::now().checked_sub(LIFE + Duration::from_secs(1)) else {
            return;
        };
        p.shown = Some(long_ago);
        assert_eq!(p.enter(&code), Entered::NoCode);
    }

    #[test]
    fn codes_are_random() {
        let codes: std::collections::HashSet<_> = (0..20).map(|_| new_code().unwrap()).collect();
        assert!(codes.len() > 15);
        assert!(codes.iter().all(|c| c.len() == 6));
    }
}
