//! Lenient reading of server JSON: numbers may arrive as strings, `null`
//! counts as missing, and anything unexpected reads as absent rather than
//! failing the whole document.

use serde_json::Value;

/// The value at `key`, unless missing or `null`.
pub(crate) fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|x| !x.is_null())
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
    .filter(|x| x.is_finite())
}

pub(crate) fn num(v: &Value, key: &str) -> Option<f64> {
    get(v, key).and_then(as_f64)
}

/// A positive pixel count or similar; implausible values read as absent.
pub(crate) fn uint(v: &Value, key: &str) -> Option<u32> {
    num(v, key)
        .filter(|x| *x > 0.0 && *x < 1e7)
        .map(|x| x as u32)
}

/// Trimmed, non-empty text (a number is given as text).
pub(crate) fn text(v: &Value, key: &str) -> Option<String> {
    match get(v, key)? {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// A boolean, also from `0`/`1` and `"true"`/`"yes"`/`"no"`...
pub(crate) fn boolean(v: &Value, key: &str) -> Option<bool> {
    match get(v, key)? {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|x| x != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// The array at `key`, empty when missing or not an array.
pub(crate) fn array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    get(v, key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_leniently() {
        let v = json!({"n": "42", "z": null, "t": "  hi ", "b": "yes", "u": 1e9, "a": 3});
        assert_eq!(num(&v, "n"), Some(42.0));
        assert_eq!(get(&v, "z"), None);
        assert_eq!(text(&v, "t").as_deref(), Some("hi"));
        assert_eq!(text(&v, "n").as_deref(), Some("42"));
        assert_eq!(boolean(&v, "b"), Some(true));
        assert_eq!(uint(&v, "u"), None, "implausible");
        assert!(array(&v, "a").is_empty());
    }
}
