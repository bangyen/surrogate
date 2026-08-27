pub mod utils;

use shakmaty::Chess;
use shakmaty_syzygy::Tablebase;

pub struct SyzygyTablebase {
    pub tb: Tablebase<Chess>,
}

impl SyzygyTablebase {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let mut tb = Tablebase::new();
        tb.add_directory(path)?;
        Ok(Self { tb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rejects_a_missing_directory() {
        // A bad path must surface an error rather than yielding a
        // tablebase that silently answers nothing.
        let err = SyzygyTablebase::new("/nonexistent/syzygy/path");
        assert!(err.is_err(), "missing directory should not load");
    }

    #[test]
    fn test_new_accepts_an_empty_directory() {
        // An existing but empty directory is a legitimate starting state:
        // it loads, holding zero tables.
        let dir = std::env::temp_dir().join(format!("syzygy-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = SyzygyTablebase::new(dir.to_str().unwrap());
        std::fs::remove_dir_all(&dir).ok();

        assert!(
            result.is_ok(),
            "empty directory should load cleanly: {:?}",
            result.err()
        );
    }
}
