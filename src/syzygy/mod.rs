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
