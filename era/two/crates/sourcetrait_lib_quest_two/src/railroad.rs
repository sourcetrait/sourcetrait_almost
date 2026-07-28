//! The railroad: a training run's own git repository, laid per run.
use crate::*;

/// The variable naming the throwaway home a railroad is laid under.
pub const TMP_HOME_ENV: &str = "XDGX_TMP_HOME";

/// The railroad root beneath that home.
const RAILROAD_RELATIVE: &str = "quest/railroad";

/// The branch every railroad's history lives on.
pub const BRANCH: &str = "train";

/// What the base commit says; every later one says `REV N`.
const INIT_SUBJECT: &str = "init";

/// The alphabet, in the order that makes a nom sort as its number does.
const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// One training run's identity, minted fresh and never reused.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrainNom(String);

impl TrainNom {
    /// Mint one from the clock, hashed at nanosecond resolution.
    pub fn fresh() -> LibQuestResult<Self> {
        let elapsed = match time::SystemTime::now().duration_since(time::UNIX_EPOCH) {
            Ok(elapsed) => elapsed,
            Err(error) => snafu::whatever!("the clock reads before the epoch: {error}"),
        };
        let hashed = r::hash::xxh3_64(&elapsed.as_nanos().to_be_bytes());
        Ok(Self(base62(hashed)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TrainNom {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A laid railroad: one run's repository, and the nom naming it.
#[derive(Debug, Clone)]
pub struct Railroad {
    nom: TrainNom,
    dir: PathBuf,
}

impl Railroad {
    /// The railroad root, under the home the environment names.
    pub fn home() -> LibQuestResult<PathBuf> {
        let Ok(home) = env::var(TMP_HOME_ENV) else {
            snafu::whatever!("${TMP_HOME_ENV} is not set, so no railroad can be laid");
        };
        snafu::ensure_whatever!(
            !home.trim().is_empty(),
            "${TMP_HOME_ENV} is empty, so no railroad can be laid"
        );
        let home = PathBuf::from(home);
        snafu::ensure_whatever!(
            home.is_dir(),
            "${TMP_HOME_ENV} names {}, which is not a directory",
            home.display()
        );
        Ok(home.join(RAILROAD_RELATIVE))
    }

    /// Lay a fresh railroad under the standing root.
    pub fn lay() -> LibQuestResult<Self> {
        Self::lay_in(&Self::home()?)
    }

    /// Lay one under a root of the caller's choosing.
    pub fn lay_in(root: &Path) -> LibQuestResult<Self> {
        fs::create_dir_all(root)?;
        identity_ready(root)?;
        let nom = TrainNom::fresh()?;
        let dir = root.join(nom.as_str());
        snafu::ensure_whatever!(
            !dir.exists(),
            "a railroad already stands at {}",
            dir.display()
        );
        fs::create_dir_all(&dir)?;
        git(&dir, &["init", "-b", BRANCH])?;
        fs::write(dir.join(".gitignore"), "")?;
        git(&dir, &["add", "--", ".gitignore"])?;
        git(&dir, &["commit", "-m", INIT_SUBJECT])?;
        Ok(Self { nom, dir })
    }

    /// Open one already laid, so a later process can commit to it.
    pub fn at(dir: &Path) -> LibQuestResult<Self> {
        let Some(nom) = dir.file_name().and_then(|nom| nom.to_str()) else {
            snafu::whatever!("{} does not end in a railroad nom", dir.display());
        };
        snafu::ensure_whatever!(
            dir.join(".git").is_dir(),
            "{} carries no repository, so it is not a railroad",
            dir.display()
        );
        Ok(Self {
            nom: TrainNom(nom.to_string()),
            dir: dir.to_path_buf(),
        })
    }

    pub fn nom(&self) -> &TrainNom {
        &self.nom
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Commit everything standing as the next `REV N`, returning N.
    pub fn commit(&self) -> LibQuestResult<usize> {
        git(&self.dir, &["add", "--all"])?;
        let rev = self.next_rev()?;
        git(&self.dir, &["commit", "-m", &format!("REV {rev}")])?;
        Ok(rev)
    }

    /// The number the next commit takes, counted off the history.
    fn next_rev(&self) -> LibQuestResult<usize> {
        let counted = git(&self.dir, &["rev-list", "--count", BRANCH])?;
        match counted.parse::<usize>() {
            Ok(counted) => Ok(counted),
            Err(error) => {
                snafu::whatever!("git counted {counted:?}, which does not parse: {error}")
            }
        }
    }
}

/// Refuse before anything is created if git cannot author a commit.
fn identity_ready(root: &Path) -> LibQuestResult<()> {
    for field in ["user.name", "user.email"] {
        let output = process::Command::new("git")
            .current_dir(root)
            .args(["config", "--get", field])
            .output()?;
        snafu::ensure_whatever!(
            output.status.success()
                && !String::from_utf8_lossy(&output.stdout).trim().is_empty(),
            "git carries no {field}, so a railroad's commits would have no author"
        );
    }
    Ok(())
}

/// Run one git command in a railroad, raising with git's own stderr.
fn git(dir: &Path, args: &[&str]) -> LibQuestResult<String> {
    let output = process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()?;
    if !output.status.success() {
        snafu::whatever!(
            "git {} failed in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// A number as base62 digits, most significant first.
fn base62(mut value: u64) -> String {
    if value == 0 {
        return String::from("0");
    }
    let mut digits = Vec::new();
    while value > 0 {
        digits.push(BASE62[(value % 62) as usize]);
        value /= 62;
    }
    digits.reverse();
    String::from_utf8(digits).expect("the alphabet is ascii")
}
