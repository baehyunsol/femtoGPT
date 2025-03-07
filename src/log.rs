use rusqlite::Connection;

const CREATE_TABLE: &str = "
CREATE TABLE IF NOT EXISTS train_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_type TEXT,  -- step | dataset_file | save

    -- only for write_log_step
    step INTEGER,
    loss REAL,
    elapsed_ms INTEGER,

    -- only for write_log_dataset_file
    dataset_file TEXT,
    context_size INTEGER,

    timestamp TEXT  -- chrono::Local::now()
);
";

pub fn initialize_log() {
    let conn = Connection::open("train.db").unwrap();
    conn.execute(CREATE_TABLE, ()).unwrap();
}

pub fn write_log_dataset_file(file: &str, context_size: usize) {
    let conn = Connection::open("train.db").unwrap();
    conn.execute(
        "INSERT INTO train_log (log_type, dataset_file, context_size, timestamp) VALUES (?, ?, ?, ?)",
        ("dataset_file", file, context_size, now()),
    ).unwrap();
}

pub fn write_log_step(step: usize, loss: f32, elapsed_ms: u128) {
    let conn = Connection::open("train.db").unwrap();
    conn.execute(
        "INSERT INTO train_log (log_type, step, loss, elapsed_ms, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("step", step, loss, elapsed_ms as u64, now()),
    ).unwrap();
}

pub fn write_log_save() {
    let conn = Connection::open("train.db").unwrap();
    conn.execute(
        "INSERT INTO train_log (log_type, timestamp) VALUES (?, ?)",
        ("save", now()),
    ).unwrap();
}

fn now() -> String {
    chrono::Local::now().to_rfc3339()
}
