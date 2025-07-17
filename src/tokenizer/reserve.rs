// This token should never appear in a dataset,
// so it's long and ugly.
pub fn generate_reserved_token() -> String {
    format!("<|reserved_token_{:032x}|>", rand::random::<u128>())
}

// I don't want to use regex. In order to do that,
// I have to include 2 new dependencies: regex and lazy_static.
// Why would I?
pub fn is_reserved_token(s: &str) -> bool {
    s.len() == 51 &&
    s.starts_with("<|reserved_token_") &&
    s.ends_with("|>") &&
    s.get(17..49).unwrap().chars().all(
        |c| '0' <= c && c <= '9' || 'a' <= c && c <= 'f'
    )
}
