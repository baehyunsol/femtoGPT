use super::{
    TokenId,
    Tokenizer,
};
use std::collections::HashMap;

pub type StateMachine = HashMap<StateMachineKey, StateMachineValue>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum StateMachineKey {
    Empty,  // termination
    Byte(u8),
    Bytes(Vec<u8>),  // the state machine is not fully built yet
}

#[derive(Clone)]
pub enum StateMachineValue {
    Id(TokenId),
    Recurs(StateMachine),
}

type Key = StateMachineKey;
type Value = StateMachineValue;

impl Tokenizer {
    /// It builds a state machine each time you call this.
    pub fn encode(&self, s: &[u8]) -> Vec<TokenId> {
        let state_machine = self.build_state_machine();
        self.encode_with_state_machine(s, &state_machine)
    }

    /// If you've already built a state machine, this function is much faster than `encode`.
    /// You can build one with `Tokenizer::build_state_machine()`.
    pub fn encode_with_state_machine(&self, s: &[u8], state_machine: &StateMachine) -> Vec<TokenId> {
        let mut cursor = 0;
        let mut result = vec![];

        while cursor < s.len() {
            let (cursor_, token_id) = step_cursor(s, cursor, &state_machine, self.unk);
            result.push(token_id);
            cursor = cursor_;
        }

        result
    }

    pub fn build_state_machine(&self) -> StateMachine {
        let mut result = StateMachine::new();

        for (token_id, bytes) in self.tokens.iter() {
            let token_id = Value::Id(*token_id);
            let prefix = Key::Byte(bytes[0]);
            let suffix = bytes[1..].to_vec();
            let suffix = match suffix.len() {
                0 => Key::Empty,
                1 => Key::Byte(suffix[0]),
                _ => Key::Bytes(suffix),
            };

            match result.get_mut(&prefix) {
                Some(Value::Recurs(value)) => {
                    value.insert(suffix, token_id);
                },
                None => {
                    result.insert(
                        prefix,
                        Value::Recurs([(suffix, token_id)].into_iter().collect()),
                    );
                },
                _ => unreachable!(),
            }
        }

        for v in result.values_mut() {
            *v = unfold_state_machine(v);
        }

        fold_state_machine(&mut result);
        result
    }
}

// { "bc": 3, "bcd": 4, "cd": 6 } -> { "b": { "c": { "d": 4, "": 3 } }, "c": { "d": 6 } }
fn unfold_state_machine(v: &Value) -> Value {
    let Value::Recurs(v) = v else { unreachable!() };
    let mut result = StateMachine::with_capacity(v.len());

    for (k, v) in v.iter() {
        match k {
            Key::Empty => { result.insert(k.clone(), v.clone()); },
            Key::Byte(_) => {
                match result.get_mut(k) {
                    // k = "b"
                    // { "b": { "c": 3 }, "b": 4 } -> { "b": { "c": 3, "": 4 } }
                    Some(Value::Recurs(value)) => {
                        value.insert(Key::Empty, v.clone());
                    },
                    // k = "b"
                    // { "b": 3, "b": 4 }
                    Some(Value::Id(_)) => unreachable!(),
                    // k = "b"
                    // { "b": 3 } -> { "b": 3 }
                    None => {
                        result.insert(k.clone(), v.clone());
                    },
                }
            },
            Key::Bytes(b) => {
                let prefix = Key::Byte(b[0]);
                let suffix = b[1..].to_vec();
                let suffix = match suffix.len() {
                    0 => Key::Empty,
                    1 => Key::Byte(suffix[0]),
                    _ => Key::Bytes(suffix),
                };

                match result.get_mut(&prefix) {
                    // prefix = "b", suffix = "d"
                    // { "b": { "c": 3 }, "bd": 4 } -> { "b": { "c": 3, "d": 4 } }
                    Some(Value::Recurs(value)) => {
                        value.insert(suffix, v.clone());
                    },
                    // prefix = "b", "suffix" = "c"
                    // { "b": 3, "bc": 4 } -> { "b": { "": 3, "c": 4 } }
                    Some(Value::Id(id)) => {
                        let id = *id;
                        result.insert(
                            prefix,
                            Value::Recurs([
                                (Key::Empty, Value::Id(id)),
                                (suffix, v.clone()),
                            ].into_iter().collect()),
                        );
                    },
                    // prefix = "b", suffix = "c"
                    // { "bc": 3 } -> { "b": { "c": 3 } }
                    None => {
                        result.insert(
                            prefix,
                            Value::Recurs([(suffix, v.clone())].into_iter().collect()),
                        );
                    },
                }
            },
        }
    }

    for v in result.values_mut() {
        if let Value::Recurs(_) = v {
            *v = unfold_state_machine(v);
        }
    }

    fold_state_machine(&mut result);
    Value::Recurs(result)
}

// { "b": { "": 1 }, "c": 2 } -> { "b": 1, "c": 2 }
fn fold_state_machine(s: &mut StateMachine) {
    for v in s.values_mut() {
        match v {
            Value::Recurs(r) => if r.len() == 1 && r.contains_key(&Key::Empty) {
                *v = r.values().collect::<Vec<_>>()[0].clone();
            } else {
                fold_state_machine(r);
            },
            _ => {},
        }
    }
}

fn step_cursor(s: &[u8], cursor: usize, state_machine: &StateMachine, unk_id: TokenId) -> (usize, TokenId) {
    let c = s[cursor];
    let state = state_machine.get(&Key::Byte(c));

    match state {
        None => (cursor + 1, unk_id),
        Some(Value::Id(id)) => (cursor + 1, *id),
        Some(Value::Recurs(r)) => if cursor + 1 == s.len() {
            match r.get(&Key::Empty) {
                Some(Value::Id(id)) => (cursor + 1, *id),
                _ => (cursor + 1, unk_id),
            }
        } else {
            let (new_cursor, id) = step_cursor(s, cursor + 1, r, unk_id);

            if id == unk_id && r.contains_key(&Key::Empty) {
                let Some(Value::Id(id)) = r.get(&Key::Empty) else { unreachable!() };
                (cursor + 1, *id)
            }

            else {
                (new_cursor, id)
            }
        },
    }
}
