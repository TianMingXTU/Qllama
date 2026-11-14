//! JSON Schema 约束解码：增量状态机、合法 token 掩码与完成性校验。
//!
//! 支持 `type`（object/array/string/number/integer/boolean/null）、`properties`、
//! `required`、`items`、`enum`、`const`、`additionalProperties: false`、嵌套结构、
//! `minItems` / `maxItems`。组合/引用关键字显式拒绝，不做静默放宽。
//! `apply_mask` 使用有限负哨兵替代被禁 logits，保证既有 sampler 的非有限值检查不会拒绝输入。

use serde_json::{Map, Value};

/// 用于被禁候选的有限 logits 哨兵；保持有限值且不改变允许项。
const MASK_SENTINEL: f32 = f32::MIN / 8.0;

/// 编译后的 JSON Schema 约束。
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchema {
    root: Value,
}

impl JsonSchema {
    /// 编译时根 schema；生成循环据此做词表可达性检查。
    pub fn root(&self) -> &Value {
        &self.root
    }
}

/// 增量 JSON 词法状态机。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonState {
    text: String,
    in_string: bool,
    escaped: bool,
    depth: isize,
    completed: bool,
}

impl JsonState {
    /// 创建空状态。
    pub fn new() -> Self {
        Self {
            text: String::new(),
            in_string: false,
            escaped: false,
            depth: 0,
            completed: false,
        }
    }

    /// 当前已接收字符；用于 token 边界回退与调试。
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 喂入一段 token 文本，只维护词法状态与括号深度；是否合法终态由
    /// [`JsonState::is_complete`] 与 [`JsonState::is_possible`] 判定。
    pub fn push_token(&mut self, token: &str) -> Result<(), String> {
        if self.completed {
            return Err("JSON document is already complete".to_string());
        }
        for ch in token.chars() {
            if self.in_string {
                self.text.push(ch);
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match ch {
                '"' => {
                    self.in_string = true;
                    self.text.push(ch);
                }
                '{' | '[' => {
                    self.depth += 1;
                    self.text.push(ch);
                }
                '}' | ']' => {
                    self.depth -= 1;
                    if self.depth < 0 {
                        return Err("unbalanced JSON bracket".to_string());
                    }
                    self.text.push(ch);
                }
                ch if ch.is_ascii_alphanumeric()
                    || ch.is_ascii_whitespace()
                    || matches!(ch, ':' | ',' | '.' | '-' | '+') =>
                {
                    self.text.push(ch);
                }
                ch if ch.is_alphanumeric() => self.text.push(ch),
                _ => return Err(format!("unexpected JSON character: {ch}")),
            }
        }
        Ok(())
    }

    /// 当前文本是否已经构成单个完整 JSON 值。
    pub fn is_complete(&self) -> bool {
        !self.in_string
            && self.depth == 0
            && serde_json::from_str::<Value>(self.text.trim()).is_ok()
    }

    /// 当前文本是否可能通过追加字符抵达合法 JSON 值。
    ///
    /// 用 `serde_json::Deserializer` 的尾部解析判定：能正常解析即完整值；
    /// 因 EOF 中断即合法前缀；语法错误（非 EOF）即不可能。这比手写惰性 JSON
    /// 文法更可靠，也覆盖转义、unicode 与多字节 token 边界。
    pub fn is_possible(&self) -> bool {
        use serde::de::IgnoredAny;
        let mut stream =
            serde_json::Deserializer::from_str(self.text.trim()).into_iter::<IgnoredAny>();
        match stream.next() {
            Some(Ok(_)) => true,
            Some(Err(error)) => error.is_eof(),
            None => true,
        }
    }

    /// 结束并校验 UTF-8 文本是否为单值 JSON；供流式收尾调用。
    pub fn finish(&mut self) -> Result<Value, String> {
        let value = serde_json::from_str(self.text.trim())
            .map_err(|error| format!("incomplete or invalid JSON: {error}"))?;
        self.completed = true;
        Ok(value)
    }
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

/// 从 `response_format` 中的 schema 值编译约束；不支持关键字返回 400 契约错误。
pub fn from_value(value: &Value) -> Result<JsonSchema, String> {
    validate_schema(value)?;
    Ok(JsonSchema {
        root: value.clone(),
    })
}

/// 递归校验 schema 关键字与结构；拒绝 `$ref` / `oneOf` / `anyOf` / `allOf` / `pattern` 等。
pub fn validate_schema(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "schema must be a JSON object".to_string())?;
    const UNSUPPORTED: &[&str] = &[
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "pattern",
        "not",
        "if",
        "then",
        "else",
        "$defs",
        "definitions",
        "patternProperties",
        "dependentSchemas",
        "unevaluatedProperties",
    ];
    if let Some(key) = UNSUPPORTED.iter().find(|key| object.contains_key(**key)) {
        return Err(format!("unsupported schema keyword: {key}"));
    }
    if let Some(kind) = object.get("type") {
        let valid = match kind {
            Value::String(name) => matches!(
                name.as_str(),
                "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
            ),
            Value::Array(items) => items.iter().all(|item| {
                matches!(
                    item.as_str(),
                    Some("object" | "array" | "string" | "number" | "integer" | "boolean" | "null")
                )
            }),
            _ => false,
        };
        if !valid {
            return Err("unsupported schema type".to_string());
        }
    }
    if let Some(additional) = object.get("additionalProperties") {
        if additional.as_bool() != Some(false) && !additional.is_object() {
            return Err("additionalProperties must be false or a schema object".to_string());
        }
        if additional.is_object() {
            validate_schema(additional)?;
        }
    }
    if let Some(properties) = object.get("properties") {
        for nested in properties
            .as_object()
            .ok_or_else(|| "properties must be an object".to_string())?
            .values()
        {
            validate_schema(nested)?;
        }
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| "required must be an array".to_string())?;
        if required.iter().any(|item| !item.is_string()) {
            return Err("required entries must be strings".to_string());
        }
    }
    if let Some(items) = object.get("items") {
        validate_schema(items)?;
    }
    for key in ["minItems", "maxItems"] {
        if let Some(value) = object.get(key) {
            if value.as_u64().is_none() {
                return Err(format!("{key} must be a nonnegative integer"));
            }
        }
    }
    if let Some(min) = object.get("minItems").and_then(Value::as_u64) {
        if let Some(max) = object.get("maxItems").and_then(Value::as_u64) {
            if min > max {
                return Err("minItems must not exceed maxItems".to_string());
            }
        }
    }
    for key in ["enum", "const"] {
        if let Some(value) = object.get(key) {
            if key == "enum" && value.as_array().is_none() {
                return Err("enum must be an array".to_string());
            }
        }
    }
    Ok(())
}

/// 依据当前 JSON 状态与 schema 屏蔽不可能产生合法文档的词表 token。
///
/// `token_text` 返回词表 id 对应的解码文本；返回 None 的 id 一律屏蔽。
/// 完成态候选不仅要求词法合法，还必须通过 schema 校验——否则标量值
/// （如 `true`）会在 `{"type":"object"}` 约束下成为终态并触发 500。
/// 不改变允许候选的 logits，不使用 `-inf` / `NaN`。
pub fn apply_mask<F>(
    logits: &mut [f32],
    state: &JsonState,
    schema: Option<&JsonSchema>,
    token_text: F,
) where
    F: Fn(usize) -> Option<String>,
{
    for (index, logit) in logits.iter_mut().enumerate() {
        let Some(candidate) = token_text(index) else {
            *logit = MASK_SENTINEL;
            continue;
        };
        if candidate.is_empty()
            || state.is_complete()
            || !candidate_keeps_json_valid(state, schema, &candidate)
        {
            *logit = MASK_SENTINEL;
        }
    }
}

fn candidate_keeps_json_valid(
    state: &JsonState,
    schema: Option<&JsonSchema>,
    candidate: &str,
) -> bool {
    let mut probe = state.clone();
    if probe.push_token(candidate).is_err() {
        return false;
    }
    if probe.is_complete() {
        return match probe.finish() {
            Ok(value) => schema.is_none_or(|schema| validate_json_object(schema, &value).is_ok()),
            Err(_) => false,
        };
    }
    // 未完成态：词法上仍可继续，且必须是 schema 根类型的合法前缀——
    // 否则标量前缀（如 `tr`…`true`）会在 object 约束下走入死路，
    // 掩码放行到 max_tokens 结束时只能产出不可解析的垃圾文本。
    probe.is_possible()
        && schema.is_none_or(|schema| schema_prefix_possible(schema.root(), probe.text()))
}

/// 未完成文本是否仍可能是 schema 根类型的合法前缀。
/// 只做根类型级检查（object 必须（或将）以 `{` 开头，array 同理）；
/// 嵌套约束由完成态的 `validate_json_object` 兜底。
fn schema_prefix_possible(root: &Value, text: &str) -> bool {
    let Some(object) = root.as_object() else {
        return true;
    };
    let opener = match object.get("type").and_then(Value::as_str) {
        Some("object") => '{',
        Some("array") => '[',
        _ => return true,
    };
    let trimmed = text.trim_start();
    trimmed.is_empty() || trimmed.starts_with(opener)
}

/// 依据 schema 类型约束生成「schema 合法性」掩码：非对象/数组等类型
/// 不允许提前终止。`json_object` 输出 `{...}` 是唯一出口；词表缺 `{` 时
/// 掩码后所有 logits 均为哨兵值，上层据此报「schema 对该词表不可满足」。
/// 标量类型（number/string/boolean）走 `JsonState` 词法判定即可。
///
/// 返回 true 表示当前状态在 schema 下仍可能有合法完成路径。
pub fn schema_allows_progress(root: &Value, text: &str) -> bool {
    let object = match root.as_object() {
        Some(object) => object,
        None => return true,
    };
    let kind = object.get("type").and_then(Value::as_str);
    match kind {
        Some("object") | Some("array") => {
            // 结构类型：文本必须包含至少一个起始结构字符才算有进展。
            let opener = if kind == Some("object") { '{' } else { '[' };
            text.contains(opener)
        }
        _ => true,
    }
}

/// 词表中是否存在能推进 schema 类型的 token；不可满足时返回 Err 文本，
/// 上层在请求边界转为 400/明确降级，而不是生成到一半 500。
pub fn check_vocab_support(
    root: &Value,
    token_text_of: &dyn Fn(usize) -> Option<String>,
    vocab: usize,
) -> Result<(), String> {
    let object = match root.as_object() {
        Some(object) => object,
        None => return Ok(()),
    };
    let kind = object.get("type").and_then(Value::as_str);
    let opener = match kind {
        Some("object") => '{',
        Some("array") => '[',
        _ => return Ok(()),
    };
    for id in 0..vocab {
        if let Some(text) = token_text_of(id) {
            if text.contains(opener) {
                return Ok(());
            }
        }
    }
    Err(format!(
        "schema requires type {kind:?} but the model vocabulary has no token containing '{opener}'"
    ))
}

/// 校验最终 JSON 值是否满足已编译 schema。
pub fn validate_json_object(schema: &JsonSchema, value: &Value) -> Result<(), String> {
    validate_value(&schema.root, value)
}

fn validate_value(schema: &Value, value: &Value) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or_else(|| "schema must be an object".to_string())?;
    if let Some(expected) = object.get("const") {
        return (value == expected)
            .then_some(())
            .ok_or_else(|| "value does not match const".to_string());
    }
    if let Some(options) = object.get("enum").and_then(Value::as_array) {
        return options
            .contains(value)
            .then_some(())
            .ok_or_else(|| "value is not in enum".to_string());
    }
    if let Some(kind) = object.get("type").and_then(Value::as_str) {
        let matches = match kind {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => false,
        };
        if !matches {
            return Err(format!("value does not match type {kind}"));
        }
    }
    match value {
        Value::Object(map) => {
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !map.contains_key(key) {
                        return Err(format!("missing required property: {key}"));
                    }
                }
            }
            let properties = object.get("properties").and_then(Value::as_object);
            for (key, child) in map {
                if let Some(child_schema) = properties.and_then(|map| map.get(key)) {
                    validate_value(child_schema, child)?;
                } else if object.get("additionalProperties").and_then(Value::as_bool) == Some(false)
                {
                    return Err(format!("additional property is not allowed: {key}"));
                } else if let Some(child_schema) = object.get("additionalProperties") {
                    validate_value(child_schema, child)?;
                }
            }
        }
        Value::Array(items) => {
            if let Some(min) = object.get("minItems").and_then(Value::as_u64) {
                if items.len() < min as usize {
                    return Err("array has fewer items than minItems".to_string());
                }
            }
            if let Some(max) = object.get("maxItems").and_then(Value::as_u64) {
                if items.len() > max as usize {
                    return Err("array has more items than maxItems".to_string());
                }
            }
            if let Some(item_schema) = object.get("items") {
                for item in items {
                    validate_value(item_schema, item)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// `{"type":"json_object"}` 的宽松根约束：只要求模型输出可解析对象。
pub fn json_object_schema() -> JsonSchema {
    JsonSchema {
        root: Value::Object(Map::from_iter([(
            "type".to_string(),
            Value::String("object".to_string()),
        )])),
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::{
        apply_mask, from_value, json_object_schema, validate_json_object, validate_schema,
        JsonState, MASK_SENTINEL,
    };
    use serde_json::{json, Value};

    fn schema(value: Value) -> super::JsonSchema {
        from_value(&value).expect("schema compiles")
    }

    fn parse(text: &str) -> Value {
        let mut state = JsonState::new();
        state.push_token(text).expect("state accepts text");
        state.finish().expect("state parses text")
    }

    #[test]
    fn accepts_supported_schema_and_rejects_unsupported_keywords() {
        let supported = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}},
            "required": ["name"],
            "additionalProperties": false,
        });
        validate_schema(&supported).unwrap();
        for keyword in ["$ref", "oneOf", "anyOf", "allOf", "pattern"] {
            let value = json!({keyword: "x"});
            assert!(validate_schema(&value)
                .unwrap_err()
                .contains("unsupported schema keyword"));
        }
        assert!(validate_schema(&json!({"type": "bogus"})).is_err());
        assert!(validate_schema(&json!({"minItems": -1})).is_err());
        assert!(validate_schema(&json!({"enum": 1})).is_err());
    }

    #[test]
    fn validates_object_required_and_additional_properties() {
        let schema = schema(
            json!({"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer"}},"required":["name"],"additionalProperties":false}),
        );
        validate_json_object(&schema, &json!({"name":"a","n":1})).unwrap();
        assert!(validate_json_object(&schema, &json!({"n":1})).is_err());
        assert!(validate_json_object(&schema, &json!({"name":"a","extra":1})).is_err());
        assert!(validate_json_object(&schema, &json!({"name":1})).is_err());
    }

    #[test]
    fn validates_nested_arrays_enum_const_and_scalar_types() {
        let nested = schema(json!({
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"enum": ["a", "b"]},
                            "ok": {"const": true}
                        },
                        "required": ["kind"]
                    },
                    "minItems": 1,
                    "maxItems": 2
                }
            }
        }));
        validate_json_object(&nested, &json!({"rows":[{"kind":"a","ok":true}]})).unwrap();
        assert!(validate_json_object(&nested, &json!({"rows":[]})).is_err());
        assert!(validate_json_object(&nested, &json!({"rows":[{"kind":"c"}]})).is_err());
        for (kind, value) in [
            ("string", json!("x")),
            ("number", json!(1.5)),
            ("integer", json!(2)),
            ("boolean", json!(true)),
            ("null", json!(null)),
        ] {
            let compiled = schema(json!({"type": kind}));
            validate_json_object(&compiled, &value).unwrap();
        }
    }

    #[test]
    fn state_handles_streaming_multibyte_and_brackets() {
        assert_eq!(parse("{\"x\":\"你好\"}"), json!({"x":"你好"}));
        let mut state = JsonState::new();
        state.push_token("{\"x\"").unwrap();
        state.push_token(":\"你").unwrap();
        state.push_token("好\"}").unwrap();
        assert_eq!(state.finish().unwrap(), json!({"x":"你好"}));
        let mut invalid = JsonState::new();
        assert!(invalid.push_token("}").is_err());
        assert!(!JsonState::new().is_complete());
        let mut partial = JsonState::new();
        partial.push_token("{\"a\":").unwrap();
        assert!(partial.is_possible());
        assert!(!partial.is_complete());
        let mut whitespace = JsonState::new();
        whitespace.push_token("   ").unwrap();
        assert!(!whitespace.is_complete());
    }

    #[test]
    fn mask_keeps_finite_values_and_filters_invalid_candidates() {
        let state = JsonState::new();
        let mut logits = [3.0, 2.0, 1.0, 0.0];
        let tokens = ["{\"a\":", "坏字", "1}", ""];
        apply_mask(&mut logits, &state, None, |index| {
            tokens.get(index).map(|s| s.to_string())
        });
        assert_eq!(logits[0], 3.0);
        assert_eq!(logits[1], MASK_SENTINEL);
        assert_eq!(logits[2], MASK_SENTINEL);
        assert_eq!(logits[3], MASK_SENTINEL);
        assert!(logits.iter().all(|value| value.is_finite()));
        let mut completed = JsonState::new();
        completed.push_token("{}").unwrap();
        let mut logits = [1.0, 2.0];
        apply_mask(&mut logits, &completed, None, |index| {
            ["{}", " "][index].to_string().into()
        });
        assert_eq!(logits[0], MASK_SENTINEL);
        assert_eq!(logits[1], MASK_SENTINEL);
    }

    #[test]
    fn json_object_mode_requires_object_root() {
        let schema = json_object_schema();
        validate_json_object(&schema, &json!({"ok":true})).unwrap();
        assert!(validate_json_object(&schema, &json!([])).is_err());
    }
}
