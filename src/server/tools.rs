//! 原生 JSON 工具调用的解析、校验与 OpenAI 兼容序列化契约。
//!
//! 本模块不执行工具，也不生成模型文本；工具参数可能包含用户输入，
//! 错误信息只使用固定描述，调用方不得把参数、提示词或模型响应写入日志与指标。

use std::collections::HashSet;

use serde_json::{json, Value};

/// 调用方指定的工具选择策略。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// 禁止调用任何工具。
    None,
    /// 由模型自行决定是否调用。
    Auto,
    /// 必须至少调用一个工具。
    Required,
    /// 必须调用指定名称的工具。
    Function(String),
}

/// 一个可供模型选择的函数工具定义。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    /// 参数的 JSON Schema 对象；本模块只检查结构，不做完整 schema 校验。
    pub parameters: Value,
}

/// 模型生成的一次工具调用；`arguments` 约定为 JSON 对象。
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// 递归判断参数 schema 是否含有本实现不支持的组合/引用关键字。
fn has_unsupported_keyword(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object
                .keys()
                .any(|key| matches!(key.as_str(), "$ref" | "oneOf" | "anyOf" | "allOf"))
                || object.values().any(has_unsupported_keyword)
        }
        Value::Array(items) => items.iter().any(has_unsupported_keyword),
        _ => false,
    }
}

/// 检查直接构造的工具定义是否仍满足边界约束。
fn check_definitions(tools: &[ToolDefinition]) -> Result<(), String> {
    let mut seen: HashSet<&str> = HashSet::new();
    for tool in tools {
        if tool.name.trim().is_empty() {
            return Err("tool definition name must not be empty".to_string());
        }
        if !seen.insert(tool.name.as_str()) {
            return Err("tool definition names must be unique".to_string());
        }
        if !tool.parameters.is_object() {
            return Err("tool parameters must be a JSON object".to_string());
        }
        if has_unsupported_keyword(&tool.parameters) {
            return Err("tool parameters contain an unsupported schema keyword".to_string());
        }
    }
    Ok(())
}

/// 解析 OpenAI `tools` 数组；`None` 表示请求未提供工具。
///
/// 支持 `description` 缺省或为 `null`；`parameters` 缺省时视为空对象。
/// 非 function 类型、缺失名称、重名、非对象参数以及 `$ref`/`oneOf`/`anyOf`/`allOf`
/// 均拒绝，避免把无法约束的输出传给模型。
pub fn parse_tools(value: Option<&Value>) -> Result<Vec<ToolDefinition>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| "tools must be an array".to_string())?;
    let mut tools = Vec::with_capacity(items.len());
    let mut seen: HashSet<String> = HashSet::new();
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| "each tool must be an object".to_string())?;
        if object.get("type").and_then(Value::as_str) != Some("function") {
            return Err("only function tools are supported".to_string());
        }
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| "tool function must be an object".to_string())?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "tool function name must be a string".to_string())?;
        if name.trim().is_empty() {
            return Err("tool function name must not be empty".to_string());
        }
        if !seen.insert(name.to_owned()) {
            return Err("tool definition names must be unique".to_string());
        }
        let description = match function.get("description") {
            Some(Value::String(text)) => Some(text.clone()),
            Some(Value::Null) | None => None,
            _ => return Err("tool description must be a string or null".to_string()),
        };
        let parameters = match function.get("parameters") {
            None => json!({}),
            Some(value @ Value::Object(_)) => value.clone(),
            Some(_) => return Err("tool parameters must be a JSON object".to_string()),
        };
        if has_unsupported_keyword(&parameters) {
            return Err("tool parameters contain an unsupported schema keyword".to_string());
        }
        tools.push(ToolDefinition {
            name: name.to_owned(),
            description,
            parameters,
        });
    }
    Ok(tools)
}

/// 解析 OpenAI `tool_choice`；`None` 表示未提供，按 `auto` 处理。
pub fn parse_choice(value: Option<&Value>) -> Result<ToolChoice, String> {
    let Some(value) = value else {
        return Ok(ToolChoice::Auto);
    };
    match value {
        Value::String(choice) => match choice.as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" => Ok(ToolChoice::Required),
            _ => Err("unsupported tool_choice value".to_string()),
        },
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err("unsupported tool_choice object".to_string());
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| "tool_choice function must be an object".to_string())?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "tool_choice function name must be a string".to_string())?;
            if name.trim().is_empty() {
                return Err("tool_choice function name must not be empty".to_string());
            }
            Ok(ToolChoice::Function(name.to_owned()))
        }
        _ => Err("unsupported tool_choice value".to_string()),
    }
}

/// 校验工具列表与选择策略是否自洽。
///
/// `ToolChoice::None` 允许携带工具但不产生调用；`parallel` 不限制工具定义数量，
/// 只在解析模型输出时限制调用数量。
pub fn validate_tools(
    tools: &[ToolDefinition],
    choice: &ToolChoice,
    _parallel: bool,
) -> Result<(), String> {
    check_definitions(tools)?;
    match choice {
        ToolChoice::None | ToolChoice::Auto => Ok(()),
        ToolChoice::Required if tools.is_empty() => {
            Err("tool_choice requires tools to be defined".to_string())
        }
        ToolChoice::Required => Ok(()),
        ToolChoice::Function(name) => {
            if tools.is_empty() {
                return Err("tool_choice requires tools to be defined".to_string());
            }
            if !tools.iter().any(|tool| tool.name.as_str() == name.as_str()) {
                return Err("named tool_choice does not match a defined tool".to_string());
            }
            Ok(())
        }
    }
}

/// 从模型输出中的单个调用对象提取名称与对象参数。
fn parse_call_value(value: &Value) -> Result<ToolCall, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "generated tool call must be an object".to_string())?;
    if object
        .keys()
        .any(|key| key.as_str() != "name" && key.as_str() != "arguments")
    {
        return Err("generated tool call has an unsupported shape".to_string());
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "generated tool call name must be a string".to_string())?;
    if name.trim().is_empty() {
        return Err("generated tool call name must not be empty".to_string());
    }
    let arguments = object
        .get("arguments")
        .ok_or_else(|| "generated tool call arguments are missing".to_string())?;
    if !arguments.is_object() {
        return Err("generated tool call arguments must be a JSON object".to_string());
    }
    Ok(ToolCall {
        id: String::new(),
        name: name.to_owned(),
        arguments: arguments.clone(),
    })
}

/// 解析模型生成的工具调用，并按索引生成确定性 ID。
///
/// 接受 `{"tool_calls":[...]}`、单个调用对象或空对象（表示零调用）。
/// 不执行工具，不校验参数是否符合各工具的 schema，也不在错误中回显生成文本。
pub fn parse_generated_calls(
    text: &str,
    tools: &[ToolDefinition],
    choice: &ToolChoice,
    parallel: bool,
) -> Result<Vec<ToolCall>, String> {
    validate_tools(tools, choice, parallel)?;
    let root: Value = serde_json::from_str(text)
        .map_err(|_| "generated tool calls are not valid JSON".to_string())?;
    let root = root
        .as_object()
        .ok_or_else(|| "generated tool calls must be a JSON object".to_string())?
        .clone();
    if root
        .keys()
        .any(|key| !matches!(key.as_str(), "tool_calls" | "name" | "arguments"))
    {
        return Err("generated tool calls have an unsupported shape".to_string());
    }
    let mut calls = if let Some(raw) = root.get("tool_calls") {
        if root.contains_key("name") || root.contains_key("arguments") {
            return Err("generated tool calls have an unsupported shape".to_string());
        }
        let array = raw
            .as_array()
            .ok_or_else(|| "tool_calls must be an array".to_string())?;
        array
            .iter()
            .map(parse_call_value)
            .collect::<Result<Vec<_>, _>>()?
    } else if root.contains_key("name") || root.contains_key("arguments") {
        vec![parse_call_value(&Value::Object(root.clone()))?]
    } else {
        Vec::new()
    };

    if matches!(choice, ToolChoice::None) && !calls.is_empty() {
        return Err("tool calls are disabled by tool choice".to_string());
    }
    if !parallel && calls.len() > 1 {
        return Err("parallel tool calls are disabled".to_string());
    }
    match choice {
        ToolChoice::Required if calls.is_empty() => {
            return Err("tool_choice requires at least one tool call".to_string());
        }
        ToolChoice::Function(name) => {
            if calls.is_empty() {
                return Err("named tool_choice requires at least one tool call".to_string());
            }
            if calls.iter().any(|call| call.name.as_str() != name.as_str()) {
                return Err("generated tool name is not allowed by tool choice".to_string());
            }
        }
        _ => {}
    }
    for call in &calls {
        if !tools
            .iter()
            .any(|tool| tool.name.as_str() == call.name.as_str())
        {
            return Err("generated tool name is not defined".to_string());
        }
    }
    for (index, call) in calls.iter_mut().enumerate() {
        call.id = format!("call_{}", index);
    }
    Ok(calls)
}

/// 将工具调用转换为 OpenAI 非流式 `tool_calls` 形状。
///
/// OpenAI 要求 `function.arguments` 是 JSON 字符串，而不是对象。
pub fn openai_tool_calls(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .map(|call| {
            json!({
                "id": &call.id,
                "type": "function",
                "function": {
                    "name": &call.name,
                    "arguments": call.arguments.to_string(),
                },
            })
        })
        .collect()
}

/// 将参数 JSON 字符串按 Unicode 字符切成 OpenAI 流式增量。
///
/// 每个调用的首个增量带 `function.name`，其余增量只带参数片段；
/// `chunk_size` 为 0 时直接 panic（签名不返回 `Result`），调用方应在边界校验。
/// 调用方不得记录返回的增量内容。
pub fn stream_tool_deltas(calls: &[ToolCall], chunk_size: usize) -> Vec<Value> {
    assert!(chunk_size > 0, "chunk_size must be greater than zero");
    let mut deltas = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        let chars: Vec<char> = call.arguments.to_string().chars().collect();
        for (offset, chunk) in chars.chunks(chunk_size).enumerate() {
            let arguments: String = chunk.iter().copied().collect();
            let function = if offset == 0 {
                json!({ "name": &call.name, "arguments": arguments })
            } else {
                json!({ "arguments": arguments })
            };
            deltas.push(json!({
                "index": index,
                "id": &call.id,
                "type": "function",
                "function": function,
            }));
        }
    }
    deltas
}

/// 生成供响应约束解码使用的 JSON Schema。
///
/// `ToolChoice::None` 只允许空对象；其余策略要求 `tool_calls` 数组，
/// 用名称枚举和 `minItems`/`maxItems` 限制调用数量，且不使用 `$schema`
/// 或本实现不支持的组合关键字。
pub fn tool_response_schema(
    tools: &[ToolDefinition],
    choice: &ToolChoice,
    parallel: bool,
) -> Result<Value, String> {
    validate_tools(tools, choice, parallel)?;
    if matches!(choice, ToolChoice::None) {
        return Ok(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }));
    }
    let allowed_names: Vec<&str> = match choice {
        ToolChoice::Function(name) => vec![name.as_str()],
        _ => tools.iter().map(|tool| tool.name.as_str()).collect(),
    };
    let min_items: usize = match choice {
        ToolChoice::Auto => 0,
        _ => 1,
    };
    let max_items: usize = if parallel {
        tools.len()
    } else if tools.is_empty() {
        0
    } else {
        1
    };
    let item = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "enum": allowed_names },
            "arguments": { "type": "object" },
        },
        "required": ["name", "arguments"],
        "additionalProperties": false,
    });
    Ok(json!({
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": item,
                "minItems": min_items,
                "maxItems": max_items,
            },
        },
        "required": ["tool_calls"],
        "additionalProperties": false,
    }))
}
