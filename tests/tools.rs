//! 原生 JSON 工具调用解析与 OpenAI 序列化契约测试。
//!
//! 所有输入均为本地 JSON，不执行工具、不访问模型、不启动 HTTP 服务。

#![cfg(feature = "server")]

use qllama::server::tools::{
    openai_tool_calls, parse_choice, parse_generated_calls, parse_tools, stream_tool_deltas,
    tool_response_schema, validate_tools, ToolCall, ToolChoice, ToolDefinition,
};
use serde_json::{json, Value};

/// 构造一个参数为空对象的工具定义。
fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: Some(format!("{} tool", name)),
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

/// 把若干定义包装为 OpenAI `tools` 请求字段。
fn tools_value(names: &[&str]) -> Value {
    Value::Array(
        names
            .iter()
            .map(|name| {
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": format!("{} tool", name),
                        "parameters": { "type": "object", "properties": {} },
                    },
                })
            })
            .collect(),
    )
}

/// 构造两个参数不同的调用，供并行、流式等用例复用。
fn sample_calls() -> Vec<ToolCall> {
    vec![
        ToolCall {
            id: "call_0".to_string(),
            name: "lookup".to_string(),
            arguments: json!({ "city": "北京" }),
        },
        ToolCall {
            id: "call_1".to_string(),
            name: "sum".to_string(),
            arguments: json!({ "a": 1, "b": 2 }),
        },
    ]
}

/// 未提供 tools 或 tool_choice 时使用空列表和 Auto，并校验所有 choice 分支。
#[test]
fn parse_optional_inputs_and_all_choice_branches() {
    assert!(parse_tools(None).expect("缺省 tools").is_empty());
    assert_eq!(parse_choice(None).expect("缺省 choice"), ToolChoice::Auto);
    assert_eq!(
        parse_choice(Some(&json!("none"))).unwrap(),
        ToolChoice::None
    );
    assert_eq!(
        parse_choice(Some(&json!("auto"))).unwrap(),
        ToolChoice::Auto
    );
    assert_eq!(
        parse_choice(Some(&json!("required"))).unwrap(),
        ToolChoice::Required
    );
    assert_eq!(
        parse_choice(Some(
            &json!({ "type": "function", "function": { "name": "lookup" } })
        ))
        .unwrap(),
        ToolChoice::Function("lookup".to_string())
    );
    assert!(parse_choice(Some(&json!("sometimes"))).is_err());
    assert!(parse_choice(Some(&json!({ "type": "other" }))).is_err());
    assert!(parse_choice(Some(&json!(7))).is_err());
}

/// 解析单个工具的名称、描述和参数结构。
#[test]
fn parse_single_tool_definition() {
    let value = tools_value(&["lookup"]);
    let tools = parse_tools(Some(&value)).expect("单个工具应解析成功");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "lookup");
    assert_eq!(tools[0].description.as_deref(), Some("lookup tool"));
    assert_eq!(
        tools[0].parameters,
        json!({ "type": "object", "properties": {} })
    );
    validate_tools(&tools, &ToolChoice::Auto, false).expect("合法定义应通过校验");
}

/// 解析多个工具，并允许省略 description 与 parameters。
#[test]
fn parse_parallel_tool_definitions_with_optional_fields() {
    let value = json!([
        { "type": "function", "function": { "name": "a" } },
        { "type": "function", "function": { "name": "b", "description": null, "parameters": {} } },
    ]);
    let tools = parse_tools(Some(&value)).expect("多个工具应解析成功");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].description, None);
    assert_eq!(tools[0].parameters, json!({}));
    assert_eq!(tools[1].parameters, json!({}));
    validate_tools(&tools, &ToolChoice::Auto, false).expect("parallel=false 不限制工具定义数量");
    validate_tools(&tools, &ToolChoice::Auto, true).expect("parallel=true 也应通过");
}

/// 非 function 类型、缺名称、参数非对象、空 tools 非数组均被拒绝。
#[test]
fn parse_tools_rejects_invalid_shapes() {
    for value in [
        json!({ "type": "function" }),
        json!([{ "type": "retrieval", "function": { "name": "a" } }]),
        json!([{ "type": "function", "function": {} }]),
        json!([{ "type": "function", "function": { "name": "  " } }]),
        json!([{ "type": "function", "function": { "name": "a", "parameters": [] } }]),
        json!([{ "type": "function", "function": { "name": "a", "description": 1 } }]),
        json!(null),
    ] {
        assert!(parse_tools(Some(&value)).is_err(), "应拒绝 {}", value);
    }
}

/// 重复名称与非法的 schema 组合关键字（含数组内嵌套）均被拒绝。
#[test]
fn parse_tools_rejects_duplicates_and_illegal_schemas() {
    let duplicate = json!([
        { "type": "function", "function": { "name": "same" } },
        { "type": "function", "function": { "name": "same" } },
    ]);
    let error = parse_tools(Some(&duplicate)).expect_err("重名应失败");
    assert!(error.contains("unique"), "实际错误：{}", error);

    for keyword in ["$ref", "oneOf", "anyOf", "allOf"] {
        let nested = json!({
            "type": "object",
            "properties": { "value": { keyword: [{}] } },
        });
        let value =
            json!([{ "type": "function", "function": { "name": "a", "parameters": nested } }]);
        assert!(parse_tools(Some(&value)).is_err(), "{} 应被拒绝", keyword);
    }
    let directly_constructed = vec![ToolDefinition {
        name: "a".to_string(),
        description: None,
        parameters: json!({ "allOf": [] }),
    }];
    assert!(validate_tools(&directly_constructed, &ToolChoice::Auto, false).is_err());
}

/// Required 与命名选择需要工具；名称必须匹配；None 不因携带工具而失败。
#[test]
fn validate_choice_against_tool_definitions() {
    let tools = vec![tool("lookup")];
    assert!(validate_tools(&[], &ToolChoice::Required, false).is_err());
    assert!(validate_tools(&[], &ToolChoice::Function("lookup".into()), false).is_err());
    assert!(validate_tools(&[], &ToolChoice::None, false).is_ok());
    assert!(validate_tools(&tools, &ToolChoice::None, false).is_ok());
    assert!(validate_tools(&tools, &ToolChoice::Function("other".into()), false).is_err());
    assert!(validate_tools(&tools, &ToolChoice::Function("lookup".into()), false).is_ok());
}

/// 单个调用包装形式生成确定性 call_0，参数结构保持对象。
#[test]
fn parse_generated_single_call_and_ids() {
    let tools = vec![tool("lookup")];
    let text = r#"{"tool_calls":[{"name":"lookup","arguments":{"city":"北京"}}]}"#;
    let calls = parse_generated_calls(text, &tools, &ToolChoice::Auto, false).expect("单调用");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_0");
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, json!({ "city": "北京" }));
}

/// 单个调用对象与空 tool_calls 数组也按契约解析。
#[test]
fn parse_generated_single_object_and_empty_array() {
    let tools = vec![tool("lookup")];
    let single = parse_generated_calls(
        r#"{"name":"lookup","arguments":{"q":1}}"#,
        &tools,
        &ToolChoice::Auto,
        false,
    )
    .expect("单调用对象");
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].id, "call_0");
    let empty = parse_generated_calls(r#"{"tool_calls":[]}"#, &tools, &ToolChoice::Auto, false)
        .expect("空调用数组");
    assert!(empty.is_empty());
}

/// 并行开启时允许两个调用，ID 按索引生成；关闭时拒绝多个调用。
#[test]
fn parse_generated_parallel_calls_and_parallel_disabled() {
    let tools = vec![tool("lookup"), tool("sum")];
    let text = r#"{"tool_calls":[{"name":"lookup","arguments":{"city":"北京"}},{"name":"sum","arguments":{"a":1}}]}"#;
    let calls = parse_generated_calls(text, &tools, &ToolChoice::Auto, true).expect("并行调用");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_0");
    assert_eq!(calls[1].id, "call_1");
    let error = parse_generated_calls(text, &tools, &ToolChoice::Auto, false)
        .expect_err("parallel=false 应拒绝");
    assert!(error.contains("parallel"), "实际错误：{}", error);
}

/// 所有 choice 分支在解析生成结果时都得到执行，Auto 允许零调用。
#[test]
fn parse_generated_calls_applies_each_choice() {
    let tools = vec![tool("lookup"), tool("sum")];
    let one = r#"{"tool_calls":[{"name":"lookup","arguments":{}}]}"#;
    let none = r#"{"tool_calls":[]}"#;
    assert!(parse_generated_calls(one, &tools, &ToolChoice::None, false).is_err());
    assert!(parse_generated_calls(none, &tools, &ToolChoice::None, false).is_ok());
    assert!(parse_generated_calls(none, &tools, &ToolChoice::Auto, false).is_ok());
    assert!(parse_generated_calls(one, &tools, &ToolChoice::Required, false).is_ok());
    assert!(parse_generated_calls(none, &tools, &ToolChoice::Required, false).is_err());
    assert!(
        parse_generated_calls(one, &tools, &ToolChoice::Function("lookup".into()), false).is_ok()
    );
    let mismatch = parse_generated_calls(
        r#"{"tool_calls":[{"name":"sum","arguments":{}}]}"#,
        &tools,
        &ToolChoice::Function("lookup".into()),
        false,
    )
    .expect_err("命名选择不匹配");
    assert!(mismatch.contains("allowed"), "实际错误：{}", mismatch);
    assert!(
        parse_generated_calls(none, &tools, &ToolChoice::Function("lookup".into()), false).is_err()
    );
}

/// 未知工具、非对象参数、非对象调用与畸形 JSON 均被拒绝，且不回显输入。
#[test]
fn parse_generated_calls_rejects_bad_output_without_echo() {
    let tools = vec![tool("lookup")];
    let secret = "secret argument value";
    let cases = [
        format!(
            r#"{{"tool_calls":[{{"name":"unknown","arguments":{{"q":"{}"}}}}]}}"#,
            secret
        ),
        format!(
            r#"{{"tool_calls":[{{"name":"lookup","arguments":"{}"}}]}}"#,
            secret
        ),
        r#"{"tool_calls":[42]}"#.to_string(),
        r#"{"unexpected":true}"#.to_string(),
        format!(
            r#"{{"tool_calls":[{{"name":"lookup","arguments":{{"q":"{}"}},"extra":1}}]}}"#,
            secret
        ),
        "not json".to_string(),
        "[]".to_string(),
    ];
    for text in cases {
        let error = parse_generated_calls(&text, &tools, &ToolChoice::Auto, false)
            .expect_err("畸形输出应失败");
        assert!(!error.contains(secret), "错误不应回显参数：{}", error);
        assert!(!error.contains("secret"), "错误不应回显提示内容：{}", error);
    }
}

/// OpenAI 非流式 tool_calls 的 arguments 必须是 JSON 字符串，且保留 id/name/type。
#[test]
fn openai_tool_calls_serialize_arguments_as_string() {
    let calls = sample_calls();
    let values = openai_tool_calls(&calls);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0]["id"], json!("call_0"));
    assert_eq!(values[0]["type"], json!("function"));
    assert_eq!(values[0]["function"]["name"], json!("lookup"));
    let arguments = values[0]["function"]["arguments"]
        .as_str()
        .expect("arguments 必须是字符串");
    assert_eq!(
        serde_json::from_str::<Value>(arguments).unwrap(),
        calls[0].arguments
    );
    assert_eq!(values[1]["id"], json!("call_1"));
    assert_eq!(values[1]["function"]["name"], json!("sum"));
    assert!(openai_tool_calls(&[]).is_empty());
}

/// 流式增量按 Unicode 字符切分，首片带名称，拼接后可还原完整参数 JSON。
#[test]
fn stream_tool_deltas_split_unicode_arguments_and_name_only_on_first_delta() {
    let calls = sample_calls();
    let deltas = stream_tool_deltas(&calls, 3);
    let first_arguments = calls[0].arguments.to_string();
    let first_count = first_arguments.chars().count().div_ceil(3);
    let second_count = calls[1].arguments.to_string().chars().count().div_ceil(3);
    assert_eq!(deltas.len(), first_count + second_count);
    assert_eq!(deltas[0]["index"], json!(0));
    assert_eq!(deltas[0]["id"], json!("call_0"));
    assert_eq!(deltas[0]["type"], json!("function"));
    assert_eq!(deltas[0]["function"]["name"], json!("lookup"));
    assert!(deltas[0]["function"]["arguments"].is_string());
    assert!(
        deltas[1]["function"].get("name").is_none(),
        "第二片不应重复名称"
    );
    let mut assembled = String::new();
    for delta in &deltas[..first_count] {
        assembled.push_str(delta["function"]["arguments"].as_str().unwrap());
    }
    assert_eq!(assembled, first_arguments);
    assert_eq!(deltas[first_count]["index"], json!(1));
    assert_eq!(deltas[first_count]["id"], json!("call_1"));
    assert_eq!(deltas[first_count]["function"]["name"], json!("sum"));
    assert!(stream_tool_deltas(&[], 1).is_empty());
}

/// chunk_size 为 0 时按文档约定 panic，调用方必须在边界校验。
#[test]
#[should_panic(expected = "chunk_size must be greater than zero")]
fn stream_tool_deltas_rejects_zero_chunk_size() {
    let _ = stream_tool_deltas(&sample_calls(), 0);
}

/// None 的响应 schema 只允许空对象，且不含 $schema 等约束解码不支持的关键字。
#[test]
fn response_schema_for_none_is_empty_object() {
    let tools = vec![tool("lookup")];
    let schema = tool_response_schema(&tools, &ToolChoice::None, false).expect("None schema");
    assert_eq!(
        schema,
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })
    );
    assert!(schema.get("$schema").is_none());
    assert!(!schema.to_string().contains("oneOf"));
}

/// Auto/并行/Required/命名选择的 schema 枚举、required 和数量边界符合契约。
#[test]
fn response_schema_limits_names_and_call_counts() {
    let tools = vec![tool("lookup"), tool("sum")];
    let auto = tool_response_schema(&tools, &ToolChoice::Auto, false).unwrap();
    let calls = &auto["properties"]["tool_calls"];
    assert_eq!(calls["minItems"], json!(0));
    assert_eq!(calls["maxItems"], json!(1));
    assert_eq!(
        calls["items"]["properties"]["name"]["enum"],
        json!(["lookup", "sum"])
    );
    assert_eq!(calls["items"]["required"], json!(["name", "arguments"]));
    assert_eq!(calls["items"]["additionalProperties"], json!(false));
    assert_eq!(auto["required"], json!(["tool_calls"]));
    assert_eq!(auto["additionalProperties"], json!(false));

    let parallel = tool_response_schema(&tools, &ToolChoice::Auto, true).unwrap();
    assert_eq!(parallel["properties"]["tool_calls"]["maxItems"], json!(2));
    let required = tool_response_schema(&tools, &ToolChoice::Required, false).unwrap();
    assert_eq!(required["properties"]["tool_calls"]["minItems"], json!(1));
    let named = tool_response_schema(&tools, &ToolChoice::Function("sum".into()), false).unwrap();
    assert_eq!(named["properties"]["tool_calls"]["minItems"], json!(1));
    assert_eq!(named["properties"]["tool_calls"]["maxItems"], json!(1));
    assert_eq!(
        named["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"],
        json!(["sum"])
    );
}

/// 非法工具定义和命名选择不匹配时，响应 schema 不生成。
#[test]
fn response_schema_propagates_validation_errors() {
    let duplicate = vec![tool("same"), tool("same")];
    assert!(tool_response_schema(&duplicate, &ToolChoice::Auto, false).is_err());
    assert!(tool_response_schema(&[], &ToolChoice::Required, false).is_err());
    assert!(tool_response_schema(
        &[tool("lookup")],
        &ToolChoice::Function("sum".into()),
        false
    )
    .is_err());
}
