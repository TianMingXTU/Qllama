//! Ollama 原生 API 响应形状的契约测试。
//!
//! 仅验证纯函数返回的字段结构和调用方提供的用量，不启动 HTTP 服务。

#![cfg(feature = "server")]

use qllama::server::ollama::{
    chat_chunk, generate_chunk, ps_document, show_document, tags_document, unsupported_error,
    version_document, OllamaTagsEntry, OllamaUsage, UNSUPPORTED_ENDPOINTS,
};
use serde_json::{json, Value};

/// 构造一条名称与数值互不相同的模型描述，防止断言被默认值蒙混过关。
fn entry(name: &str) -> OllamaTagsEntry {
    OllamaTagsEntry {
        name: name.to_string(),
        digest: format!("sha256:{}", name),
        size: name.len() as u64 * 101,
        format: "gguf",
        family: "llama",
        parameter_size: "3B".to_string(),
        quantization_level: "Q4_K_M".to_string(),
    }
}

/// 构造每个字段均可区分的用量数据。
fn usage() -> OllamaUsage {
    OllamaUsage {
        prompt_eval_count: 7,
        eval_count: 11,
        total_duration_ns: 1_200_003,
        load_duration_ns: 400_004,
        prompt_eval_duration_ns: 300_005,
        eval_duration_ns: 500_006,
    }
}

/// version 响应只携带调用方给定的版本号。
#[test]
fn version_document_uses_supplied_version() {
    assert_eq!(version_document("1.2.3"), json!({ "version": "1.2.3" }));
}

/// tags 响应的每个模型都含 name/model/digest/size/details 及四个细节字段。
#[test]
fn tags_document_contains_all_static_fields() {
    let first = entry("first");
    let second = entry("second");
    let document = tags_document(&[first.clone(), second.clone()]);
    let models = document["models"].as_array().expect("models 应为数组");
    assert_eq!(models.len(), 2);
    for (index, (model, source)) in models.iter().zip([&first, &second]).enumerate() {
        assert_eq!(model["name"], json!(source.name), "第 {} 项 name", index);
        assert_eq!(model["model"], json!(source.name), "第 {} 项 model", index);
        assert_eq!(
            model["digest"],
            json!(source.digest),
            "第 {} 项 digest",
            index
        );
        assert_eq!(model["size"], json!(source.size), "第 {} 项 size", index);
        assert_eq!(
            model["details"],
            json!({
                "format": source.format,
                "family": source.family,
                "parameter_size": source.parameter_size,
                "quantization_level": source.quantization_level,
            })
        );
        assert!(model.get("modified_at").is_none(), "不伪造修改时间");
    }
}

/// ps 响应与 tags 共用模型字段，且不伪造过期时间或显存占用。
#[test]
fn ps_document_omits_invented_runtime_fields() {
    let models = vec![entry("running")];
    assert_eq!(ps_document(&models), tags_document(&models));
    let document = ps_document(&models);
    let model = &document["models"][0];
    assert!(model.get("expires_at").is_none());
    assert!(model.get("size_vram").is_none());
}

/// show 响应包含模板、上下文长度和细节，不携带提示词或伪造的 Modelfile。
#[test]
fn show_document_contains_template_and_context_length() {
    let document = show_document(&entry("shown"), "{{ .Prompt }}", 4096);
    assert_eq!(document["modelfile"], json!(""));
    assert_eq!(document["parameters"], json!(""));
    assert_eq!(document["template"], json!("{{ .Prompt }}"));
    assert_eq!(document["details"]["parameter_size"], json!("3B"));
    assert_eq!(
        document["model_info"]["general.architecture"],
        json!("llama")
    );
    assert_eq!(document["model_info"]["llama.context_length"], json!(4096));
    let serialized = document.to_string();
    assert!(
        !serialized.contains("prompt\":{"),
        "show 不应包含 prompt 对象"
    );
}

/// generate chunk 的所有用量与计时逐项来自调用方，且保留正文与结束原因。
#[test]
fn generate_chunk_preserves_text_done_reason_and_all_usage() {
    let document = generate_chunk("tiny", "2026-11-14T12:00:00Z", "hi", false, None, &usage());
    assert_eq!(document["model"], json!("tiny"));
    assert_eq!(document["created_at"], json!("2026-11-14T12:00:00Z"));
    assert_eq!(document["response"], json!("hi"));
    assert_eq!(document["done"], json!(false));
    assert!(document["done_reason"].is_null());
    assert_usage(&document);
}

/// chat chunk 使用 message 形状，并逐项携带调用方提供的用量。
#[test]
fn chat_chunk_uses_message_shape_and_all_usage() {
    let document = chat_chunk(
        "tiny",
        "2026-11-14T12:00:01Z",
        "assistant",
        "hello",
        true,
        Some("stop"),
        &usage(),
    );
    assert_eq!(
        document["message"],
        json!({ "role": "assistant", "content": "hello" })
    );
    assert_eq!(document["done"], json!(true));
    assert_eq!(document["done_reason"], json!("stop"));
    assert!(document.get("response").is_none());
    assert_usage(&document);
}

/// 不同 done_reason 原样传递，不改变用量字段。
#[test]
fn chunk_done_reason_accepts_protocol_values() {
    for reason in ["stop", "length", "error", "unload"] {
        let document = generate_chunk("m", "t", "", true, Some(reason), &usage());
        assert_eq!(document["done_reason"], json!(reason));
        assert_eq!(document["eval_count"], json!(11));
    }
}

/// unsupported error 的形状固定为 OpenAI 风格，qllama 信息位于顶层。
#[test]
fn unsupported_error_has_openai_shape_and_top_level_qllama() {
    let document = unsupported_error("/api/embed");
    assert_eq!(document["error"]["type"], json!("not_implemented_error"));
    assert_eq!(document["error"]["code"], json!(501));
    assert_eq!(document["qllama"], json!({ "unsupported": ["/api/embed"] }));
    let message = document["error"]["message"]
        .as_str()
        .expect("message 应为字符串");
    assert!(!message.is_empty());
    assert!(
        !message.contains("/api/embed"),
        "能力名只在 qllama 字段中传递"
    );
    assert_eq!(document.as_object().expect("错误应为对象").len(), 2);
}

/// 错误信息不含提示词或请求正文；qllama 数组只含给定能力名。
#[test]
fn unsupported_error_does_not_echo_request_content() {
    let simulated_request = "secret prompt text";
    let document = unsupported_error("/api/pull");
    let serialized = document.to_string();
    assert!(!serialized.contains(simulated_request));
    assert!(!serialized.contains("prompt"));
    assert!(!serialized.contains("response"));
    assert_eq!(document["qllama"]["unsupported"], json!(["/api/pull"]));
}

/// 不受支持端点列表与任务书规定的顺序、内容完全一致。
#[test]
fn unsupported_endpoints_match_contract_exactly() {
    let expected = [
        "/api/embed",
        "/api/pull",
        "/api/push",
        "/api/create",
        "/api/copy",
        "/api/delete",
        "/api/blobs/:digest",
        "/api/web_search",
        "/api/web_fetch",
    ];
    assert_eq!(UNSUPPORTED_ENDPOINTS, expected.as_slice());
}

/// 空模型列表生成空数组，而不是省略 models 字段。
#[test]
fn empty_model_lists_keep_models_array() {
    assert_eq!(tags_document(&[]), json!({ "models": [] }));
    assert_eq!(ps_document(&[]), json!({ "models": [] }));
}

/// 两类 chunk 的用量字段逐项断言，避免任何数字被硬编码为 0。
fn assert_usage(document: &Value) {
    assert_eq!(document["prompt_eval_count"], json!(7));
    assert_eq!(document["eval_count"], json!(11));
    assert_eq!(document["total_duration"], json!(1_200_003));
    assert_eq!(document["load_duration"], json!(400_004));
    assert_eq!(document["prompt_eval_duration"], json!(300_005));
    assert_eq!(document["eval_duration"], json!(500_006));
}
