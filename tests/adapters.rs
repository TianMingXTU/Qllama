#![cfg(feature = "server")]
//! `adapters` 模块的行为测试。
//!
//! 校验 5 个占位接口全部如实返回 501 语义错误、不产生任何向量或
//! 伪后端标识；V1 不得从 decoder logits 派生 embedding/rerank。

use qllama::server::adapters::{
    AcceleratorAdapter, EmbeddingAdapter, MultimodalAdapter, NotImplementedAdapter,
    RegistryAdapter, RerankAdapter,
};

#[test]
fn embedding_adapter_never_produces_vectors() {
    let adapter = NotImplementedAdapter;
    let texts = vec!["text-a".to_string(), "text-b".to_string()];
    let err = adapter.embed(&texts).unwrap_err();
    assert!(err.contains("not_implemented"), "err={err}");
    assert!(err.contains("501"), "err={err}");
    assert!(err.contains("embedding"), "err={err}");
    // 空输入同样不伪装成功。
    assert!(adapter.embed(&[]).unwrap_err().contains("501"));
}

#[test]
fn rerank_adapter_never_produces_scores() {
    let adapter = NotImplementedAdapter;
    let docs = vec!["doc-a".to_string(), "doc-b".to_string()];
    let err = adapter.rerank("query", &docs).unwrap_err();
    assert!(err.contains("not_implemented"), "err={err}");
    assert!(err.contains("501"), "err={err}");
    assert!(err.contains("rerank"), "err={err}");
    assert!(adapter.rerank("q", &[]).unwrap_err().contains("501"));
}

#[test]
fn multimodal_adapter_reports_no_modality() {
    let modality = NotImplementedAdapter.modality();
    assert_eq!(modality, "none");
    assert!(!modality.is_empty());
}

#[test]
fn accelerator_adapter_reports_no_backend() {
    let backend = NotImplementedAdapter.backend();
    assert_eq!(backend, "none");
    // 不得冒充 GPU/专用后端。
    for fake in ["cuda", "metal", "rocm", "vulkan"] {
        assert_ne!(backend, fake);
    }
}

#[test]
fn registry_adapter_never_fakes_load_success() {
    let err = NotImplementedAdapter.load("model-1").unwrap_err();
    assert!(err.contains("not_implemented"), "err={err}");
    assert!(err.contains("501"), "err={err}");
    assert!(NotImplementedAdapter.load("").unwrap_err().contains("501"));
}
