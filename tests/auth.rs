#![cfg(feature = "server")]
//! `auth` 模块的行为测试。
//!
//! 每条用例只断言可从公开 API 观察到的行为：错误串、上下文字段、
//! CORS/traceparent 判定与密钥文件解析结果。密钥值以固定字面量给出，
//! 断言其不出现在任何错误信息中。

use qllama::server::auth::{
    authenticate, constant_time_eq, cors_allowed, hash_key, oidc_status, parse_key_file,
    parse_traceparent, ApiKey, AuthConfig, Role,
};

/// 构造 `(name, value)` 头列表。
fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 构造一个带 Bearer 头的请求头列表。
fn bearer(token: &str) -> Vec<(String, String)> {
    headers(&[("Authorization", &format!("Bearer {token}"))])
}

/// 构造带 Bearer + traceparent 头的请求头列表。
fn bearer_with_traceparent(token: &str, traceparent: &str) -> Vec<(String, String)> {
    headers(&[
        ("Authorization", &format!("Bearer {token}")),
        ("traceparent", traceparent),
    ])
}

/// 构造只含一枚有效密钥的配置。
fn config_with(id: &str, secret: &str, role: Role) -> AuthConfig {
    AuthConfig {
        keys: vec![ApiKey {
            id: id.to_string(),
            secret: secret.to_string(),
            role,
            tenant: "default".to_string(),
            revoked: false,
            expires_at: None,
        }],
        ..AuthConfig::default()
    }
}

/// 32 位合法 traceparent 头值（版本 00）。
const VALID_TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

#[test]
fn authenticate_rejects_missing_header() {
    let config = config_with("k1", "s1", Role::Admin);
    let err = authenticate(&config, &[], Role::Inference, 100).unwrap_err();
    assert_eq!(err, "missing_authorization");
}

#[test]
fn authenticate_rejects_non_bearer_and_wrong_key() {
    let config = config_with("k1", "s1", Role::Admin);
    let malformed = authenticate(
        &config,
        headers(&[("Authorization", "Basic abc")]).as_slice(),
        Role::Inference,
        100,
    )
    .unwrap_err();
    assert_eq!(malformed, "malformed_authorization");

    let wrong = authenticate(
        &config,
        bearer("not-the-secret").as_slice(),
        Role::Inference,
        100,
    )
    .unwrap_err();
    assert_eq!(wrong, "invalid_credentials");
}

#[test]
fn authenticate_rejects_when_no_keys_configured() {
    let config = AuthConfig::default();
    let err =
        authenticate(&config, bearer("anything").as_slice(), Role::Inference, 100).unwrap_err();
    assert_eq!(err, "invalid_credentials");
}

#[test]
fn authenticate_rejects_revoked_key() {
    let mut config = config_with("k1", "s1", Role::Admin);
    config.keys[0].revoked = true;
    let err = authenticate(&config, bearer("s1").as_slice(), Role::Inference, 100).unwrap_err();
    assert_eq!(err, "revoked_key");
}

#[test]
fn authenticate_rejects_expired_key() {
    let mut config = config_with("k1", "s1", Role::Admin);
    config.keys[0].expires_at = Some(1000);
    // now == expires_at 即过期（含边界）。
    let err = authenticate(&config, bearer("s1").as_slice(), Role::Inference, 1000).unwrap_err();
    assert_eq!(err, "expired_key");
}

#[test]
fn authenticate_rejects_role_escalation() {
    let config = config_with("k1", "s1", Role::Read);
    // Read 密钥请求 Inference 权限：角色不足。
    let err = authenticate(&config, bearer("s1").as_slice(), Role::Inference, 100).unwrap_err();
    assert_eq!(err, "insufficient_role");
    // Read 密钥请求 Read 权限：放行。
    let ok = authenticate(&config, bearer("s1").as_slice(), Role::Read, 100).unwrap();
    assert_eq!(ok.role, Role::Read);
    assert_eq!(ok.key_id, "k1");
    assert_eq!(ok.tenant, "default");
    assert_eq!(ok.key_hash, hash_key("s1"));
}

#[test]
fn authenticate_header_name_is_case_insensitive() {
    let config = config_with("k1", "s1", Role::Admin);
    let ok = authenticate(
        &config,
        headers(&[("AUTHORIZATION", "Bearer s1")]).as_slice(),
        Role::Read,
        100,
    );
    assert!(ok.is_ok());
}

#[test]
fn authenticate_prefers_valid_traceparent_and_generates_fallback() {
    let config = config_with("k1", "s1", Role::Admin);
    let with_tp = bearer_with_traceparent("s1", VALID_TRACEPARENT);
    let ctx = authenticate(&config, with_tp.as_slice(), Role::Read, 100).unwrap();
    assert_eq!(ctx.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
    assert!(ctx.request_id.starts_with("req-"));

    // 无合法 traceparent 时生成 32 位小写十六进制 trace id。
    let ctx = authenticate(&config, bearer("s1").as_slice(), Role::Read, 100).unwrap();
    assert_eq!(ctx.trace_id.len(), 32);
    assert!(ctx
        .trace_id
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    assert_ne!(ctx.trace_id, "0".repeat(32));
}

#[test]
fn parse_traceparent_accepts_valid_and_rejects_invalid() {
    let valid = parse_traceparent(&headers(&[("traceparent", VALID_TRACEPARENT)]));
    assert_eq!(valid.as_deref(), Some("4bf92f3577b34da6a3ce929d0e0e4736"));

    // 头名大小写不敏感。
    let valid = parse_traceparent(&headers(&[("TRACEPARENT", VALID_TRACEPARENT)]));
    assert!(valid.is_some());

    let cases: [Vec<(String, String)>; 5] = [
        headers(&[(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        )]), // 段数不足
        headers(&[(
            "traceparent",
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )]), // 版本非 00
        headers(&[(
            "traceparent",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        )]), // 大写十六进制
        headers(&[("traceparent", "00-short-00f067aa0ba902b7-01")]), // trace id 长度错误
        headers(&[("other", VALID_TRACEPARENT)]),                    // 头缺失
    ];
    for case in cases {
        assert!(parse_traceparent(&case).is_none());
    }
}

#[test]
fn cors_exact_match_star_and_empty_rules() {
    let mut config = AuthConfig::default();
    // 空列表拒绝所有。
    assert!(!cors_allowed(&config, "https://a.example"));
    config.cors_origins = vec!["https://a.example".to_string()];
    assert!(cors_allowed(&config, "https://a.example"));
    assert!(!cors_allowed(&config, "https://b.example"));
    // * 放行所有。
    config.cors_origins = vec!["*".to_string()];
    assert!(cors_allowed(&config, "https://anything.example"));
}

#[test]
fn hash_key_is_stable_and_matches_fnv1a64() {
    // 期望值来自独立实现的 FNV-1a 64 位 oracle（非被测代码推得）。
    assert_eq!(hash_key("abc"), "e71fa2190541574b");
    assert_eq!(hash_key(""), "cbf29ce484222325");
    assert_eq!(hash_key("abc"), hash_key("abc"));
    assert_ne!(hash_key("abc"), hash_key("abd"));
}

#[test]
fn constant_time_eq_compares_bytes() {
    assert!(constant_time_eq(b"secret", b"secret"));
    assert!(!constant_time_eq(b"secret", b"secreT"));
    assert!(!constant_time_eq(b"secret", b"secr"));
    assert!(constant_time_eq(b"", b""));
}

#[test]
fn parse_key_file_supports_defaults_comments_and_trim() {
    let text = "# 注释行\n\
                \n\
                k1:sec1\n\
                k2 : sec2 : admin : team-b\n\
                k3:sec3:read\n";
    let keys = parse_key_file(text).unwrap();
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0].id, "k1");
    assert_eq!(keys[0].secret, "sec1");
    assert_eq!(keys[0].role, Role::Inference);
    assert_eq!(keys[0].tenant, "default");
    assert_eq!(keys[1].role, Role::Admin);
    assert_eq!(keys[1].tenant, "team-b");
    assert_eq!(keys[2].role, Role::Read);
}

#[test]
fn parse_key_file_rejects_bad_lines_without_leaking_secret() {
    let duplicate = parse_key_file("k1:sec1\nk1:other-secret\n").unwrap_err();
    assert!(duplicate.contains("duplicate"));
    assert!(!duplicate.contains("other-secret"));

    let empty_secret = parse_key_file("k1:\n").unwrap_err();
    assert!(empty_secret.contains("secret must not be empty"));

    let empty_id = parse_key_file(":sec1\n").unwrap_err();
    assert!(empty_id.contains("id must not be empty"));

    let bad_role = parse_key_file("k1:sec1:superuser\n").unwrap_err();
    assert!(bad_role.contains("unknown role"));

    let too_many = parse_key_file("k1:sec1:admin:t1:extra\n").unwrap_err();
    assert!(too_many.contains("expected id:secret"));
}

#[test]
fn oidc_status_reports_disabled_or_not_implemented() {
    assert_eq!(oidc_status(&AuthConfig::default()), "disabled");
    let config = AuthConfig {
        oidc_enabled: true,
        ..AuthConfig::default()
    };
    assert_eq!(oidc_status(&config), "not_implemented");
}
