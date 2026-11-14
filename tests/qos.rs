#![cfg(feature = "server")]
//! `qos` 模块的行为测试。
//!
//! 用单线程时间推进模拟窗口；每条用例只断言公开 API 的返回与
//! `snapshot` 可见状态，错误串与并发归还是重点。

use qllama::server::qos::{QosManager, Quota};
use std::collections::HashMap;

/// 构造只含默认配额的管理器。
fn manager() -> QosManager {
    QosManager::new(Quota::default(), HashMap::new())
}

#[test]
fn acquire_and_release_enforce_in_flight_limit() {
    let mgr = manager();
    let p1 = mgr.acquire("k1", 10, 0).unwrap();
    let p2 = mgr.acquire("k1", 10, 1).unwrap();
    let p3 = mgr.acquire("k1", 10, 2).unwrap();
    let p4 = mgr.acquire("k1", 10, 3).unwrap();
    let err = mgr.acquire("k1", 10, 4).unwrap_err();
    assert_eq!(err, "in_flight");
    assert_eq!(mgr.snapshot("k1")["in_flight"], 4);

    p1.release(&mgr, 200);
    assert_eq!(mgr.snapshot("k1")["in_flight"], 3);
    // 重复 release 不得二次递减。
    p1.release(&mgr, 500);
    assert_eq!(mgr.snapshot("k1")["in_flight"], 3);
    assert!(mgr.acquire("k1", 10, 5).is_ok());

    p2.release(&mgr, 200);
    p3.release(&mgr, 200);
    p4.release(&mgr, 200);
}

#[test]
fn request_rate_limit_is_per_window() {
    let mgr = QosManager::new(
        Quota {
            max_in_flight: 100,
            requests_per_minute: 2,
            tokens_per_minute: 100_000,
        },
        HashMap::new(),
    );
    assert!(mgr.acquire("k1", 1, 0).is_ok());
    assert!(mgr.acquire("k1", 1, 1).is_ok());
    let err = mgr.acquire("k1", 1, 2).unwrap_err();
    assert_eq!(err, "request_rate");
    // 窗口边界：`ts + 60_000 <= now` 时旧记录被裁剪。
    assert!(mgr.acquire("k1", 1, 60_000).is_ok());
    assert!(mgr.acquire("k1", 1, 60_001).is_ok());
    assert_eq!(mgr.acquire("k1", 1, 60_002).unwrap_err(), "request_rate");
}

#[test]
fn token_quota_limit_counts_estimates() {
    let mgr = QosManager::new(
        Quota {
            max_in_flight: 100,
            requests_per_minute: 100,
            tokens_per_minute: 100,
        },
        HashMap::new(),
    );
    assert!(mgr.acquire("k1", 60, 0).is_ok());
    assert!(mgr.acquire("k1", 40, 1).is_ok());
    // 恰好等于上限允许，超 1 即拒绝。
    assert!(mgr.acquire("k1", 0, 2).is_ok());
    assert_eq!(mgr.acquire("k1", 1, 3).unwrap_err(), "token_quota");
    // 60 秒窗口后旧 token 记录被裁剪，额度恢复。
    assert!(mgr.acquire("k1", 100, 60_001).is_ok());
}

#[test]
fn per_key_override_takes_precedence() {
    let per_key = HashMap::from([(
        "vip".to_string(),
        Quota {
            max_in_flight: 1,
            requests_per_minute: 1,
            tokens_per_minute: 10,
        },
    )]);
    let mgr = QosManager::new(Quota::default(), per_key);
    let p = mgr.acquire("vip", 1, 0).unwrap();
    assert_eq!(mgr.acquire("vip", 1, 1).unwrap_err(), "in_flight");
    p.release(&mgr, 200);
    assert_eq!(mgr.acquire("vip", 1, 2).unwrap_err(), "request_rate");

    // 普通 key 仍使用默认配额（4 并发）。
    for _ in 0..4 {
        assert!(mgr.acquire("normal", 1, 3).is_ok());
    }
    assert_eq!(mgr.acquire("normal", 1, 4).unwrap_err(), "in_flight");
}

#[test]
fn snapshot_has_documented_shape_and_isolated_keys() {
    let mgr = QosManager::new(
        Quota {
            max_in_flight: 7,
            requests_per_minute: 8,
            tokens_per_minute: 256,
        },
        HashMap::new(),
    );
    let empty = mgr.snapshot("missing");
    assert_eq!(empty["key_id"], "missing");
    assert_eq!(empty["in_flight"], 0);
    assert_eq!(empty["window_requests"], 0);
    assert_eq!(empty["window_tokens"], 0);
    assert_eq!(empty["quota"]["max_in_flight"], 7);
    assert_eq!(empty["quota"]["requests_per_minute"], 8);
    assert_eq!(empty["quota"]["tokens_per_minute"], 256);

    mgr.acquire("k1", 123, 0).unwrap();
    mgr.acquire("k2", 5, 0).unwrap();
    let s1 = mgr.snapshot("k1");
    assert_eq!(s1["in_flight"], 1);
    assert_eq!(s1["window_requests"], 1);
    assert_eq!(s1["window_tokens"], 123);
    // 快照不得包含任何业务体字段。
    let serialized = s1.to_string();
    assert!(!serialized.contains("prompt"));
    assert_eq!(mgr.snapshot("k2")["window_tokens"], 5);
}

#[test]
fn time_rollback_resets_window_without_panic() {
    let mgr = QosManager::new(
        Quota {
            max_in_flight: 100,
            requests_per_minute: 2,
            tokens_per_minute: 100,
        },
        HashMap::new(),
    );
    assert!(mgr.acquire("k1", 50, 10_000).is_ok());
    assert!(mgr.acquire("k1", 50, 10_001).is_ok());
    assert_eq!(mgr.acquire("k1", 1, 10_002).unwrap_err(), "request_rate");
    // 时钟回拨：窗口清空重计，不 panic、不产生负耗时。
    assert!(mgr.acquire("k1", 50, 5_000).is_ok());
    assert_eq!(mgr.snapshot("k1")["window_requests"], 1);
    assert_eq!(mgr.snapshot("k1")["window_tokens"], 50);
    assert!(mgr.acquire("k1", 50, 5_001).is_ok());
}
