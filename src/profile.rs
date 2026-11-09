//! 编译期开关控制的增量前向计时；事件按层归集，计数范围见 `Snapshot`。

use std::cell::RefCell;
use std::time::Instant;

/// 同一层内互不重叠的前向区段；模型级事件的 layer 为 `None`。
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum Stage {
    Embedding,
    RmsNorm,
    Query,
    Key,
    Value,
    Rope,
    Attention,
    Output,
    Gate,
    Up,
    Activation,
    Down,
    Residual,
    FinalNorm,
    LmHead,
}

impl Stage {
    pub const ALL: [Self; 15] = [
        Self::Embedding,
        Self::RmsNorm,
        Self::Query,
        Self::Key,
        Self::Value,
        Self::Rope,
        Self::Attention,
        Self::Output,
        Self::Gate,
        Self::Up,
        Self::Activation,
        Self::Down,
        Self::Residual,
        Self::FinalNorm,
        Self::LmHead,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Embedding => "token_embd",
            Self::RmsNorm => "rms_norm",
            Self::Query => "q_projection",
            Self::Key => "k_projection",
            Self::Value => "v_projection",
            Self::Rope => "rope",
            Self::Attention => "attention_kv_softmax",
            Self::Output => "o_projection",
            Self::Gate => "mlp_gate",
            Self::Up => "mlp_up",
            Self::Activation => "mlp_silu_mul",
            Self::Down => "mlp_down",
            Self::Residual => "residual",
            Self::FinalNorm => "final_norm",
            Self::LmHead => "lm_head",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub nanoseconds: [u128; 15],
    pub calls: [u64; 15],
    pub allocations: u64,
}

/// 只统计 Qllama 已埋点的张量缓冲、Arc、形状、步长及前向临时 Vec 分配。
///
/// 不包含 Rayon/系统分配器内部、错误格式化或基准报告分配；不是进程分配器总次数。
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub model: Timings,
    pub layers: Vec<Timings>,
}

#[derive(Default)]
struct Collector {
    enabled: bool,
    layer: Option<usize>,
    model: Timings,
    layers: Vec<Timings>,
}

thread_local! {
    static COLLECTOR: RefCell<Collector> = RefCell::new(Collector::default());
}

/// 在执行前向的同一线程上开始一次采集；初始化分配发生在计时之前。
pub fn start(num_layers: usize) {
    COLLECTOR.with(|collector| {
        *collector.borrow_mut() = Collector {
            enabled: true,
            layer: None,
            model: Timings::default(),
            layers: vec![Timings::default(); num_layers],
        };
    });
}

/// 结束采集并取出结果。调用方须保证所有计时区段已退出。
pub fn finish() -> Snapshot {
    COLLECTOR.with(|collector| {
        let mut collector = collector.borrow_mut();
        collector.enabled = false;
        Snapshot {
            model: std::mem::take(&mut collector.model),
            layers: std::mem::take(&mut collector.layers),
        }
    })
}

pub(crate) fn set_layer(layer: Option<usize>) {
    COLLECTOR.with(|collector| collector.borrow_mut().layer = layer);
}

pub(crate) fn allocations(count: usize) {
    COLLECTOR.with(|collector| {
        let mut collector = collector.borrow_mut();
        if collector.enabled {
            let layer = collector.layer;
            let timings = match layer {
                Some(index) => &mut collector.layers[index],
                None => &mut collector.model,
            };
            timings.allocations += count as u64;
        }
    });
}

pub(crate) struct Span {
    started: Option<Instant>,
    stage: Stage,
    layer: Option<usize>,
}

impl Span {
    pub(crate) fn new(stage: Stage) -> Self {
        let (enabled, layer) = COLLECTOR.with(|collector| {
            let collector = collector.borrow();
            (collector.enabled, collector.layer)
        });
        Self {
            started: enabled.then(Instant::now),
            stage,
            layer,
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            let elapsed = started.elapsed().as_nanos();
            COLLECTOR.with(|collector| {
                let mut collector = collector.borrow_mut();
                let timings = match self.layer {
                    Some(index) => &mut collector.layers[index],
                    None => &mut collector.model,
                };
                timings.nanoseconds[self.stage as usize] += elapsed;
                timings.calls[self.stage as usize] += 1;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn counts_tensor_storage_metadata_and_copy_on_write() {
        start(1);
        set_layer(Some(0));
        let span = Span::new(Stage::Attention);
        let mut tensor = Tensor::new(vec![1.0, 2.0], vec![2]);
        let reshaped = tensor.reshape(vec![1, 2]).unwrap();
        let cloned = tensor.clone();
        tensor.set(&[0], 3.0).unwrap();
        assert_eq!(reshaped.data(), &[1.0, 2.0]);
        assert_eq!(cloned.data(), &[1.0, 2.0]);
        drop(span);
        let snapshot = finish();
        assert_eq!(snapshot.layers[0].allocations, 10);
        assert_eq!(snapshot.layers[0].calls[Stage::Attention as usize], 1);
        assert_eq!(snapshot.model.allocations, 0);

        start(1);
        let snapshot = finish();
        assert_eq!(snapshot.layers[0].allocations, 0);
        assert_eq!(snapshot.layers[0].calls, [0; 15]);
    }
}
