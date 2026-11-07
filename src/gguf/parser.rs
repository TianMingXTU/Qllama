//! GGUF file parser

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use super::types::{GgmlType, GgufHeader, GgufValue, TensorInfo};

/// GGUF 规范当前版本，对应 gguf.h 的 GGUF_VERSION。
const GGUF_VERSION: u32 = 3;

/// ggml.h 的张量名称长度上限 GGML_MAX_NAME，名称需额外容纳终止字节。
const GGML_MAX_NAME: usize = 64;

/// GGUF 规范对 metadata 键名的长度上限。
const MAX_METADATA_KEY_BYTES: usize = 65535;

/// GGUF file parser
pub struct GgufFile {
    /// File header
    pub header: GgufHeader,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor information list
    pub tensor_infos: Vec<TensorInfo>,
    /// Offset to tensor data start
    pub tensor_data_offset: u64,
    /// 权重数据段的内存映射；仅 `from_path` 构造时存在。
    ///
    /// `Mmap` 独立于创建它的 `File`，句柄自身保活映射，因此无需在此额外持有文件。
    /// 对 loader 模块可见，便于在同一父模块内拆分实现。
    pub(super) tensor_data: Option<Arc<Mmap>>,
}

impl GgufFile {
    /// 解析并只读映射 GGUF 文件；映射及其权重存活期间不得修改或截断原文件。
    pub fn from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        // 复用既有 mmap 入口。调用方须保持原文件不变；Arc 让权重共享并保活映射，
        // 不复制文件内容，也不为每个权重建立新映射。
        let tensor_data = Some(Arc::new(unsafe { Mmap::map(&file)? }));
        Self::from_reader(&mut file).map(|parsed| Self {
            tensor_data,
            ..parsed
        })
    }

    /// Parse a GGUF file from a reader
    pub fn from_reader<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        // Read header
        let header = Self::read_header(reader).map_err(|error| with_section("header", error))?;

        // Validate magic
        if &header.magic != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: {:?}", header.magic),
            ));
        }

        Self::validate_version(header.version)?;

        // Read metadata
        let metadata = Self::read_metadata(reader, header.metadata_kv_count)
            .map_err(|error| with_section("metadata", error))?;

        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(GgufValue::Uint32(value)) if *value != 0 && *value % 8 == 0 => u64::from(*value),
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "general.alignment must be a nonzero UINT32 multiple of 8",
                ));
            }
        };

        let tensor_infos = Self::read_tensor_infos(reader, header.tensor_count)
            .map_err(|error| with_section("tensor index", error))?;

        // 先求余数，避免已对齐的位置在上取整中发生不必要的溢出。
        let current_pos = reader.stream_position()?;
        let padding = (alignment - current_pos % alignment) % alignment;
        let tensor_data_offset = current_pos.checked_add(padding).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Tensor data offset overflow")
        })?;

        Ok(Self {
            header,
            metadata,
            tensor_infos,
            tensor_data_offset,
            // 读端无法映射：权重数据不在本结构内，加载张量会得到明确错误。
            tensor_data: None,
        })
    }

    fn read_header<R: Read>(reader: &mut R) -> io::Result<GgufHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        let version = read_u32(reader)?;
        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    /// 校验 GGUF 版本号。
    ///
    /// 低 16 位为 0 的版本号（如 0x30000000）是大端文件按小端读取的结果，
    /// 若继续解析会把后续所有字段整个错位，因此必须在此拦截。
    fn validate_version(version: u32) -> io::Result<()> {
        if version == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bad GGUF version: 0",
            ));
        }
        if (version & 0x0000_FFFF) == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF version {version} has a zero low half; the file may be big-endian"),
            ));
        }
        if version == 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUFv1 is no longer supported",
            ));
        }
        if version > GGUF_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF version {version} exceeds supported version {GGUF_VERSION}"),
            ));
        }
        Ok(())
    }

    fn read_metadata<R: Read>(
        reader: &mut R,
        count: u64,
    ) -> io::Result<HashMap<String, GgufValue>> {
        let count = usize_from_u64(count)?;
        let mut metadata = HashMap::new();

        for _ in 0..count {
            let key = Self::read_metadata_key(reader)?;
            // 同名键会被 insert 静默覆盖，使 metadata_count 与真实条目数不一致。
            if metadata.contains_key(&key) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Duplicate metadata key: {key}"),
                ));
            }
            let value = Self::read_value(reader)?;
            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    /// 读取 metadata 键名：非空且不超过规范长度上限。
    fn read_metadata_key<R: Read>(reader: &mut R) -> io::Result<String> {
        let key = read_string(reader)?;
        if key.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF metadata key must not be empty",
            ));
        }
        if key.len() > MAX_METADATA_KEY_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF metadata key exceeds {MAX_METADATA_KEY_BYTES} bytes: {}",
                    key.len()
                ),
            ));
        }
        Ok(key)
    }

    fn read_value<R: Read>(reader: &mut R) -> io::Result<GgufValue> {
        let value_type = read_u32(reader)?;
        if value_type != 9 {
            return Self::read_value_of_type(reader, value_type);
        }

        let array_type = read_u32(reader)?;
        // 与上游读取器一致，嵌套数组不受支持；空数组也必须验证元素类型。
        if array_type == 9 || array_type > 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported array element type: {}", array_type),
            ));
        }
        let array_count = read_usize(reader)?;
        // 只有读到元素后才增长容器，避免损坏的计数触发巨额预分配。
        let mut array = Vec::new();
        for _ in 0..array_count {
            array.push(Self::read_value_of_type(reader, array_type)?);
        }
        Ok(GgufValue::Array(array))
    }

    fn read_value_of_type<R: Read>(reader: &mut R, value_type: u32) -> io::Result<GgufValue> {
        Ok(match value_type {
            0 => GgufValue::Uint8(read_u8(reader)?),
            1 => GgufValue::Int8(read_i8(reader)?),
            2 => GgufValue::Uint16(read_u16(reader)?),
            3 => GgufValue::Int16(read_i16(reader)?),
            4 => GgufValue::Uint32(read_u32(reader)?),
            5 => GgufValue::Int32(read_i32(reader)?),
            6 => GgufValue::Float32(read_f32(reader)?),
            7 => GgufValue::Bool(read_bool(reader)?),
            8 => GgufValue::String(read_string(reader)?),
            10 => GgufValue::Uint64(read_u64(reader)?),
            11 => GgufValue::Int64(read_i64(reader)?),
            12 => GgufValue::Float64(read_f64(reader)?),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown GGUF value type: {}", value_type),
                ))
            }
        })
    }

    fn read_tensor_infos<R: Read>(reader: &mut R, count: u64) -> io::Result<Vec<TensorInfo>> {
        let count = usize_from_u64(count)?;
        let mut tensor_infos: Vec<TensorInfo> = Vec::new();

        for _ in 0..count {
            let name = Self::read_tensor_name(reader)?;
            // 重名张量只能通过 find_tensor 定位到第一个 offset，其余权重将无法访问。
            if tensor_infos.iter().any(|info| info.name == name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Duplicate tensor name: {name}"),
                ));
            }
            let n_dims = read_u32(reader)?;
            if n_dims > 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Tensor dimension count exceeds GGML limit: {}", n_dims),
                ));
            }

            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(read_u64(reader)?);
            }

            let ggml_type_u32 = read_u32(reader)?;
            let ggml_type = Self::ggml_type_from_u32(ggml_type_u32)?;
            let offset = read_u64(reader)?;

            tensor_infos.push(TensorInfo {
                name,
                n_dims,
                dims,
                ggml_type,
                offset,
            });
        }

        Ok(tensor_infos)
    }

    /// 读取张量名称：长度必须小于 ggml.h 的 GGML_MAX_NAME（含终止字节）。
    fn read_tensor_name<R: Read>(reader: &mut R) -> io::Result<String> {
        let name = read_string(reader)?;
        if name.len() >= GGML_MAX_NAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF tensor name must be shorter than {GGML_MAX_NAME} bytes: {}",
                    name.len()
                ),
            ));
        }
        Ok(name)
    }

    fn ggml_type_from_u32(value: u32) -> io::Result<GgmlType> {
        Ok(match value {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            // ggml.h 中 16..23 属于 IQ 系列，不能当作整数或 F64 解读。
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unsupported GGML type: {}", value),
                ))
            }
        })
    }

    /// Get tensor count
    pub fn tensor_count(&self) -> u64 {
        self.header.tensor_count
    }

    /// Get metadata count
    pub fn metadata_count(&self) -> u64 {
        self.header.metadata_kv_count
    }

    /// Find tensor by name
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }
}

// Helper functions for reading binary data

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> io::Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> io::Result<i16> {
    Ok(read_u16(reader)? as i16)
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    Ok(read_u32(reader)? as i32)
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> io::Result<i64> {
    Ok(read_u64(reader)? as i64)
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(reader: &mut R) -> io::Result<bool> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid GGUF boolean: {}", value),
        )),
    }
}

fn usize_from_u64(value: u64) -> io::Result<usize> {
    usize::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF length exceeds usize range",
        )
    })
}

fn read_usize<R: Read>(reader: &mut R) -> io::Result<usize> {
    usize_from_u64(read_u64(reader)?)
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    // 这是读取器的资源上限，并非 GGUF 规范限制；必须在分配前检查。
    const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
    let len = read_u64(reader)?;
    if len > MAX_STRING_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF string exceeds 64 MiB limit",
        ));
    }
    let mut buf = vec![0u8; usize_from_u64(len)?];
    reader.read_exact(&mut buf)?;

    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// 给读取阶段的 io 错误补上"读到哪一段"的上下文。
///
/// 裸的 `failed to fill whole buffer` 无法告诉使用者文件坏在哪里；对截断或损坏的
/// 模型文件，段名是唯一能定位问题的线索。
fn with_section(section: &str, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("while reading the GGUF {section}: {error}"),
    )
}
