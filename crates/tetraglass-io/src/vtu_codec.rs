//! Binary, base64, appended, and zlib decoding helpers for VTK `.vtu` `DataArray` payloads.

use std::io::Read;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use flate2::read::ZlibDecoder;
use tetraglass_core::{
    ComponentCount, DataArray, try_u32_from_i64, try_u32_from_u64, try_u32_from_usize,
};

use crate::error::IoError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    LittleEndian,
    BigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderType {
    UInt32,
    UInt64,
}

impl HeaderType {
    pub(crate) const fn word_size(self) -> usize {
        match self {
            Self::UInt32 => 4,
            Self::UInt64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Compressor {
    None,
    ZLib,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VtuConfig {
    pub header_type: HeaderType,
    pub byte_order: ByteOrder,
    pub compressor: Compressor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendedEncoding {
    Base64,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VtuDataType {
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl VtuDataType {
    pub(crate) fn parse(raw: &str, offset: u64) -> Result<Self, IoError> {
        match raw {
            "Float32" => Ok(Self::Float32),
            "Float64" => Ok(Self::Float64),
            "Int8" => Ok(Self::Int8),
            "UInt8" => Ok(Self::UInt8),
            "Int16" => Ok(Self::Int16),
            "UInt16" => Ok(Self::UInt16),
            "Int32" => Ok(Self::Int32),
            "UInt32" => Ok(Self::UInt32),
            "Int64" => Ok(Self::Int64),
            "UInt64" => Ok(Self::UInt64),
            other => Err(IoError::UnsupportedFeature {
                offset,
                feature: format!("DataArray type '{other}'"),
            }),
        }
    }

    pub(crate) const fn byte_width(self) -> usize {
        match self {
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Float32 | Self::Int32 | Self::UInt32 => 4,
            Self::Float64 | Self::Int64 | Self::UInt64 => 8,
        }
    }
}

const MAX_DECOMPRESSED_BYTES: usize = 512 * 1024 * 1024;
const MAX_ZLIB_BLOCKS: usize = 65_536;

fn b64_encoded_len(raw_len: usize) -> Option<usize> {
    raw_len.checked_add(2)?.checked_div(3)?.checked_mul(4)
}

fn read_header_word(
    bytes: &[u8],
    cfg: VtuConfig,
    array_name: &str,
    xml_offset: u64,
) -> Result<usize, IoError> {
    let raw_u64 = match (cfg.header_type, cfg.byte_order) {
        (HeaderType::UInt32, ByteOrder::LittleEndian) => {
            let b: [u8; 4] = bytes[..4].try_into().map_err(|_| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated UInt32 header".to_string(),
            })?;
            u64::from(u32::from_le_bytes(b))
        }
        (HeaderType::UInt32, ByteOrder::BigEndian) => {
            let b: [u8; 4] = bytes[..4].try_into().map_err(|_| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated UInt32 header".to_string(),
            })?;
            u64::from(u32::from_be_bytes(b))
        }
        (HeaderType::UInt64, ByteOrder::LittleEndian) => {
            let b: [u8; 8] = bytes[..8].try_into().map_err(|_| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated UInt64 header".to_string(),
            })?;
            u64::from_le_bytes(b)
        }
        (HeaderType::UInt64, ByteOrder::BigEndian) => {
            let b: [u8; 8] = bytes[..8].try_into().map_err(|_| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated UInt64 header".to_string(),
            })?;
            u64::from_be_bytes(b)
        }
    };
    usize::try_from(raw_u64).map_err(|_| IoError::BinaryData {
        array: array_name.to_string(),
        offset: xml_offset,
        reason: format!("header length {raw_u64} overflows usize"),
    })
}

pub(crate) fn decode_inline_binary(
    array_name: &str,
    text: &str,
    xml_offset: u64,
    cfg: VtuConfig,
) -> Result<Vec<u8>, IoError> {
    let cleaned: String = text.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    decode_base64_stream(array_name, &cleaned, xml_offset, cfg)
}

pub(crate) fn decode_appended_slice(
    array_name: &str,
    appended_payload: &[u8],
    data_offset: usize,
    xml_offset: u64,
    encoding: AppendedEncoding,
    cfg: VtuConfig,
) -> Result<Vec<u8>, IoError> {
    let tail = appended_payload
        .get(data_offset..)
        .ok_or_else(|| IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: format!(
                "appended offset {data_offset} exceeds appended section length {}",
                appended_payload.len()
            ),
        })?;

    match encoding {
        AppendedEncoding::Base64 => {
            let s = std::str::from_utf8(tail).map_err(|e| IoError::Base64Decode {
                array: array_name.to_string(),
                offset: xml_offset,
                message: e.to_string(),
            })?;
            decode_base64_stream(
                array_name,
                s.trim_end_matches(|c: char| c.is_ascii_whitespace()),
                xml_offset,
                cfg,
            )
        }
        AppendedEncoding::Raw => decode_raw_stream(array_name, tail, xml_offset, cfg),
    }
}

fn decode_base64_stream(
    array_name: &str,
    s: &str,
    xml_offset: u64,
    cfg: VtuConfig,
) -> Result<Vec<u8>, IoError> {
    let word_size = cfg.header_type.word_size();
    let first_word_b64_len = b64_encoded_len(word_size).ok_or_else(|| IoError::BinaryData {
        array: array_name.to_string(),
        offset: xml_offset,
        reason: "header word size overflow".to_string(),
    })?;

    if s.len() < first_word_b64_len {
        return Err(IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: "truncated base64 header".to_string(),
        });
    }

    match cfg.compressor {
        Compressor::None => {
            let first_chunk = &s[..first_word_b64_len];
            let first_decoded =
                BASE64_STANDARD
                    .decode(first_chunk)
                    .map_err(|e| IoError::Base64Decode {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        message: e.to_string(),
                    })?;
            let payload_len = read_header_word(&first_decoded, cfg, array_name, xml_offset)?;
            if payload_len > MAX_DECOMPRESSED_BYTES {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: format!("payload length {payload_len} exceeds safety limit"),
                });
            }

            if first_chunk.ends_with('=') {
                let payload_b64_len =
                    b64_encoded_len(payload_len).ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "payload base64 length overflow".to_string(),
                    })?;
                let end = first_word_b64_len
                    .checked_add(payload_b64_len)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "base64 slice overflow".to_string(),
                    })?;
                let payload_b64 =
                    s.get(first_word_b64_len..end)
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            reason: "truncated base64 payload".to_string(),
                        })?;
                let decoded =
                    BASE64_STANDARD
                        .decode(payload_b64)
                        .map_err(|e| IoError::Base64Decode {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            message: e.to_string(),
                        })?;
                if decoded.len() < payload_len {
                    return Err(IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "decoded payload shorter than header length".to_string(),
                    });
                }
                Ok(decoded[..payload_len].to_vec())
            } else {
                let total_raw =
                    word_size
                        .checked_add(payload_len)
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            reason: "header + payload size overflow".to_string(),
                        })?;
                let total_b64_len =
                    b64_encoded_len(total_raw).ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "base64 length overflow".to_string(),
                    })?;
                let combined_b64 = s.get(..total_b64_len).ok_or_else(|| IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "truncated combined base64 block".to_string(),
                })?;
                let decoded =
                    BASE64_STANDARD
                        .decode(combined_b64)
                        .map_err(|e| IoError::Base64Decode {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            message: e.to_string(),
                        })?;
                if decoded.len() < total_raw {
                    return Err(IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "decoded combined block shorter than header + payload".to_string(),
                    });
                }
                Ok(decoded[word_size..total_raw].to_vec())
            }
        }
        Compressor::ZLib => {
            let first_decoded = BASE64_STANDARD
                .decode(&s[..first_word_b64_len])
                .map_err(|e| IoError::Base64Decode {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    message: e.to_string(),
                })?;
            let num_blocks = read_header_word(&first_decoded, cfg, array_name, xml_offset)?;
            if num_blocks == 0 {
                return Ok(Vec::new());
            }
            if num_blocks > MAX_ZLIB_BLOCKS {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: format!("zlib num_blocks {num_blocks} exceeds limit"),
                });
            }
            let header_words =
                3usize
                    .checked_add(num_blocks)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "zlib header word count overflow".to_string(),
                    })?;
            let header_raw_bytes =
                header_words
                    .checked_mul(word_size)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "zlib header byte count overflow".to_string(),
                    })?;
            let header_b64_len =
                b64_encoded_len(header_raw_bytes).ok_or_else(|| IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "zlib header base64 length overflow".to_string(),
                })?;
            let header_b64 = s.get(..header_b64_len).ok_or_else(|| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated zlib base64 header".to_string(),
            })?;
            let header_bytes =
                BASE64_STANDARD
                    .decode(header_b64)
                    .map_err(|e| IoError::Base64Decode {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        message: e.to_string(),
                    })?;
            if header_bytes.len() < header_raw_bytes {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "decoded zlib header shorter than expected".to_string(),
                });
            }

            let block_size = read_header_word(
                &header_bytes[word_size..2 * word_size],
                cfg,
                array_name,
                xml_offset,
            )?;
            let last_block_size = read_header_word(
                &header_bytes[2 * word_size..3 * word_size],
                cfg,
                array_name,
                xml_offset,
            )?;

            let mut block_csizes = Vec::with_capacity(num_blocks);
            let mut total_compressed = 0usize;
            for b_idx in 0..num_blocks {
                let off = (3 + b_idx) * word_size;
                let csize = read_header_word(
                    &header_bytes[off..off + word_size],
                    cfg,
                    array_name,
                    xml_offset,
                )?;
                total_compressed =
                    total_compressed
                        .checked_add(csize)
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            reason: "compressed block size sum overflow".to_string(),
                        })?;
                block_csizes.push(csize);
            }

            let payload_b64_len =
                b64_encoded_len(total_compressed).ok_or_else(|| IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "compressed payload base64 length overflow".to_string(),
                })?;
            let payload_end =
                header_b64_len
                    .checked_add(payload_b64_len)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "compressed base64 slice overflow".to_string(),
                    })?;
            let payload_b64 =
                s.get(header_b64_len..payload_end)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "truncated compressed base64 payload".to_string(),
                    })?;
            let compressed_bytes =
                BASE64_STANDARD
                    .decode(payload_b64)
                    .map_err(|e| IoError::Base64Decode {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        message: e.to_string(),
                    })?;
            if compressed_bytes.len() < total_compressed {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "decoded compressed payload shorter than expected".to_string(),
                });
            }

            decompress_zlib_blocks(
                array_name,
                xml_offset,
                &compressed_bytes[..total_compressed],
                &block_csizes,
                block_size,
                last_block_size,
            )
        }
    }
}

fn decode_raw_stream(
    array_name: &str,
    raw: &[u8],
    xml_offset: u64,
    cfg: VtuConfig,
) -> Result<Vec<u8>, IoError> {
    let word_size = cfg.header_type.word_size();
    if raw.len() < word_size {
        return Err(IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: "truncated raw binary header".to_string(),
        });
    }
    match cfg.compressor {
        Compressor::None => {
            let payload_len = read_header_word(raw, cfg, array_name, xml_offset)?;
            if payload_len > MAX_DECOMPRESSED_BYTES {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: format!("payload length {payload_len} exceeds safety limit"),
                });
            }
            let end = word_size
                .checked_add(payload_len)
                .ok_or_else(|| IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "raw payload slice overflow".to_string(),
                })?;
            let slice = raw.get(word_size..end).ok_or_else(|| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated raw binary payload".to_string(),
            })?;
            Ok(slice.to_vec())
        }
        Compressor::ZLib => {
            let num_blocks = read_header_word(raw, cfg, array_name, xml_offset)?;
            if num_blocks == 0 {
                return Ok(Vec::new());
            }
            if num_blocks > MAX_ZLIB_BLOCKS {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: format!("zlib num_blocks {num_blocks} exceeds limit"),
                });
            }
            let header_words =
                3usize
                    .checked_add(num_blocks)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "zlib header word count overflow".to_string(),
                    })?;
            let header_len =
                header_words
                    .checked_mul(word_size)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "zlib header byte count overflow".to_string(),
                    })?;
            if raw.len() < header_len {
                return Err(IoError::BinaryData {
                    array: array_name.to_string(),
                    offset: xml_offset,
                    reason: "truncated raw zlib header".to_string(),
                });
            }
            let block_size =
                read_header_word(&raw[word_size..2 * word_size], cfg, array_name, xml_offset)?;
            let last_block_size = read_header_word(
                &raw[2 * word_size..3 * word_size],
                cfg,
                array_name,
                xml_offset,
            )?;
            let mut block_csizes = Vec::with_capacity(num_blocks);
            let mut total_compressed = 0usize;
            for b_idx in 0..num_blocks {
                let off = (3 + b_idx) * word_size;
                let csize =
                    read_header_word(&raw[off..off + word_size], cfg, array_name, xml_offset)?;
                total_compressed =
                    total_compressed
                        .checked_add(csize)
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: xml_offset,
                            reason: "compressed block size sum overflow".to_string(),
                        })?;
                block_csizes.push(csize);
            }
            let payload_end =
                header_len
                    .checked_add(total_compressed)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "raw compressed slice overflow".to_string(),
                    })?;
            let compressed_slice =
                raw.get(header_len..payload_end)
                    .ok_or_else(|| IoError::BinaryData {
                        array: array_name.to_string(),
                        offset: xml_offset,
                        reason: "truncated raw compressed payload".to_string(),
                    })?;
            decompress_zlib_blocks(
                array_name,
                xml_offset,
                compressed_slice,
                &block_csizes,
                block_size,
                last_block_size,
            )
        }
    }
}

fn decompress_zlib_blocks(
    array_name: &str,
    xml_offset: u64,
    compressed_bytes: &[u8],
    block_csizes: &[usize],
    block_size: usize,
    last_block_size: usize,
) -> Result<Vec<u8>, IoError> {
    let expected_uncompressed = block_csizes
        .len()
        .saturating_sub(1)
        .checked_mul(block_size)
        .and_then(|n| n.checked_add(last_block_size))
        .ok_or_else(|| IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: "uncompressed size overflow".to_string(),
        })?;
    if expected_uncompressed > MAX_DECOMPRESSED_BYTES {
        return Err(IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: format!("uncompressed size {expected_uncompressed} exceeds safety limit"),
        });
    }

    let mut out = Vec::with_capacity(expected_uncompressed);
    let mut cursor = 0usize;
    for &csize in block_csizes {
        let block = compressed_bytes
            .get(cursor..cursor + csize)
            .ok_or_else(|| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: "truncated compressed block".to_string(),
            })?;
        let mut decoder = ZlibDecoder::new(block);
        decoder
            .read_to_end(&mut out)
            .map_err(|e| IoError::BinaryData {
                array: array_name.to_string(),
                offset: xml_offset,
                reason: format!("zlib decompression failed: {e}"),
            })?;
        cursor += csize;
    }
    Ok(out)
}

pub(crate) fn parse_ascii_f64(
    array_name: &str,
    text: &str,
    xml_offset: u64,
) -> Result<Vec<f64>, IoError> {
    text.split_ascii_whitespace()
        .map(|tok| {
            tok.parse::<f64>().map_err(|_| IoError::InvalidNumber {
                array: array_name.to_string(),
                text: tok.to_string(),
                offset: xml_offset,
            })
        })
        .collect()
}

pub(crate) fn parse_ascii_u32_indices(
    array_name: &str,
    text: &str,
    xml_offset: u64,
) -> Result<Vec<u32>, IoError> {
    text.split_ascii_whitespace()
        .map(|tok| {
            let val = tok.parse::<i64>().map_err(|_| IoError::InvalidNumber {
                array: array_name.to_string(),
                text: tok.to_string(),
                offset: xml_offset,
            })?;
            Ok(try_u32_from_i64(val)?)
        })
        .collect()
}

pub(crate) fn parse_ascii_u8(
    array_name: &str,
    text: &str,
    xml_offset: u64,
) -> Result<Vec<u8>, IoError> {
    text.split_ascii_whitespace()
        .map(|tok| {
            tok.parse::<u8>().map_err(|_| IoError::InvalidNumber {
                array: array_name.to_string(),
                text: tok.to_string(),
                offset: xml_offset,
            })
        })
        .collect()
}

pub(crate) fn parse_ascii_data_array(
    array_name: &str,
    dtype: VtuDataType,
    components: ComponentCount,
    text: &str,
    xml_offset: u64,
) -> Result<DataArray, IoError> {
    match dtype {
        VtuDataType::Float32 => {
            let vals: Vec<f32> = text
                .split_ascii_whitespace()
                .map(|tok| {
                    tok.parse::<f32>().map_err(|_| IoError::InvalidNumber {
                        array: array_name.to_string(),
                        text: tok.to_string(),
                        offset: xml_offset,
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(DataArray::from_f32(components, vals)?)
        }
        VtuDataType::Float64 => {
            let vals = parse_ascii_f64(array_name, text, xml_offset)?;
            Ok(DataArray::from_f64(components, vals)?)
        }
        VtuDataType::Int32 | VtuDataType::Int16 | VtuDataType::UInt16 | VtuDataType::Int8 => {
            let vals: Vec<i32> = text
                .split_ascii_whitespace()
                .map(|tok| {
                    tok.parse::<i32>().map_err(|_| IoError::InvalidNumber {
                        array: array_name.to_string(),
                        text: tok.to_string(),
                        offset: xml_offset,
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(DataArray::from_i32(components, vals)?)
        }
        VtuDataType::Int64 | VtuDataType::UInt32 | VtuDataType::UInt64 => {
            let vals: Vec<i64> = text
                .split_ascii_whitespace()
                .map(|tok| {
                    tok.parse::<i64>().map_err(|_| IoError::InvalidNumber {
                        array: array_name.to_string(),
                        text: tok.to_string(),
                        offset: xml_offset,
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(DataArray::from_i64(components, vals)?)
        }
        VtuDataType::UInt8 => {
            let vals = parse_ascii_u8(array_name, text, xml_offset)?;
            Ok(DataArray::from_u8(components, vals)?)
        }
    }
}

fn check_byte_multiple(
    array_name: &str,
    bytes: &[u8],
    dtype: VtuDataType,
    xml_offset: u64,
) -> Result<(), IoError> {
    let w = dtype.byte_width();
    if !bytes.len().is_multiple_of(w) {
        return Err(IoError::BinaryData {
            array: array_name.to_string(),
            offset: xml_offset,
            reason: format!(
                "binary byte length {} is not a multiple of element width {w}",
                bytes.len()
            ),
        });
    }
    Ok(())
}

pub(crate) fn decode_bytes_f64(
    array_name: &str,
    bytes: &[u8],
    dtype: VtuDataType,
    byte_order: ByteOrder,
    xml_offset: u64,
) -> Result<Vec<f64>, IoError> {
    check_byte_multiple(array_name, bytes, dtype, xml_offset)?;
    match dtype {
        VtuDataType::Float64 => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| match byte_order {
                ByteOrder::LittleEndian => f64::from_bits(u64::from_le_bytes(b)),
                ByteOrder::BigEndian => f64::from_bits(u64::from_be_bytes(b)),
            })
            .collect()),
        VtuDataType::Float32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => f32::from_bits(u32::from_le_bytes(b)),
                    ByteOrder::BigEndian => f32::from_bits(u32::from_be_bytes(b)),
                };
                f64::from(v)
            })
            .collect()),
        other => Err(IoError::UnsupportedFeature {
            offset: xml_offset,
            feature: format!("Points coordinate type {other:?}"),
        }),
    }
}

pub(crate) fn decode_bytes_u32_indices(
    array_name: &str,
    bytes: &[u8],
    dtype: VtuDataType,
    byte_order: ByteOrder,
    xml_offset: u64,
) -> Result<Vec<u32>, IoError> {
    check_byte_multiple(array_name, bytes, dtype, xml_offset)?;
    match dtype {
        VtuDataType::Int32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => i32::from_le_bytes(b),
                    ByteOrder::BigEndian => i32::from_be_bytes(b),
                };
                Ok(try_u32_from_i64(i64::from(v))?)
            })
            .collect(),
        VtuDataType::UInt32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| match byte_order {
                ByteOrder::LittleEndian => u32::from_le_bytes(b),
                ByteOrder::BigEndian => u32::from_be_bytes(b),
            })
            .collect()),
        VtuDataType::Int64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => i64::from_le_bytes(b),
                    ByteOrder::BigEndian => i64::from_be_bytes(b),
                };
                Ok(try_u32_from_i64(v)?)
            })
            .collect(),
        VtuDataType::UInt64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => u64::from_le_bytes(b),
                    ByteOrder::BigEndian => u64::from_be_bytes(b),
                };
                Ok(try_u32_from_u64(v)?)
            })
            .collect(),
        VtuDataType::UInt8 => Ok(bytes.iter().map(|&b| u32::from(b)).collect()),
        VtuDataType::UInt16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => u16::from_le_bytes(b),
                    ByteOrder::BigEndian => u16::from_be_bytes(b),
                };
                u32::from(v)
            })
            .collect()),
        VtuDataType::Int16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| {
                let v = match byte_order {
                    ByteOrder::LittleEndian => i16::from_le_bytes(b),
                    ByteOrder::BigEndian => i16::from_be_bytes(b),
                };
                Ok(try_u32_from_i64(i64::from(v))?)
            })
            .collect(),
        other => Err(IoError::UnsupportedFeature {
            offset: xml_offset,
            feature: format!("index DataArray type {other:?}"),
        }),
    }
}

pub(crate) fn decode_bytes_data_array(
    array_name: &str,
    bytes: &[u8],
    dtype: VtuDataType,
    components: ComponentCount,
    byte_order: ByteOrder,
    xml_offset: u64,
) -> Result<DataArray, IoError> {
    check_byte_multiple(array_name, bytes, dtype, xml_offset)?;
    match dtype {
        VtuDataType::Float32 => {
            let vals: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| match byte_order {
                    ByteOrder::LittleEndian => f32::from_bits(u32::from_le_bytes(b)),
                    ByteOrder::BigEndian => f32::from_bits(u32::from_be_bytes(b)),
                })
                .collect();
            Ok(DataArray::from_f32(components, vals)?)
        }
        VtuDataType::Float64 => {
            let vals = decode_bytes_f64(array_name, bytes, dtype, byte_order, xml_offset)?;
            Ok(DataArray::from_f64(components, vals)?)
        }
        VtuDataType::Int32 => {
            let vals: Vec<i32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| match byte_order {
                    ByteOrder::LittleEndian => i32::from_le_bytes(b),
                    ByteOrder::BigEndian => i32::from_be_bytes(b),
                })
                .collect();
            Ok(DataArray::from_i32(components, vals)?)
        }
        VtuDataType::Int64 => {
            let vals: Vec<i64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&b| match byte_order {
                    ByteOrder::LittleEndian => i64::from_le_bytes(b),
                    ByteOrder::BigEndian => i64::from_be_bytes(b),
                })
                .collect();
            Ok(DataArray::from_i64(components, vals)?)
        }
        VtuDataType::UInt8 => Ok(DataArray::from_u8(components, bytes.to_vec())?),
        VtuDataType::Int8 => {
            let vals: Vec<i32> = bytes.iter().map(|&b| i32::from(b as i8)).collect();
            Ok(DataArray::from_i32(components, vals)?)
        }
        VtuDataType::Int16 => {
            let vals: Vec<i32> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| {
                    let v = match byte_order {
                        ByteOrder::LittleEndian => i16::from_le_bytes(b),
                        ByteOrder::BigEndian => i16::from_be_bytes(b),
                    };
                    i32::from(v)
                })
                .collect();
            Ok(DataArray::from_i32(components, vals)?)
        }
        VtuDataType::UInt16 => {
            let vals: Vec<i32> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| {
                    let v = match byte_order {
                        ByteOrder::LittleEndian => u16::from_le_bytes(b),
                        ByteOrder::BigEndian => u16::from_be_bytes(b),
                    };
                    i32::from(v)
                })
                .collect();
            Ok(DataArray::from_i32(components, vals)?)
        }
        VtuDataType::UInt32 => {
            let vals: Vec<i64> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| {
                    let v = match byte_order {
                        ByteOrder::LittleEndian => u32::from_le_bytes(b),
                        ByteOrder::BigEndian => u32::from_be_bytes(b),
                    };
                    i64::from(v)
                })
                .collect();
            Ok(DataArray::from_i64(components, vals)?)
        }
        VtuDataType::UInt64 => {
            let vals: Vec<i64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&b| {
                    let v = match byte_order {
                        ByteOrder::LittleEndian => u64::from_le_bytes(b),
                        ByteOrder::BigEndian => u64::from_be_bytes(b),
                    };
                    i64::try_from(v).unwrap_or(i64::MAX)
                })
                .collect();
            Ok(DataArray::from_i64(components, vals)?)
        }
    }
}

/// Converts VTK 9.4+ VTU 2.3 polyhedron arrays (`face_connectivity`, `face_offsets`,
/// `polyhedron_to_faces`, `polyhedron_offsets`) into classic `(face_offsets, faces)` stream.
pub(crate) fn convert_vtu23_polyhedra_to_stream(
    cell_types: &[u8],
    face_connectivity: &[u32],
    face_offsets: &[u32],
    polyhedron_to_faces: &[u32],
    polyhedron_offsets: &[u32],
    xml_offset: u64,
) -> Result<(Vec<u32>, Vec<u32>), IoError> {
    if polyhedron_offsets.len() != cell_types.len() {
        return Err(IoError::InvalidVtuStructure {
            offset: xml_offset,
            reason: format!(
                "polyhedron_offsets length {} does not match cell count {}",
                polyhedron_offsets.len(),
                cell_types.len()
            ),
        });
    }

    let mut out_face_offsets = Vec::with_capacity(cell_types.len());
    let mut out_faces = Vec::new();
    let mut prev_poly_off = 0usize;

    for (cell_idx, (&vtk_type, &poly_end_u32)) in
        cell_types.iter().zip(polyhedron_offsets.iter()).enumerate()
    {
        let poly_end = poly_end_u32 as usize;
        if poly_end < prev_poly_off || poly_end > polyhedron_to_faces.len() {
            return Err(IoError::InvalidVtuStructure {
                offset: xml_offset,
                reason: format!("invalid polyhedron_offsets entry at cell {cell_idx}"),
            });
        }
        let face_ids = &polyhedron_to_faces[prev_poly_off..poly_end];
        if vtk_type == 42 {
            out_faces.push(try_u32_from_usize(face_ids.len())?);
            for &face_id_u32 in face_ids {
                let face_id = face_id_u32 as usize;
                let f_end =
                    *face_offsets
                        .get(face_id)
                        .ok_or_else(|| IoError::InvalidVtuStructure {
                            offset: xml_offset,
                            reason: format!("face_id {face_id} out of bounds in face_offsets"),
                        })? as usize;
                let f_start = if face_id == 0 {
                    0
                } else {
                    face_offsets[face_id - 1] as usize
                };
                if f_end < f_start || f_end > face_connectivity.len() {
                    return Err(IoError::InvalidVtuStructure {
                        offset: xml_offset,
                        reason: format!("invalid face_offsets range for face {face_id}"),
                    });
                }
                let verts = &face_connectivity[f_start..f_end];
                out_faces.push(try_u32_from_usize(verts.len())?);
                out_faces.extend_from_slice(verts);
            }
        }
        out_face_offsets.push(try_u32_from_usize(out_faces.len())?);
        prev_poly_off = poly_end;
    }

    Ok((out_face_offsets, out_faces))
}
