//! Pure-Rust 8-bit RGBA PNG encoder and decoder backed by `flate2`.

use std::io::{Read, Write};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::error::RenderError;

const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Encodes tightly-packed 8-bit sRGB `RGBA` pixels (`width * height * 4` bytes) into a PNG byte vector.
pub fn encode_rgba8_png(width: u32, height: u32, rgba8: &[u8]) -> Result<Vec<u8>, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }
    let row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| RenderError::Png("image width overflow".to_string()))?;
    let expected_len = row_bytes
        .checked_mul(height as usize)
        .ok_or_else(|| RenderError::Png("image size overflow".to_string()))?;
    if rgba8.len() != expected_len {
        return Err(RenderError::Png(format!(
            "expected {expected_len} RGBA8 bytes for {width}x{height} image, got {}",
            rgba8.len()
        )));
    }

    // Build filtered scanlines using PNG Sub filter (type 1) for compact output.
    let mut raw_scanlines = Vec::with_capacity((row_bytes + 1) * (height as usize));
    for row in rgba8.chunks_exact(row_bytes) {
        raw_scanlines.push(1u8); // Filter type 1: Sub
        for i in 0..row_bytes {
            let left = if i >= 4 { row[i - 4] } else { 0 };
            raw_scanlines.push(row[i].wrapping_sub(left));
        }
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&raw_scanlines)
        .map_err(|e| RenderError::Png(format!("zlib compression failed: {e}")))?;
    let idat_payload = encoder
        .finish()
        .map_err(|e| RenderError::Png(format!("zlib finish failed: {e}")))?;

    let mut out = Vec::with_capacity(64 + idat_payload.len());
    out.extend_from_slice(&PNG_SIGNATURE);

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // 8 bits per channel
    ihdr[9] = 6; // color type 6 = RGBA
    ihdr[10] = 0; // deflate compression
    ihdr[11] = 0; // adaptive filtering
    ihdr[12] = 0; // no interlace
    write_chunk(&mut out, *b"IHDR", &ihdr);
    write_chunk(&mut out, *b"IDAT", &idat_payload);
    write_chunk(&mut out, *b"IEND", &[]);

    Ok(out)
}

/// Decodes an 8-bit `RGBA` (or `RGB`) non-interlaced PNG image into `(width, height, rgba8)`.
pub fn decode_rgba8_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), RenderError> {
    if bytes.len() < 8 || bytes[..8] != PNG_SIGNATURE {
        return Err(RenderError::Png("invalid PNG signature".to_string()));
    }

    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut color_type = 0u8;
    let mut saw_ihdr = false;
    let mut saw_iend = false;
    let mut idat_compressed = Vec::new();

    while pos < bytes.len() {
        if pos + 12 > bytes.len() {
            return Err(RenderError::Png("truncated PNG chunk header".to_string()));
        }
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let chunk_type: [u8; 4] = [
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ];
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or_else(|| RenderError::Png("PNG chunk length overflow".to_string()))?;
        let crc_end = data_end
            .checked_add(4)
            .ok_or_else(|| RenderError::Png("PNG chunk CRC overflow".to_string()))?;
        if crc_end > bytes.len() {
            return Err(RenderError::Png("truncated PNG chunk payload".to_string()));
        }

        let chunk_data = &bytes[data_start..data_end];
        let expected_crc = u32::from_be_bytes([
            bytes[data_end],
            bytes[data_end + 1],
            bytes[data_end + 2],
            bytes[data_end + 3],
        ]);
        let actual_crc = crc32(&chunk_type, chunk_data);
        if actual_crc != expected_crc {
            return Err(RenderError::Png(format!(
                "CRC mismatch in {} chunk",
                String::from_utf8_lossy(&chunk_type)
            )));
        }

        match &chunk_type {
            b"IHDR" => {
                if chunk_data.len() != 13 {
                    return Err(RenderError::Png("invalid IHDR chunk length".to_string()));
                }
                width = u32::from_be_bytes([
                    chunk_data[0],
                    chunk_data[1],
                    chunk_data[2],
                    chunk_data[3],
                ]);
                height = u32::from_be_bytes([
                    chunk_data[4],
                    chunk_data[5],
                    chunk_data[6],
                    chunk_data[7],
                ]);
                let bit_depth = chunk_data[8];
                color_type = chunk_data[9];
                let interlace = chunk_data[12];
                if bit_depth != 8 || (color_type != 6 && color_type != 2) || interlace != 0 {
                    return Err(RenderError::Png(format!(
                        "unsupported PNG format: bit_depth={bit_depth}, color_type={color_type}, interlace={interlace}"
                    )));
                }
                saw_ihdr = true;
            }
            b"IDAT" => {
                idat_compressed.extend_from_slice(chunk_data);
            }
            b"IEND" => {
                saw_iend = true;
                break;
            }
            _ => {}
        }

        pos = crc_end;
    }

    if !saw_ihdr || !saw_iend || width == 0 || height == 0 {
        return Err(RenderError::Png(
            "incomplete PNG (missing IHDR or IEND)".to_string(),
        ));
    }

    let bpp = if color_type == 6 { 4usize } else { 3usize };
    let stride = (width as usize) * bpp;
    let expected_raw_len = (stride + 1) * (height as usize);
    let mut decompressed = Vec::with_capacity(expected_raw_len);
    ZlibDecoder::new(&idat_compressed[..])
        .read_to_end(&mut decompressed)
        .map_err(|e| RenderError::Png(format!("zlib decompression failed: {e}")))?;

    if decompressed.len() != expected_raw_len {
        return Err(RenderError::Png(format!(
            "decompressed scanline size mismatch: expected {expected_raw_len}, got {}",
            decompressed.len()
        )));
    }

    let mut unfiltered = vec![0u8; stride * (height as usize)];
    for y in 0..(height as usize) {
        let src_offset = y * (stride + 1);
        let filter = decompressed[src_offset];
        let src_row = &decompressed[src_offset + 1..src_offset + 1 + stride];
        let dst_offset = y * stride;

        for i in 0..stride {
            let left = if i >= bpp {
                unfiltered[dst_offset + i - bpp]
            } else {
                0
            };
            let up = if y > 0 {
                unfiltered[dst_offset - stride + i]
            } else {
                0
            };
            let up_left = if y > 0 && i >= bpp {
                unfiltered[dst_offset - stride + i - bpp]
            } else {
                0
            };

            let reconstructed = match filter {
                0 => src_row[i],
                1 => src_row[i].wrapping_add(left),
                2 => src_row[i].wrapping_add(up),
                3 => {
                    let avg = ((u16::from(left) + u16::from(up)) / 2) as u8;
                    src_row[i].wrapping_add(avg)
                }
                4 => src_row[i].wrapping_add(paeth_predictor(left, up, up_left)),
                other => {
                    return Err(RenderError::Png(format!(
                        "invalid PNG scanline filter type {other}"
                    )));
                }
            };
            unfiltered[dst_offset + i] = reconstructed;
        }
    }

    let rgba8 = if bpp == 4 {
        unfiltered
    } else {
        let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for &[r, g, b] in unfiltered.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
        rgba
    };

    Ok((width, height, rgba8))
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a_i = i32::from(a);
    let b_i = i32::from(b);
    let c_i = i32::from(c);
    let p = a_i + b_i - c_i;
    let pa = (p - a_i).abs();
    let pb = (p - b_i).abs();
    let pc = (p - c_i).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn write_chunk(out: &mut Vec<u8>, chunk_type: [u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(&chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip_preserves_rgba8_pixels() {
        let width = 17u32;
        let height = 11u32;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 13 + y * 7) % 256) as u8);
                pixels.push(((x * 29 + y * 3) % 256) as u8);
                pixels.push(((x * 5 + y * 41) % 256) as u8);
                pixels.push(255u8);
            }
        }
        let encoded = encode_rgba8_png(width, height, &pixels).expect("encode PNG");
        let (dec_w, dec_h, dec_pixels) = decode_rgba8_png(&encoded).expect("decode PNG");
        assert_eq!(dec_w, width);
        assert_eq!(dec_h, height);
        assert_eq!(dec_pixels, pixels);
    }

    #[test]
    fn png_decoder_rejects_corrupted_crc() {
        let pixels = vec![255u8; 4 * 4 * 4];
        let mut encoded = encode_rgba8_png(4, 4, &pixels).expect("encode PNG");
        let flip_idx = encoded.len() - 15;
        encoded[flip_idx] ^= 0xFF;
        assert!(decode_rgba8_png(&encoded).is_err());
    }
}
