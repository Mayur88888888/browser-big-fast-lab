#!/usr/bin/env python3
"""Parse a GGUF header from a remote URL via HTTP Range (no full download).

Prints all KV metadata (truncating huge arrays) and the full tensor list
(name, shape, dtype). Fetches more bytes on demand if the header is bigger
than the initial window.
"""
import json
import struct
import sys
import urllib.request

GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian

# gguf value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}


def fetch(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


class Cursor:
    def __init__(self, url: str, window: int = 8 * 1024 * 1024):
        self.url = url
        self.buf = fetch(url, 0, window - 1)
        self.pos = 0

    def need(self, n: int):
        while self.pos + n > len(self.buf):
            # grow: fetch another 8MB
            start = len(self.buf)
            self.buf += fetch(self.url, start, start + 8 * 1024 * 1024 - 1)

    def read(self, n: int) -> bytes:
        self.need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):  return struct.unpack("<I", self.read(4))[0]
    def u64(self):  return struct.unpack("<Q", self.read(8))[0]
    def i32(self):  return struct.unpack("<i", self.read(4))[0]
    def i64(self):  return struct.unpack("<q", self.read(8))[0]
    def f32(self):  return struct.unpack("<f", self.read(4))[0]
    def f64(self):  return struct.unpack("<d", self.read(8))[0]
    def u8(self):   return self.read(1)[0]
    def i8(self):   return struct.unpack("<b", self.read(1))[0]
    def u16(self):  return struct.unpack("<H", self.read(2))[0]
    def i16(self):  return struct.unpack("<h", self.read(2))[0]
    def s(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, t):
        if t == T_U8: return self.u8()
        if t == T_I8: return self.i8()
        if t == T_U16: return self.u16()
        if t == T_I16: return self.i16()
        if t == T_U32: return self.u32()
        if t == T_I32: return self.i32()
        if t == T_F32: return self.f32()
        if t == T_BOOL: return bool(self.u8())
        if t == T_STR: return self.s()
        if t == T_U64: return self.u64()
        if t == T_I64: return self.i64()
        if t == T_F64: return self.f64()
        if t == T_ARR:
            et = self.u32()
            n = self.u64()
            vals = [self.value(et) for _ in range(n)]
            return vals
        raise ValueError(f"bad type {t}")


def main(url: str):
    c = Cursor(url)
    magic = c.u32()
    assert magic == GGUF_MAGIC, f"not GGUF: {magic:#x}"
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    print(f"GGUF v{version} · tensors={n_tensors} · kv={n_kv}\n")

    print("== KV metadata ==")
    kvs = {}
    for _ in range(n_kv):
        k = c.s()
        t = c.u32()
        v = c.value(t)
        kvs[k] = v
        if isinstance(v, list) and len(v) > 8:
            print(f"{k} = [len {len(v)}] {v[:5]} ...")
        elif isinstance(v, str) and len(v) > 300:
            print(f"{k} = <str len {len(v)}> {v[:200]!r} ...")
        else:
            print(f"{k} = {v!r}")

    print("\n== Tensors ==")
    total_bytes_est = 0
    for _ in range(n_tensors):
        name = c.s()
        nd = c.u32()
        dims = [c.u64() for _ in range(nd)]
        ttype = c.u32()
        offset = c.u64()
        print(f"{name:60s} {str(dims):28s} {GGML_TYPES.get(ttype, ttype)}")


if __name__ == "__main__":
    main(sys.argv[1])
