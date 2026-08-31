/** Binary framing shared with demo/protocol.py */

export const TYPE_JSON = 0x00;
export const TYPE_PCM = 0x01;
export const TYPE_TEXT = 0x02;
export const TYPE_EOS = 0x03;
export const SAMPLE_RATE_IN = 16000;
export const SAMPLE_RATE_OUT = 24000;

export function packFrame(type, payloadBytes = new Uint8Array(0)) {
  const out = new Uint8Array(1 + payloadBytes.byteLength);
  out[0] = type;
  out.set(payloadBytes, 1);
  return out.buffer;
}

export function unpackFrame(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  if (u8.byteLength < 1) throw new Error("empty websocket message");
  return { type: u8[0], payload: u8.subarray(1) };
}

export function packJson(obj) {
  const enc = new TextEncoder().encode(JSON.stringify(obj));
  return packFrame(TYPE_JSON, enc);
}

export function unpackJson(payload) {
  return JSON.parse(new TextDecoder().decode(payload));
}

export function packPcmI16(int16Array) {
  return packFrame(TYPE_PCM, new Uint8Array(int16Array.buffer, int16Array.byteOffset, int16Array.byteLength));
}

export function packEos() {
  return packFrame(TYPE_EOS);
}

/** Float32 mono [-1,1] → Int16Array */
export function floatToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const x = Math.max(-1, Math.min(1, float32[i]));
    out[i] = (x * 32767) | 0;
  }
  return out;
}

/** Simple average downsample float32 mono. */
export function downsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return float32;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(float32.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += float32[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}
