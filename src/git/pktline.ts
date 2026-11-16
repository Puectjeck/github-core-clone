// The pkt-line framing every git smart-HTTP message uses: a 4-hex-digit length (counting the
// length bytes themselves) before each payload, with three zero-length control packets —
// 0000 flush, 0001 delimiter, 0002 response-end.

export type Packet =
  | { readonly kind: 'data'; readonly payload: Buffer }
  | { readonly kind: 'flush' }
  | { readonly kind: 'delim' }
  | { readonly kind: 'response-end' };

const HEX = '0123456789abcdef';

export function pktLine(payload: Buffer | string): Buffer {
  const body = typeof payload === 'string' ? Buffer.from(payload, 'utf8') : payload;
  const length = body.length + 4;
  if (length > 0xffff) throw new Error(`pkt-line payload too large: ${body.length}`);
  // One allocation and one copy: the framing is on every git response's path, and the old
  // Buffer.from(header) + Buffer.concat pair allocated and copied twice per line.
  const out = Buffer.allocUnsafe(length);
  out[0] = HEX.charCodeAt((length >> 12) & 0xf);
  out[1] = HEX.charCodeAt((length >> 8) & 0xf);
  out[2] = HEX.charCodeAt((length >> 4) & 0xf);
  out[3] = HEX.charCodeAt(length & 0xf);
  body.copy(out, 4);
  return out;
}

export const FLUSH = Buffer.from('0000', 'ascii');
export const DELIM = Buffer.from('0001', 'ascii');
export const RESPONSE_END = Buffer.from('0002', 'ascii');

/** The value of a lowercase hex digit byte, or -1 (the frame alphabet is lowercase only). */
function hexDigit(byte: number): number {
  if (byte >= 48 && byte <= 57) return byte - 48; // 0-9
  if (byte >= 97 && byte <= 102) return byte - 87; // a-f
  return -1;
}

/** Parse a complete buffer into packets; throws on a truncated or malformed frame. */
export function parsePackets(buffer: Buffer): Packet[] {
  const packets: Packet[] = [];
  let at = 0;
  while (at < buffer.length) {
    if (at + 4 > buffer.length) throw new Error('pkt-line stream ends mid-header');
    // Decode the length from the raw bytes: the old form stringified every header and ran a
    // regex + parseInt over it, an allocation and two passes per packet on the fetch/push path.
    const h0 = hexDigit(buffer[at] as number);
    const h1 = hexDigit(buffer[at + 1] as number);
    const h2 = hexDigit(buffer[at + 2] as number);
    const h3 = hexDigit(buffer[at + 3] as number);
    if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
      const head = buffer.toString('ascii', at, at + 4);
      throw new Error(`pkt-line header is not hex: ${JSON.stringify(head)}`);
    }
    const length = (h0 << 12) | (h1 << 8) | (h2 << 4) | h3;
    if (length === 0) {
      packets.push({ kind: 'flush' });
      at += 4;
    } else if (length === 1) {
      packets.push({ kind: 'delim' });
      at += 4;
    } else if (length === 2) {
      packets.push({ kind: 'response-end' });
      at += 4;
    } else if (length < 4) {
      throw new Error(`pkt-line length ${length} is invalid`);
    } else {
      if (at + length > buffer.length) throw new Error('pkt-line stream ends mid-packet');
      packets.push({ kind: 'data', payload: buffer.subarray(at + 4, at + length) });
      at += length;
    }
  }
  return packets;
}

/** Side-band-64k framing: each pkt-line payload starts with a one-byte band (1 pack, 2 progress, 3 error). */
export function sidebandPacket(band: 1 | 2 | 3, data: Buffer): Buffer {
  // Framed in one allocation: the band byte and the data are written into the pkt-line buffer
  // directly instead of concatenating a one-byte buffer and copying the result again.
  const length = data.length + 5;
  if (length > 0xffff) throw new Error(`pkt-line payload too large: ${data.length + 1}`);
  const out = Buffer.allocUnsafe(length);
  out[0] = HEX.charCodeAt((length >> 12) & 0xf);
  out[1] = HEX.charCodeAt((length >> 8) & 0xf);
  out[2] = HEX.charCodeAt((length >> 4) & 0xf);
  out[3] = HEX.charCodeAt(length & 0xf);
  out[4] = band;
  data.copy(out, 5);
  return out;
}

/** Split pack bytes into band-1 sideband packets of at most `chunk` payload bytes. */
export function sidebandPack(pack: Buffer, chunk = 65515): Buffer[] {
  const out: Buffer[] = [];
  for (let at = 0; at < pack.length; at += chunk) {
    out.push(sidebandPacket(1, pack.subarray(at, Math.min(pack.length, at + chunk))));
  }
  if (out.length === 0) out.push(sidebandPacket(1, Buffer.alloc(0)));
  return out;
}
