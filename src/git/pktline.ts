// The pkt-line framing every git smart-HTTP message uses: a 4-hex-digit length (counting the
// length bytes themselves) before each payload, with three zero-length control packets —
// 0000 flush, 0001 delimiter, 0002 response-end.

export type Packet =
  | { readonly kind: 'data'; readonly payload: Buffer }
  | { readonly kind: 'flush' }
  | { readonly kind: 'delim' }
  | { readonly kind: 'response-end' };

export function pktLine(payload: Buffer | string): Buffer {
  const body = typeof payload === 'string' ? Buffer.from(payload, 'utf8') : payload;
  const length = body.length + 4;
  if (length > 0xffff) throw new Error(`pkt-line payload too large: ${body.length}`);
  const head = Buffer.from(length.toString(16).padStart(4, '0'), 'ascii');
  return Buffer.concat([head, body]);
}

export const FLUSH = Buffer.from('0000', 'ascii');
export const DELIM = Buffer.from('0001', 'ascii');
export const RESPONSE_END = Buffer.from('0002', 'ascii');

/** Parse a complete buffer into packets; throws on a truncated or malformed frame. */
export function parsePackets(buffer: Buffer): Packet[] {
  const packets: Packet[] = [];
  let at = 0;
  while (at < buffer.length) {
    if (at + 4 > buffer.length) throw new Error('pkt-line stream ends mid-header');
    const head = buffer.toString('ascii', at, at + 4);
    if (!/^[0-9a-f]{4}$/.test(head)) throw new Error(`pkt-line header is not hex: ${JSON.stringify(head)}`);
    const length = parseInt(head, 16);
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
  return pktLine(Buffer.concat([Buffer.from([band]), data]));
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
