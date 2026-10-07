/**
 * AudioWorklet processors (run on the real-time audio thread).
 *
 * capture-processor:  mic Float32 @ device rate -> 16 kHz mono PCM16 frames (20 ms)
 *                     + RMS level for the UI meter.
 * playback-processor: queue of PCM16 @ 24 kHz -> speakers, with instant "clear"
 *                     for barge-in and a "drained" notification when playback ends.
 */

const INT16_MAX = 32767;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { targetRate = 16000, frameMs = 20 } = options.processorOptions || {};
    this.ratio = sampleRate / targetRate; // e.g. 48000 / 16000 = 3
    this.frameSamples = Math.round((targetRate * frameMs) / 1000);
    this.pending = [];       // un-decimated input samples carried between blocks
    this.pos = 0;            // fractional read position into `pending`
    this.out = new Int16Array(this.frameSamples);
    this.outIndex = 0;
    this.levelAcc = 0;
    this.levelCount = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) this.pending.push(channel[i]);

    // Decimate with a box filter over each output window (cheap anti-aliasing).
    while (this.pos + this.ratio <= this.pending.length) {
      const start = Math.floor(this.pos);
      const end = Math.floor(this.pos + this.ratio);
      let sum = 0;
      for (let j = start; j < end; j++) sum += this.pending[j];
      const sample = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
      this.out[this.outIndex++] = Math.round(sample * INT16_MAX);
      this.levelAcc += sample * sample;
      this.levelCount++;
      this.pos += this.ratio;
      if (this.outIndex === this.frameSamples) {
        const frame = this.out.slice(0);
        this.port.postMessage({ type: "frame", buffer: frame.buffer }, [frame.buffer]);
        this.port.postMessage({ type: "level", rms: Math.sqrt(this.levelAcc / this.levelCount) });
        this.outIndex = 0;
        this.levelAcc = 0;
        this.levelCount = 0;
      }
    }
    const consumed = Math.floor(this.pos);
    if (consumed > 0) {
      this.pending.splice(0, consumed);
      this.pos -= consumed;
    }
    return true;
  }
}

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];   // Float32Array chunks
    this.offset = 0;   // read offset into queue[0]
    this.playing = false;
    this.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "chunk") {
        const pcm = new Int16Array(msg.buffer);
        const f32 = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
        this.queue.push(f32);
      } else if (msg.type === "clear") {
        this.queue = [];
        this.offset = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let written = 0;
    while (written < out.length && this.queue.length > 0) {
      const head = this.queue[0];
      const n = Math.min(out.length - written, head.length - this.offset);
      out.set(head.subarray(this.offset, this.offset + n), written);
      written += n;
      this.offset += n;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (written < out.length) out.fill(0, written);

    const nowPlaying = written > 0;
    if (nowPlaying !== this.playing) {
      this.playing = nowPlaying;
      this.port.postMessage({ type: nowPlaying ? "started" : "drained" });
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
