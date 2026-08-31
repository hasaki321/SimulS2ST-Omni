/**
 * AudioWorkletProcessor: forward mono input frames to the main thread.
 * Replaces deprecated ScriptProcessorNode.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Copy — worklet buffers are reused across callbacks.
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
