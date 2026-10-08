/* ═══════════════════════════════════════════════════════════════
   model-worker.js — ONNX Runtime Web Worker para LAMA Inpainting
   Compatible con FP16 e INT8 · reporta bbox real
   ═══════════════════════════════════════════════════════════════ */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

const WORKER_IS_ANDROID = /Android/i.test(navigator.userAgent);
let forceWasm = WORKER_IS_ANDROID;
let useWebGPU = false;
let currentSession = null;
let currentModelId = null;

let inputImageName = null;
let inputMaskName = null;
let inputImageDims = null;
let inputMaskDims = null;

let MODEL_SIZE = 512;
const PAD_USER = 25;

console.log('[worker] Iniciado. Android=' + WORKER_IS_ANDROID);

function serializeError(err) {
  if (err === null) return 'Error: null';
  if (err === undefined) return 'Error: undefined';
  if (typeof err === 'string') return err;
  if (typeof err === 'number') return 'Error code: ' + err;
  if (err instanceof Error) {
    var msg = err.message || err.name || 'Error';
    if (err.stack) {
      var firstLine = err.stack.split('\n')[0];
      if (firstLine && firstLine.indexOf(msg) < 0) msg += ' | ' + firstLine;
    }
    return msg;
  }
  try {
    var keys = Object.keys(err);
    if (keys.length > 0) {
      var parts = [];
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var v = err[k];
        if (v instanceof Error) v = v.message;
        else if (typeof v === 'object' && v !== null) {
          try { v = JSON.stringify(v); } catch (_) { v = '[obj]'; }
        }
        parts.push(k + '=' + v);
      }
      return parts.join(', ');
    }
  } catch (_) {}
  try {
    var s = String(err);
    if (s && s !== '[object Object]') return s;
  } catch (_) {}
  return 'Error no serializable (typeof=' + typeof err + ')';
}

self.addEventListener('message', function (e) {
  if (!e.data || !e.data.type) return;
  if (e.data.type === 'enable-webgpu') {
    if (forceWasm || WORKER_IS_ANDROID) return;
    useWebGPU = true;
  }
  if (e.data.type === 'force-wasm') {
    forceWasm = true;
    useWebGPU = false;
  }
});

function bboxFromMask(maskData, width, height) {
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      if (maskData[idx] > 127) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function findRegions(maskData, width, height) {
  const visited = new Uint8Array(width * height);
  const regions = [];
  const stack = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (visited[idx]) continue;
      if (maskData[idx * 4] <= 127) { visited[idx] = 1; continue; }

      stack.length = 0;
      stack.push(idx);
      visited[idx] = 1;
      let minX = x, maxX = x, minY = y, maxY = y, count = 0;

      while (stack.length) {
        const p = stack.pop();
        const py = Math.floor(p / width);
        const px = p % width;
        count++;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;

        const nbs = [p - 1, p + 1, p - width, p + width];
        for (let k = 0; k < nbs.length; k++) {
          const n = nbs[k];
          if (n < 0 || n >= width * height || visited[n]) continue;
          const npy = Math.floor(n / width);
          const npx = n % width;
          if (Math.abs(npy - py) + Math.abs(npx - px) !== 1) continue;
          visited[n] = 1;
          if (maskData[n * 4] > 127) stack.push(n);
        }
      }
      regions.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area: count });
    }
  }
  return regions;
}

function detectInputInfo(session) {
  const inputs = session.inputNames;
  inputImageName = null;
  inputMaskName = null;
  inputImageDims = null;
  inputMaskDims = null;

  const meta = session.inputMetadata || {};

  for (let i = 0; i < inputs.length; i++) {
    const name = inputs[i];
    const info = meta[name];
    const dims = info && info.dimensions ? info.dimensions.slice() : null;

    if (dims && dims.length === 4) {
      const ch = dims[1];
      if ((ch === 4 || ch === 3) && !inputImageName) {
        inputImageName = name;
        inputImageDims = dims;
      } else if (ch === 1 && !inputMaskName) {
        inputMaskName = name;
        inputMaskDims = dims;
      }
    }
  }

  if (!inputImageName && inputs.length >= 1) {
    inputImageName = inputs[0];
    inputImageDims = meta[inputs[0]] ? meta[inputs[0]].dimensions : null;
  }
  if (!inputMaskName && inputs.length >= 2) {
    inputMaskName = inputs[1];
    inputMaskDims = meta[inputs[1]] ? meta[inputs[1]].dimensions : null;
  }
}

function getModelSize() {
  let h = -1, w = -1;
  if (inputImageDims && inputImageDims.length === 4) {
    h = inputImageDims[2];
    w = inputImageDims[3];
  }
  if ((h < 0 || w < 0) && inputMaskDims && inputMaskDims.length === 4) {
    h = inputMaskDims[2];
    w = inputMaskDims[3];
  }
  if (h < 0 || w < 0) { h = 512; w = 512; }
  return { h: h, w: w, fixed: (h > 0 && w > 0) };
}

function buildInputTensor(imgData, maskData, width, height) {
  const size = width * height;
  const float = new Float32Array(4 * size);

  const imgPx = imgData.data;
  const maskPx = maskData.data;

  for (let i = 0; i < size; i++) {
    const i4 = i * 4;
    const m = maskPx[i4] > 127 ? 1 : 0;

    if (m === 1) {
      float[i]            = 0;
      float[size + i]     = 0;
      float[2 * size + i] = 0;
    } else {
      float[i]            = imgPx[i4]     / 255;
      float[size + i]     = imgPx[i4 + 1] / 255;
      float[2 * size + i] = imgPx[i4 + 2] / 255;
    }
    float[3 * size + i] = m;
  }

  return new ort.Tensor('float32', float, [1, 4, height, width]);
}

function tensorToImageData(tensor, width, height) {
  const out = new Uint8ClampedArray(width * height * 4);
  const data = tensor.data;
  const dims = tensor.dims;
  const size = width * height;

  const channels = dims.length >= 2 ? dims[1] : 3;

  let rOff, gOff, bOff;
  if (channels >= 3) {
    rOff = 0;
    gOff = size;
    bOff = 2 * size;
  } else {
    rOff = 0;
    gOff = 0;
    bOff = 0;
  }

  let maxVal = 0;
  const sampleCount = Math.min(size, 5000);
  for (let i = 0; i < sampleCount; i++) {
    const v = Math.abs(data[i]);
    if (v > maxVal) maxVal = v;
  }
  const scale = maxVal > 1.5 ? 1 : 255;

  for (let i = 0; i < size; i++) {
    let r = data[rOff + i] * scale;
    let g = data[gOff + i] * scale;
    let b = data[bOff + i] * scale;

    if (r < 0) r = 0; if (r > 255) r = 255;
    if (g < 0) g = 0; if (g > 255) g = 255;
    if (b < 0) b = 0; if (b > 255) b = 255;

    out[i * 4]     = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }

  return new ImageData(out, width, height);
}

function extractPaddedPatch(imageData, maskFull, imgW, imgH, bbox, modelSize) {
  const userCx = bbox.x + bbox.w / 2;
  const userCy = bbox.y + bbox.h / 2;

  let squareSide = Math.max(bbox.w, bbox.h) + PAD_USER * 2;
  if (squareSide < 64) squareSide = 64;
  if (squareSide > modelSize) squareSide = modelSize;

  let cropX = Math.round(userCx - squareSide / 2);
  let cropY = Math.round(userCy - squareSide / 2);

  const cropW = squareSide;
  const cropH = squareSide;

  const offsetX = Math.floor((modelSize - cropW) / 2);
  const offsetY = Math.floor((modelSize - cropH) / 2);

  const patchImg = new Uint8ClampedArray(modelSize * modelSize * 4);
  const patchMask = new Uint8ClampedArray(modelSize * modelSize * 4);

  const srcData = imageData.data;
  const srcMask = maskFull;

  for (let j = 0; j < modelSize; j++) {
    for (let i = 0; i < modelSize; i++) {
      const dstIdx = (j * modelSize + i) * 4;

      const ci = i - offsetX;
      const cj = j - offsetY;

      if (ci >= 0 && ci < cropW && cj >= 0 && cj < cropH) {
        let sx = cropX + ci;
        let sy = cropY + cj;

        if (sx < 0) sx = -sx;
        if (sx >= imgW) sx = 2 * imgW - 2 - sx;
        if (sy < 0) sy = -sy;
        if (sy >= imgH) sy = 2 * imgH - 2 - sy;

        sx = Math.max(0, Math.min(imgW - 1, sx));
        sy = Math.max(0, Math.min(imgH - 1, sy));

        const srcIdx = (sy * imgW + sx) * 4;
        patchImg[dstIdx]     = srcData[srcIdx];
        patchImg[dstIdx + 1] = srcData[srcIdx + 1];
        patchImg[dstIdx + 2] = srcData[srcIdx + 2];
        patchImg[dstIdx + 3] = 255;

        patchMask[dstIdx]     = srcMask[srcIdx];
        patchMask[dstIdx + 1] = srcMask[srcIdx + 1];
        patchMask[dstIdx + 2] = srcMask[srcIdx + 2];
        patchMask[dstIdx + 3] = 255;
      } else {
        let ciClamp = Math.max(0, Math.min(cropW - 1, ci));
        let cjClamp = Math.max(0, Math.min(cropH - 1, cj));

        let sx = cropX + ciClamp;
        let sy = cropY + cjClamp;

        if (sx < 0) sx = -sx;
        if (sx >= imgW) sx = 2 * imgW - 2 - sx;
        if (sy < 0) sy = -sy;
        if (sy >= imgH) sy = 2 * imgH - 2 - sy;

        sx = Math.max(0, Math.min(imgW - 1, sx));
        sy = Math.max(0, Math.min(imgH - 1, sy));

        const srcIdx = (sy * imgW + sx) * 4;
        patchImg[dstIdx]     = srcData[srcIdx];
        patchImg[dstIdx + 1] = srcData[srcIdx + 1];
        patchImg[dstIdx + 2] = srcData[srcIdx + 2];
        patchImg[dstIdx + 3] = 255;

        patchMask[dstIdx]     = 0;
        patchMask[dstIdx + 1] = 0;
        patchMask[dstIdx + 2] = 0;
        patchMask[dstIdx + 3] = 255;
      }
    }
  }

  return {
    patchImg: new ImageData(patchImg, modelSize, modelSize),
    patchMask: new ImageData(patchMask, modelSize, modelSize),
    cropX: cropX,
    cropY: cropY,
    cropW: cropW,
    cropH: cropH,
    offsetX: offsetX,
    offsetY: offsetY
  };
}

async function loadSession(modelArrayBuffer, modelId) {
  if (currentSession) {
    try { await currentSession.release(); } catch (_) {}
    currentSession = null;
  }

  let buf = modelArrayBuffer;
  if (buf && buf.buffer && !(buf instanceof ArrayBuffer)) {
    buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  if (!buf || !(buf instanceof ArrayBuffer)) {
    throw new Error('Buffer inválido. Tipo: ' + (typeof buf));
  }

  if (buf.byteLength < 1024) {
    throw new Error('Buffer demasiado pequeño: ' + buf.byteLength + ' bytes');
  }

  const header = new Uint8Array(buf, 0, Math.min(16, buf.byteLength));
  const headerHex = Array.from(header).map(function (b) {
    return ('0' + b.toString(16)).slice(-2);
  }).join(' ');

  if (header[0] === 0x3C) {
    throw new Error('El buffer es HTML, no un ONNX. Header: ' + headerHex);
  }

  const providers = [];
  if (useWebGPU && !forceWasm) providers.push('webgpu');
  providers.push('wasm');

  let session = null;
  let usedProvider = null;
  let lastErr = null;

  for (let i = 0; i < providers.length; i++) {
    const provider = providers[i];
    try {
      const t0 = Date.now();
      session = await ort.InferenceSession.create(buf, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all'
      });
      const dt = ((Date.now() - t0) / 1000).toFixed(2);
      usedProvider = provider;
      console.log('[worker] Provider ' + provider + ' OK en ' + dt + 's');
      break;
    } catch (err) {
      const msg = serializeError(err);
      console.warn('[worker] Provider ' + provider + ' falló: ' + msg);
      lastErr = msg;
    }
  }

  if (!session) {
    throw new Error('No se pudo inicializar el modelo. Último error: ' +
                    (lastErr || 'desconocido'));
  }

  currentSession = session;
  currentModelId = modelId;

  detectInputInfo(session);

  const reqSize = getModelSize();
  MODEL_SIZE = reqSize.w;

  return {
    provider: usedProvider,
    inputSize: reqSize,
    inputs: {
      image: inputImageName,
      mask: inputMaskName,
      names: session.inputNames.slice()
    }
  };
}

async function runOnRegion(imageData, maskFull, imgW, imgH, bbox) {
  const modelSize = MODEL_SIZE;
  const patch = extractPaddedPatch(imageData, maskFull, imgW, imgH, bbox, modelSize);

  const inputTensor = buildInputTensor(patch.patchImg, patch.patchMask, modelSize, modelSize);

  const inputNames = currentSession.inputNames;
  const feeds = {};

  if (inputNames.length === 1) {
    feeds[inputNames[0]] = inputTensor;
  } else {
    const size = modelSize * modelSize;
    const imgFloat = new Float32Array(3 * size);
    const maskFloat = new Float32Array(size);
    const src = inputTensor.data;
    for (let i = 0; i < size; i++) {
      imgFloat[i]            = src[i];
      imgFloat[size + i]     = src[size + i];
      imgFloat[2 * size + i] = src[2 * size + i];
      maskFloat[i]           = src[3 * size + i];
    }
    const imgT = new ort.Tensor('float32', imgFloat, [1, 3, modelSize, modelSize]);
    const maskT = new ort.Tensor('float32', maskFloat, [1, 1, modelSize, modelSize]);
    feeds[inputNames[0]] = imgT;
    feeds[inputNames[1]] = maskT;
  }

  const results = await currentSession.run(feeds);
  const outKey = currentSession.outputNames[0] || Object.keys(results)[0];
  const outTensor = results[outKey];

  const outDims = outTensor.dims;
  const outH = outDims[outDims.length - 2];
  const outW = outDims[outDims.length - 1];
  const resultFull = tensorToImageData(outTensor, outW, outH);

  const resultCrop = new Uint8ClampedArray(patch.cropW * patch.cropH * 4);

  for (let j = 0; j < patch.cropH; j++) {
    for (let i = 0; i < patch.cropW; i++) {
      const srcX = patch.offsetX + i;
      const srcY = patch.offsetY + j;

      const safeX = Math.max(0, Math.min(outW - 1, srcX));
      const safeY = Math.max(0, Math.min(outH - 1, srcY));

      const srcIdx = (safeY * outW + safeX) * 4;
      const dstIdx = (j * patch.cropW + i) * 4;

      resultCrop[dstIdx]     = resultFull.data[srcIdx];
      resultCrop[dstIdx + 1] = resultFull.data[srcIdx + 1];
      resultCrop[dstIdx + 2] = resultFull.data[srcIdx + 2];
      resultCrop[dstIdx + 3] = 255;
    }
  }

  return {
    imageData: new ImageData(resultCrop, patch.cropW, patch.cropH),
    pasteX: Math.max(0, patch.cropX),
    pasteY: Math.max(0, patch.cropY),
    pasteW: patch.cropW,
    pasteH: patch.cropH,
    cropOutX: Math.max(0, -patch.cropX),
    cropOutY: Math.max(0, -patch.cropY)
  };
}

function pasteResult(fullImgData, resultData, imgW, imgH, pasteX, pasteY, pasteW, pasteH, cropOutX, cropOutY) {
  const srcOffsetX = cropOutX || 0;
  const srcOffsetY = cropOutY || 0;

  const destW = Math.min(pasteW - srcOffsetX, imgW - pasteX);
  const destH = Math.min(pasteH - srcOffsetY, imgH - pasteY);

  if (destW <= 0 || destH <= 0) return;

  for (let j = 0; j < destH; j++) {
    for (let i = 0; i < destW; i++) {
      const dstIdx = ((pasteY + j) * imgW + (pasteX + i)) * 4;
      const srcIdx = ((srcOffsetY + j) * pasteW + (srcOffsetX + i)) * 4;

      fullImgData.data[dstIdx]     = resultData.data[srcIdx];
      fullImgData.data[dstIdx + 1] = resultData.data[srcIdx + 1];
      fullImgData.data[dstIdx + 2] = resultData.data[srcIdx + 2];
      fullImgData.data[dstIdx + 3] = 255;
    }
  }
}

async function runInference(imageData, maskData, width, height) {
  if (!currentSession) throw new Error('Modelo no cargado');

  const bbox = bboxFromMask(maskData.data, width, height);
  if (!bbox) return { imageData: imageData, strategy: 'empty' };

  const regions = findRegions(maskData.data, width, height);
  let totalMaskPx = 0;
  for (let i = 0; i < regions.length; i++) totalMaskPx += regions[i].area;
  const maskRatio = totalMaskPx / (width * height);

  let strategy = '';
  let processWhole = false;

  if (regions.length === 1) {
    strategy = 'single-crop';
  } else if (regions.length > 1 && maskRatio < 0.15) {
    strategy = 'multi-crop';
  } else {
    strategy = 'whole-image';
    processWhole = true;
  }

  self.postMessage({
    type: 'progress',
    stage: 'inference',
    total: regions.length,
    bboxW: bbox.w,
    bboxH: bbox.h
  });

  const fullResult = new ImageData(new Uint8ClampedArray(imageData.data), width, height);

  if (processWhole) {
    const out = await runOnRegion(imageData, maskData.data, width, height, bbox);
    pasteResult(fullResult, out.imageData, width, height,
                out.pasteX, out.pasteY, out.pasteW, out.pasteH,
                out.cropOutX, out.cropOutY);
    return { imageData: fullResult, strategy: strategy };
  }

  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];

    self.postMessage({
      type: 'progress',
      stage: 'region',
      current: i + 1,
      total: regions.length,
      bboxW: r.w,
      bboxH: r.h
    });

    const out = await runOnRegion(imageData, maskData.data, width, height, r);
    pasteResult(fullResult, out.imageData, width, height,
                out.pasteX, out.pasteY, out.pasteW, out.pasteH,
                out.cropOutX, out.cropOutY);
  }

  return { imageData: fullResult, strategy: strategy };
}

self.onmessage = async function (e) {
  const msg = e.data;
  if (!msg || !msg.action) return;

  const id = msg.id;
  const action = msg.action;
  const payload = msg.payload || {};

  try {
    if (action === 'load') {
      const info = await loadSession(payload.buffer, payload.modelId);
      self.postMessage({ id: id, type: 'result', result: info });
    }
    else if (action === 'infer') {
      const out = await runInference(
        payload.imageData, payload.maskData, payload.width, payload.height
      );
      self.postMessage({
        id: id,
        type: 'result',
        result: {
          imageData: out.imageData,
          strategy: out.strategy
        }
      });
    }
    else if (action === 'unload') {
      if (currentSession) {
        try { await currentSession.release(); } catch (_) {}
        currentSession = null;
        currentModelId = null;
      }
      self.postMessage({ id: id, type: 'result', result: { unloaded: true } });
    }
    else {
      throw new Error('Acción desconocida: ' + action);
    }
  } catch (err) {
    const errMsg = serializeError(err);
    console.error('[worker] Error capturado:', errMsg);
    self.postMessage({
      id: id,
      type: 'error',
      error: errMsg
    });
  }
};