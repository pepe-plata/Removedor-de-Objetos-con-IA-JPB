/* ═══════════════════════════════════════════════════════════════
   model-worker.js — ONNX Runtime Web Worker
   
   Soporta DOS modelos leyendo el tipo de dato REAL de cada input:
     - LAMA  → inputs "image" y "mask" en float32 (0..1)
     - MIGAN → inputs "image" y "mask" en uint8 (0..255)
   
   El detector lee session.inputMetadata[name].type para saber si
   construir tensores float32 o uint8. Ya no adivina por el número
   de canales, sino por el tipo real que espera el modelo.
   ═══════════════════════════════════════════════════════════════ */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
ort.env.logLevel = 'error';

const WORKER_IS_ANDROID = /Android/i.test(navigator.userAgent);
let forceWasm = WORKER_IS_ANDROID;
let useWebGPU = false;
let currentSession = null;
let currentModelId = null;

// Metadata completa del modelo (se rellena en detectModelType)
let modelInfo = {
  inputImageName: null,
  inputMaskName: null,
  imageChannels: 3,
  imageDtype: 'float32',      // 'float32' | 'uint8'
  maskDtype: 'float32',       // 'float32' | 'uint8'
  hasSeparateMask: false,
  imageFixedSize: null,       // { h, w } si es fijo, null si dinámico
  maskMeaning: 'eliminate',   // 'eliminate' (LaMa: 1=eliminar) | 'keep' (MIGAN: 255=conservar)
  type: 'unknown'             // 'lama' | 'migan' | 'unknown'
};

const PAD_USER = 25;

function alignTo32(v) {
  return Math.max(32, Math.ceil(v / 32) * 32);
}

console.log('[worker] Iniciado. Android=' + WORKER_IS_ANDROID);

/* ═══════════════════════════════════════════════════════════════
   Serialización de errores
   ═══════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════
   Control
   ═══════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════
   Bounding box y regiones
   ═══════════════════════════════════════════════════════════════ */
function bboxFromMask(maskData, width, height) {
  var minX = width, minY = height, maxX = -1, maxY = -1;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var idx = (y * width + x) * 4;
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
  var visited = new Uint8Array(width * height);
  var regions = [];
  var stack = [];

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      var idx = y * width + x;
      if (visited[idx]) continue;
      if (maskData[idx * 4] <= 127) { visited[idx] = 1; continue; }

      stack.length = 0;
      stack.push(idx);
      visited[idx] = 1;
      var minX = x, maxX = x, minY = y, maxY = y, count = 0;

      while (stack.length) {
        var p = stack.pop();
        var py = Math.floor(p / width);
        var px = p % width;
        count++;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;

        var nbs = [p - 1, p + 1, p - width, p + width];
        for (var k = 0; k < nbs.length; k++) {
          var n = nbs[k];
          if (n < 0 || n >= width * height || visited[n]) continue;
          var npy = Math.floor(n / width);
          var npx = n % width;
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

/* ═══════════════════════════════════════════════════════════════
   Detección completa del modelo: nombres, canales, DTYPE y tamaño.
   ═══════════════════════════════════════════════════════════════ */
function detectModelType(session) {
  modelInfo = {
    inputImageName: null,
    inputMaskName: null,
    imageChannels: 3,
    imageDtype: 'float32',
    maskDtype: 'float32',
    hasSeparateMask: false,
    imageFixedSize: null,
    maskMeaning: 'eliminate',
    type: 'unknown'
  };

  var inputs = session.inputNames;
  var meta = session.inputMetadata || {};

  // 1) Buscar por canales: el de 3 canales es la imagen, el de 1 la máscara
  for (var i = 0; i < inputs.length; i++) {
    var name = inputs[i];
    var info = meta[name] || {};
    var dims = info.dimensions ? info.dimensions.slice() : null;
    var dtype = info.type || 'tensor(float)';

    if (!dims || dims.length !== 4) continue;

    var ch = dims[1];
    if ((ch === 3 || ch === 4) && !modelInfo.inputImageName) {
      modelInfo.inputImageName = name;
      modelInfo.imageChannels = ch;
      modelInfo.imageDtype = parseDtype(dtype);
      if (dims[2] > 0 && dims[3] > 0) {
        modelInfo.imageFixedSize = { h: dims[2], w: dims[3] };
      }
    } else if (ch === 1 && !modelInfo.inputMaskName) {
      modelInfo.inputMaskName = name;
      modelInfo.maskDtype = parseDtype(dtype);
      modelInfo.hasSeparateMask = true;
    }
  }

  // 2) Fallback por nombre
  if (!modelInfo.inputImageName) {
    for (var j = 0; j < inputs.length; j++) {
      var ln = inputs[j].toLowerCase();
      if (ln.indexOf('image') >= 0 || ln.indexOf('img') >= 0) {
        var info2 = meta[inputs[j]] || {};
        var dims2 = info2.dimensions ? info2.dimensions.slice() : null;
        modelInfo.inputImageName = inputs[j];
        modelInfo.imageDtype = parseDtype(info2.type || 'tensor(float)');
        if (dims2 && dims2.length === 4) {
          modelInfo.imageChannels = dims2[1] || 3;
          if (dims2[2] > 0 && dims2[3] > 0) {
            modelInfo.imageFixedSize = { h: dims2[2], w: dims2[3] };
          }
        }
        break;
      }
    }
  }
  if (!modelInfo.inputMaskName) {
    for (var k = 0; k < inputs.length; k++) {
      var ln2 = inputs[k].toLowerCase();
      if (ln2.indexOf('mask') >= 0) {
        var info3 = meta[inputs[k]] || {};
        modelInfo.inputMaskName = inputs[k];
        modelInfo.maskDtype = parseDtype(info3.type || 'tensor(float)');
        modelInfo.hasSeparateMask = true;
        break;
      }
    }
  }

  // 3) Determinar tipo y semántica de la máscara
  //    Regla: si ambos inputs son float32 → LaMa (máscara 1=eliminar)
  //           si ambos inputs son uint8   → MIGAN (máscara 255=conservar)
  if (modelInfo.imageDtype === 'uint8') {
    modelInfo.type = 'migan';
    modelInfo.maskMeaning = 'keep';
  } else {
    modelInfo.type = 'lama';
    modelInfo.maskMeaning = 'eliminate';
  }

  console.log('[worker] Modelo detectado: ' + modelInfo.type);
  console.log('  inputImage:', modelInfo.inputImageName,
              'ch=' + modelInfo.imageChannels,
              'dtype=' + modelInfo.imageDtype,
              'fixed=' + JSON.stringify(modelInfo.imageFixedSize));
  console.log('  inputMask:', modelInfo.inputMaskName,
              'dtype=' + modelInfo.maskDtype,
              'hasSep=' + modelInfo.hasSeparateMask);
  console.log('  maskMeaning:', modelInfo.maskMeaning);
}

function parseDtype(ortType) {
  // ORT devuelve strings como 'tensor(float)', 'tensor(uint8)', 'tensor(float16)'
  if (!ortType) return 'float32';
  var t = ortType.toLowerCase();
  if (t.indexOf('uint8') >= 0) return 'uint8';
  if (t.indexOf('int8') >= 0) return 'int8';
  if (t.indexOf('float16') >= 0) return 'float16';
  if (t.indexOf('float') >= 0) return 'float32';
  return 'float32';
}

/* ═══════════════════════════════════════════════════════════════
   Conversión de salida
   ═══════════════════════════════════════════════════════════════ */
function tensorToImageData(tensor, width, height) {
  var out = new Uint8ClampedArray(width * height * 4);
  var data = tensor.data;
  var dims = tensor.dims;
  var size = width * height;

  var channels = dims.length >= 2 ? dims[1] : 3;
  var rOff, gOff, bOff;
  if (channels >= 3) { rOff = 0; gOff = size; bOff = 2 * size; }
  else { rOff = 0; gOff = 0; bOff = 0; }

  var maxVal = 0;
  var sampleCount = Math.min(size, 5000);
  for (var i = 0; i < sampleCount; i++) {
    var v = Math.abs(data[i]);
    if (v > maxVal) maxVal = v;
  }
  var scale = maxVal > 1.5 ? 1 : 255;

  for (var j = 0; j < size; j++) {
    var r = data[rOff + j] * scale;
    var g = data[gOff + j] * scale;
    var b = data[bOff + j] * scale;
    if (r < 0) r = 0; if (r > 255) r = 255;
    if (g < 0) g = 0; if (g > 255) g = 255;
    if (b < 0) b = 0; if (b > 255) b = 255;
    out[j * 4]     = r;
    out[j * 4 + 1] = g;
    out[j * 4 + 2] = b;
    out[j * 4 + 3] = 255;
  }

  return new ImageData(out, width, height);
}

/* ═══════════════════════════════════════════════════════════════
   Construcción de tensores según dtype esperado
   ═══════════════════════════════════════════════════════════════ */

/** Convierte un Uint8ClampedArray RGBA (imagen) al dtype esperado. */
function buildImageTensorFromRGBA(imgPx, maskPx, width, height) {
  var size = width * height;
  var dtype = modelInfo.imageDtype;
  var channels = modelInfo.imageChannels;
  var maskIsKeep = (modelInfo.maskMeaning === 'keep');

  if (dtype === 'uint8') {
    // ---- MIGAN: uint8 ----
    // La imagen se envía tal cual (0..255).
    // La máscara se envía con su semántica (255=conservar, 0=eliminar).
    var imgData = new Uint8Array(channels * size);
    var maskData = new Uint8Array(size);

    for (var i = 0; i < size; i++) {
      var i4 = i * 4;
      imgData[i]              = imgPx[i4];
      imgData[size + i]       = imgPx[i4 + 1];
      imgData[2 * size + i]   = imgPx[i4 + 2];
      // Nuestra UI: 255 = eliminar. MIGAN: 0 = eliminar.
      // Así que invertimos.
      maskData[i] = maskPx[i4] > 127 ? 0 : 255;
    }

    return {
      image: new ort.Tensor('uint8', imgData, [1, channels, height, width]),
      mask: new ort.Tensor('uint8', maskData, [1, 1, height, width])
    };
  }

  // ---- LaMa y similares: float32 ----
  var f;
  if (channels === 4) {
    f = new Float32Array(4 * size);
    for (var j = 0; j < size; j++) {
      var j4 = j * 4;
      var m = maskPx[j4] > 127 ? 1 : 0;
      if (m === 1) {
        f[j] = 0; f[size + j] = 0; f[2 * size + j] = 0;
      } else {
        f[j]            = imgPx[j4]     / 255;
        f[size + j]     = imgPx[j4 + 1] / 255;
        f[2 * size + j] = imgPx[j4 + 2] / 255;
      }
      f[3 * size + j] = m;
    }
    return {
      image: new ort.Tensor('float32', f, [1, 4, height, width]),
      mask: null
    };
  }

  // channels === 3
  f = new Float32Array(3 * size);
  for (var q = 0; q < size; q++) {
    var q4 = q * 4;
    var mq = maskPx[q4] > 127 ? 1 : 0;
    if (mq === 1) {
      f[q] = 0; f[size + q] = 0; f[2 * size + q] = 0;
    } else {
      f[q]            = imgPx[q4]     / 255;
      f[size + q]     = imgPx[q4 + 1] / 255;
      f[2 * size + q] = imgPx[q4 + 2] / 255;
    }
  }

  // Si el modelo espera máscara separada, también la construimos
  var maskTensor = null;
  if (modelInfo.hasSeparateMask) {
    var mf = new Float32Array(size);
    for (var r = 0; r < size; r++) {
      mf[r] = maskPx[r * 4] > 127 ? 1 : 0;
    }
    maskTensor = new ort.Tensor('float32', mf, [1, 1, height, width]);
  }

  return {
    image: new ort.Tensor('float32', f, [1, 3, height, width]),
    mask: maskTensor
  };
}

/* ═══════════════════════════════════════════════════════════════
   Cálculo de geometría del crop
   ═══════════════════════════════════════════════════════════════ */
function computeCropGeometry(bbox, imgW, imgH, fixedSize) {
  var cropW, cropH;

  if (fixedSize) {
    var side = Math.max(bbox.w, bbox.h) + PAD_USER * 2;
    if (side < 64) side = 64;
    if (side > fixedSize.w) side = fixedSize.w;
    cropW = side;
    cropH = side;
  } else {
    cropW = alignTo32(bbox.w + PAD_USER * 2);
    cropH = alignTo32(bbox.h + PAD_USER * 2);
  }

  var userCx = bbox.x + bbox.w / 2;
  var userCy = bbox.y + bbox.h / 2;

  var cropX = Math.round(userCx - cropW / 2);
  var cropY = Math.round(userCy - cropH / 2);

  if (cropX < 0) cropX = 0;
  if (cropY < 0) cropY = 0;
  if (cropX + cropW > imgW) cropX = Math.max(0, imgW - cropW);
  if (cropY + cropH > imgH) cropY = Math.max(0, imgH - cropH);

  var realW = Math.min(cropW, imgW - cropX);
  var realH = Math.min(cropH, imgH - cropY);

  if (!fixedSize) {
    realW = alignTo32(realW);
    realH = alignTo32(realH);
    if (cropX + realW > imgW) realW = Math.max(32, imgW - cropX);
    if (cropY + realH > imgH) realH = Math.max(32, imgH - cropY);
  }

  return { x: cropX, y: cropY, w: realW, h: realH };
}

/* ═══════════════════════════════════════════════════════════════
   Extracción de crops
   ═══════════════════════════════════════════════════════════════ */
function extractCrop(imageData, imgW, imgH, geom) {
  var cropImg = new Uint8ClampedArray(geom.w * geom.h * 4);
  var src = imageData.data;

  for (var j = 0; j < geom.h; j++) {
    for (var i = 0; i < geom.w; i++) {
      var dstIdx = (j * geom.w + i) * 4;
      var sx = geom.x + i;
      var sy = geom.y + j;

      if (sx < 0) sx = -sx;
      if (sx >= imgW) sx = 2 * imgW - 2 - sx;
      if (sy < 0) sy = -sy;
      if (sy >= imgH) sy = 2 * imgH - 2 - sy;
      sx = Math.max(0, Math.min(imgW - 1, sx));
      sy = Math.max(0, Math.min(imgH - 1, sy));

      var srcIdx = (sy * imgW + sx) * 4;
      cropImg[dstIdx]     = src[srcIdx];
      cropImg[dstIdx + 1] = src[srcIdx + 1];
      cropImg[dstIdx + 2] = src[srcIdx + 2];
      cropImg[dstIdx + 3] = 255;
    }
  }
  return cropImg;
}

function extractMaskCrop(maskFull, imgW, imgH, geom) {
  var cropMask = new Uint8ClampedArray(geom.w * geom.h * 4);
  for (var j = 0; j < geom.h; j++) {
    for (var i = 0; i < geom.w; i++) {
      var dstIdx = (j * geom.w + i) * 4;
      var sx = geom.x + i;
      var sy = geom.y + j;
      if (sx < 0 || sx >= imgW || sy < 0 || sy >= imgH) {
        cropMask[dstIdx] = 0; cropMask[dstIdx + 1] = 0;
        cropMask[dstIdx + 2] = 0; cropMask[dstIdx + 3] = 255;
        continue;
      }
      var srcIdx = (sy * imgW + sx) * 4;
      cropMask[dstIdx]     = maskFull[srcIdx];
      cropMask[dstIdx + 1] = maskFull[srcIdx + 1];
      cropMask[dstIdx + 2] = maskFull[srcIdx + 2];
      cropMask[dstIdx + 3] = 255;
    }
  }
  return cropMask;
}

/* ═══════════════════════════════════════════════════════════════
   Carga de sesión
   ═══════════════════════════════════════════════════════════════ */
async function loadSession(modelArrayBuffer, modelId) {
  if (currentSession) {
    try { await currentSession.release(); } catch (_) {}
    currentSession = null;
  }

  var buf = modelArrayBuffer;
  if (buf && buf.buffer && !(buf instanceof ArrayBuffer)) {
    buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  if (!buf || !(buf instanceof ArrayBuffer)) {
    throw new Error('Buffer inválido. Tipo: ' + (typeof buf));
  }
  if (buf.byteLength < 1024) {
    throw new Error('Buffer demasiado pequeño: ' + buf.byteLength + ' bytes');
  }

  var header = new Uint8Array(buf, 0, Math.min(16, buf.byteLength));
  if (header[0] === 0x3C) throw new Error('El buffer es HTML, no un ONNX.');

  var providers = [];
  if (useWebGPU && !forceWasm) providers.push('webgpu');
  providers.push('wasm');

  var session = null;
  var usedProvider = null;
  var lastErr = null;

  for (var i = 0; i < providers.length; i++) {
    var provider = providers[i];
    try {
      var t0 = Date.now();
      session = await ort.InferenceSession.create(buf, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: !WORKER_IS_ANDROID,
        enableMemPattern: !WORKER_IS_ANDROID
      });
      var dt = ((Date.now() - t0) / 1000).toFixed(2);
      usedProvider = provider;
      console.log('[worker] Provider ' + provider + ' OK en ' + dt + 's');
      break;
    } catch (err) {
      var msg = serializeError(err);
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
  detectModelType(session);

  return {
    provider: usedProvider,
    modelType: modelInfo.type,
    inputs: {
      image: modelInfo.inputImageName,
      mask: modelInfo.inputMaskName,
      channels: modelInfo.imageChannels,
      imageDtype: modelInfo.imageDtype,
      maskDtype: modelInfo.maskDtype,
      fixedSize: modelInfo.imageFixedSize,
      names: session.inputNames.slice()
    }
  };
}

/* ═══════════════════════════════════════════════════════════════
   Inferencia sobre una región
   ═══════════════════════════════════════════════════════════════ */
async function runOnRegion(imageData, maskFull, imgW, imgH, bbox) {
  var fixedSize = modelInfo.imageFixedSize;
  var geom = computeCropGeometry(bbox, imgW, imgH, fixedSize);

  console.log('[worker] Crop=' + geom.w + '×' + geom.h +
              ' en (' + geom.x + ',' + geom.y + ')' +
              ' · tipo=' + modelInfo.type +
              ' · dtype=' + modelInfo.imageDtype);

  var cropImg = extractCrop(imageData, imgW, imgH, geom);
  var cropMask = extractMaskCrop(maskFull, imgW, imgH, geom);

  var tensors = buildImageTensorFromRGBA(cropImg, cropMask, geom.w, geom.h);

  // Construir feeds
  var feeds = {};
  feeds[modelInfo.inputImageName] = tensors.image;
  if (modelInfo.inputMaskName && tensors.mask) {
    feeds[modelInfo.inputMaskName] = tensors.mask;
  }

  cropImg = null;
  cropMask = null;

  var t0 = Date.now();
  var results = await currentSession.run(feeds);
  var dt = ((Date.now() - t0) / 1000).toFixed(2);
  console.log('[worker] Inferencia OK en ' + dt + 's');

  feeds = null;

  // Convertir salida
  var outKey = currentSession.outputNames[0] || Object.keys(results)[0];
  var outTensor = results[outKey];
  var outDims = outTensor.dims;
  var outH = outDims[outDims.length - 2];
  var outW = outDims[outDims.length - 1];

  var resultImg = tensorToImageData(outTensor, outW, outH);

  // Recortar si la salida es mayor que el crop
  var finalW = Math.min(outW, geom.w);
  var finalH = Math.min(outH, geom.h);

  if (finalW !== outW || finalH !== outH) {
    var cropped = new Uint8ClampedArray(finalW * finalH * 4);
    for (var j = 0; j < finalH; j++) {
      for (var i = 0; i < finalW; i++) {
        var srcIdx = (j * outW + i) * 4;
        var dstIdx = (j * finalW + i) * 4;
        cropped[dstIdx]     = resultImg.data[srcIdx];
        cropped[dstIdx + 1] = resultImg.data[srcIdx + 1];
        cropped[dstIdx + 2] = resultImg.data[srcIdx + 2];
        cropped[dstIdx + 3] = 255;
      }
    }
    resultImg = new ImageData(cropped, finalW, finalH);
  }

  return {
    imageData: resultImg,
    pasteX: geom.x,
    pasteY: geom.y,
    pasteW: finalW,
    pasteH: finalH
  };
}

/* ═══════════════════════════════════════════════════════════════
   Pegar resultado
   ═══════════════════════════════════════════════════════════════ */
function pasteResult(fullImgData, resultData, imgW, imgH, pasteX, pasteY, pasteW, pasteH) {
  var destW = Math.min(pasteW, imgW - pasteX);
  var destH = Math.min(pasteH, imgH - pasteY);
  if (destW <= 0 || destH <= 0) return;

  for (var j = 0; j < destH; j++) {
    for (var i = 0; i < destW; i++) {
      var dstIdx = ((pasteY + j) * imgW + (pasteX + i)) * 4;
      var srcIdx = (j * pasteW + i) * 4;
      fullImgData.data[dstIdx]     = resultData.data[srcIdx];
      fullImgData.data[dstIdx + 1] = resultData.data[srcIdx + 1];
      fullImgData.data[dstIdx + 2] = resultData.data[srcIdx + 2];
      fullImgData.data[dstIdx + 3] = 255;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   Estrategia
   ═══════════════════════════════════════════════════════════════ */
async function runInference(imageData, maskData, width, height) {
  if (!currentSession) throw new Error('Modelo no cargado');

  var bbox = bboxFromMask(maskData.data, width, height);
  if (!bbox) return { imageData: imageData, strategy: 'empty' };

  var regions = findRegions(maskData.data, width, height);
  var totalMaskPx = 0;
  for (var i = 0; i < regions.length; i++) totalMaskPx += regions[i].area;
  var maskRatio = totalMaskPx / (width * height);

  var strategy = '';
  var processWhole = false;

  if (regions.length === 1) {
    strategy = 'single-crop';
  } else if (regions.length > 1 && maskRatio < 0.15) {
    strategy = 'multi-crop';
  } else {
    strategy = 'whole-image';
    processWhole = true;
  }

  console.log('[worker] Estrategia=' + strategy + ' | regiones=' + regions.length);

  self.postMessage({
    type: 'progress',
    stage: 'inference',
    total: regions.length,
    bboxW: bbox.w,
    bboxH: bbox.h
  });

  var fullResult = new ImageData(new Uint8ClampedArray(imageData.data), width, height);

  if (processWhole) {
    var out = await runOnRegion(imageData, maskData.data, width, height, bbox);
    pasteResult(fullResult, out.imageData, width, height,
                out.pasteX, out.pasteY, out.pasteW, out.pasteH);
    return { imageData: fullResult, strategy: strategy };
  }

  for (var j = 0; j < regions.length; j++) {
    var r = regions[j];
    self.postMessage({
      type: 'progress',
      stage: 'region',
      current: j + 1,
      total: regions.length,
      bboxW: r.w,
      bboxH: r.h
    });

    var out2 = await runOnRegion(imageData, maskData.data, width, height, r);
    pasteResult(fullResult, out2.imageData, width, height,
                out2.pasteX, out2.pasteY, out2.pasteW, out2.pasteH);
  }

  return { imageData: fullResult, strategy: strategy };
}

/* ═══════════════════════════════════════════════════════════════
   Handler principal
   ═══════════════════════════════════════════════════════════════ */
self.onmessage = async function (e) {
  var msg = e.data;
  if (!msg || !msg.action) return;

  var id = msg.id;
  var action = msg.action;
  var payload = msg.payload || {};

  try {
    if (action === 'load') {
      var info = await loadSession(payload.buffer, payload.modelId);
      self.postMessage({ id: id, type: 'result', result: info });
    }
    else if (action === 'infer') {
      var out = await runInference(
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
    var errMsg = serializeError(err);
    console.error('[worker] Error capturado:', errMsg);
    self.postMessage({ id: id, type: 'error', error: errMsg });
  }
};