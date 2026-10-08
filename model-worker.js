/* ═══════════════════════════════════════════════════════════════
   model-worker.js — ONNX Runtime Web Worker
   
   Soporta LaMa (float32) y MIGAN (uint8) con auto-recuperación:
   
   ⚠ BUG de ORT Web 1.17: inputMetadata.type puede mentir para
   modelos con dims dinámicas + cuantización. El tipo reportado
   (tensor(float)) no siempre coincide con el tipo real (uint8).
   
   SOLUCIÓN: si la primera inferencia falla con "Unexpected input
   data type", reintentamos con el dtype opuesto automáticamente
   y recordamos cuál funciona.
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

// Metadata completa del modelo
let modelInfo = {
  inputImageName: null,
  inputMaskName: null,
  imageChannels: 3,
  imageDtype: 'float32',
  imageDtypeOverride: null,    // dtype REAL que funcionó (si hubo override)
  maskDtype: 'float32',
  hasSeparateMask: false,
  imageFixedSize: null,
  maskMeaning: 'eliminate',
  type: 'unknown'
};

const PAD_USER = 25;
const DEFAULT_LAMA_SIZE = 512;

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

/* Detectar si un error es de "tipo de dato incorrecto" */
function isDtypeError(errMsg) {
  if (!errMsg) return false;
  var m = errMsg.toLowerCase();
  return m.indexOf('unexpected input data type') >= 0;
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
   Detección del modelo
   ═══════════════════════════════════════════════════════════════ */
function detectModelType(session) {
  modelInfo = {
    inputImageName: null,
    inputMaskName: null,
    imageChannels: 3,
    imageDtype: 'float32',
    imageDtypeOverride: null,
    maskDtype: 'float32',
    hasSeparateMask: false,
    imageFixedSize: null,
    maskMeaning: 'eliminate',
    type: 'unknown'
  };

  var inputs = session.inputNames;
  var meta = session.inputMetadata || {};

  console.log('[worker] ═══ Inspeccionando inputs ═══');

  var fixedH = 0, fixedW = 0;

  for (var i = 0; i < inputs.length; i++) {
    var name = inputs[i];
    var info = meta[name] || {};
    var dims = info.dimensions ? info.dimensions.slice() : null;
    var dtype = info.type || 'tensor(float)';

    console.log('[worker]   input[' + i + '] "' + name + '" dims=' +
                JSON.stringify(dims) + ' type=' + dtype);

    if (!dims || dims.length !== 4) continue;

    if (typeof dims[2] === 'number' && dims[2] > 0 &&
        typeof dims[3] === 'number' && dims[3] > 0) {
      fixedH = dims[2];
      fixedW = dims[3];
    }

    var ch = dims[1];
    if ((ch === 3 || ch === 4) && !modelInfo.inputImageName) {
      modelInfo.inputImageName = name;
      modelInfo.imageChannels = ch;
      modelInfo.imageDtype = parseDtype(dtype);
    } else if (ch === 1 && !modelInfo.inputMaskName) {
      modelInfo.inputMaskName = name;
      modelInfo.maskDtype = parseDtype(dtype);
      modelInfo.hasSeparateMask = true;
    }
  }

  // Fallback por nombre
  if (!modelInfo.inputImageName) {
    for (var j = 0; j < inputs.length; j++) {
      var ln = inputs[j].toLowerCase();
      if (ln.indexOf('image') >= 0 || ln.indexOf('img') >= 0) {
        var info2 = meta[inputs[j]] || {};
        modelInfo.inputImageName = inputs[j];
        modelInfo.imageDtype = parseDtype(info2.type || 'tensor(float)');
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

  // Tipo de modelo según dtype reportado
  if (modelInfo.imageDtype === 'uint8') {
    modelInfo.type = 'migan';
    modelInfo.maskMeaning = 'keep';
  } else {
    modelInfo.type = 'lama';
    modelInfo.maskMeaning = 'eliminate';
  }

  // Tamaño fijo
  if (fixedH > 0 && fixedW > 0) {
    modelInfo.imageFixedSize = { h: fixedH, w: fixedW };
  } else if (modelInfo.type === 'lama') {
    modelInfo.imageFixedSize = { h: DEFAULT_LAMA_SIZE, w: DEFAULT_LAMA_SIZE };
  } else {
    modelInfo.imageFixedSize = null;
  }

  console.log('[worker] ═══ Detección final ═══');
  console.log('[worker]   Tipo:', modelInfo.type);
  console.log('[worker]   image:', modelInfo.inputImageName,
              '· ch=' + modelInfo.imageChannels,
              '· dtype=' + modelInfo.imageDtype);
  console.log('[worker]   mask:', modelInfo.inputMaskName,
              '· dtype=' + modelInfo.maskDtype,
              '· hasSep=' + modelInfo.hasSeparateMask);
  console.log('[worker]   fixed:', JSON.stringify(modelInfo.imageFixedSize));
  console.log('[worker]   maskMeaning:', modelInfo.maskMeaning);
}

function parseDtype(ortType) {
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
   Construcción de tensores (con dtype forzable)
   ═══════════════════════════════════════════════════════════════ */
function buildImageTensorFromRGBA(imgPx, maskPx, width, height, forceDtype) {
  var size = width * height;
  var dtype = forceDtype || modelInfo.imageDtype;
  var channels = modelInfo.imageChannels;

  if (dtype === 'uint8') {
    var imgData = new Uint8Array(channels * size);
    var maskData = new Uint8Array(size);

    for (var i = 0; i < size; i++) {
      var i4 = i * 4;
      imgData[i]            = imgPx[i4];
      imgData[size + i]     = imgPx[i4 + 1];
      imgData[2 * size + i] = imgPx[i4 + 2];
      // MIGAN: invertir (blanco del usuario = 0 = eliminar)
      maskData[i] = maskPx[i4] > 127 ? 0 : 255;
    }
    return {
      image: new ort.Tensor('uint8', imgData, [1, channels, height, width]),
      mask: new ort.Tensor('uint8', maskData, [1, 1, height, width])
    };
  }

  // float32
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
   Geometría del crop
   ═══════════════════════════════════════════════════════════════ */
function computeCropGeometry(bbox, imgW, imgH, fixedSize) {
  var cropW, cropH;

  if (fixedSize) {
    cropW = fixedSize.w;
    cropH = fixedSize.h;
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

  return { x: cropX, y: cropY, w: cropW, h: cropH };
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
   ⭐ Inferencia con auto-recuperación de dtype
   
   1) Intenta con el dtype reportado por metadata.
   2) Si falla con "Unexpected input data type", reintenta con el
      dtype opuesto y recuerda cuál funcionó para próximas veces.
   ═══════════════════════════════════════════════════════════════ */
async function runInferenceWithFallback(geom, cropImg, cropMask) {
  // Decidir orden de intentos
  var startDtype = modelInfo.imageDtypeOverride || modelInfo.imageDtype;
  var altDtype = (startDtype === 'uint8') ? 'float32' : 'uint8';

  var tryOrder = modelInfo.imageDtypeOverride
    ? [modelInfo.imageDtypeOverride]
    : [startDtype, altDtype];

  var lastErr = null;

  for (var i = 0; i < tryOrder.length; i++) {
    var dt = tryOrder[i];
    console.log('[worker] Inferencia con dtype=' + dt +
                ' (metadata=' + modelInfo.imageDtype + ')');

    var tensors = buildImageTensorFromRGBA(cropImg, cropMask, geom.w, geom.h, dt);

    var feeds = {};
    feeds[modelInfo.inputImageName] = tensors.image;
    if (modelInfo.inputMaskName && tensors.mask) {
      feeds[modelInfo.inputMaskName] = tensors.mask;
    }

    try {
      var t0 = Date.now();
      var results = await currentSession.run(feeds);
      var dtms = ((Date.now() - t0) / 1000).toFixed(2);
      console.log('[worker] ✓ Inferencia OK con dtype=' + dt + ' en ' + dtms + 's');

      // Guardar override si el dtype usado difiere del metadata
      if (!modelInfo.imageDtypeOverride && dt !== modelInfo.imageDtype) {
        modelInfo.imageDtypeOverride = dt;
        console.log('[worker] ⚠ Override de dtype guardado: ' + dt +
                    ' (metadata reportaba ' + modelInfo.imageDtype + ')');
      }

      return results;
    } catch (err) {
      var msg = serializeError(err);
      lastErr = msg;

      if (isDtypeError(msg) && i < tryOrder.length - 1) {
        console.warn('[worker] Fallo por dtype. Reintentando con ' +
                     tryOrder[i + 1] + '...');
        continue;
      }

      // Si no es error de dtype o ya no quedan intentos, propagar
      throw new Error(msg);
    }
  }

  throw new Error('Todos los intentos fallaron. Último: ' + lastErr);
}

/* ═══════════════════════════════════════════════════════════════
   Inferencia sobre una región
   ═══════════════════════════════════════════════════════════════ */
async function runOnRegion(imageData, maskFull, imgW, imgH, bbox) {
  var fixedSize = modelInfo.imageFixedSize;
  var geom = computeCropGeometry(bbox, imgW, imgH, fixedSize);

  console.log('[worker] Crop=' + geom.w + '×' + geom.h +
              ' en (' + geom.x + ',' + geom.y + ')' +
              ' · tipo=' + modelInfo.type);

  var cropImg = extractCrop(imageData, imgW, imgH, geom);
  var cropMask = extractMaskCrop(maskFull, imgW, imgH, geom);

  var results = await runInferenceWithFallback(geom, cropImg, cropMask);

  cropImg = null;
  cropMask = null;

  // Convertir salida
  var outKey = currentSession.outputNames[0] || Object.keys(results)[0];
  var outTensor = results[outKey];
  var outDims = outTensor.dims;
  var outH = outDims[outDims.length - 2];
  var outW = outDims[outDims.length - 1];

  var resultImg = tensorToImageData(outTensor, outW, outH);

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