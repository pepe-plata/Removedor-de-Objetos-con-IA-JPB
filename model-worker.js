/* ═══════════════════════════════════════════════════════════════
   model-worker.js — ONNX Runtime Web Worker
   
   Soporta DOS modelos:
     - LAMA (LaMa-ONNX) → input [1,4,H,W] con RGB+máscara concatenada
     - MIGAN            → inputs separados image [1,3,H,W] + mask [1,1,H,W]
   
   Detección automática del tipo según la metadata de la sesión.
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
let currentModelType = null;  // 'lama' | 'migan' | 'unknown'

// Metadata detectada del modelo
let modelInfo = {
  type: 'unknown',
  inputImageName: null,
  inputMaskName: null,
  inputChannels: 3,       // 3 o 4 canales para el input de imagen
  fixedSize: null,        // null si es dinámico, [H, W] si es fijo
  maskInverted: false     // MIGAN usa 255=conocido, 0=eliminar (inverso a LaMa)
};

// Padding alrededor del bbox pintado por el usuario
const PAD_USER = 25;

// Tamaño mínimo alineable (múltiplo de 32)
function alignTo32(v) {
  return Math.max(32, Math.ceil(v / 32) * 32);
}

console.log('[worker] Iniciado. Android=' + WORKER_IS_ANDROID);

/* ═══════════════════════════════════════════════════════════════
   Serialización robusta de errores
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
   Mensajes de control
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
   Detección del modelo (LaMa vs MIGAN)
   
   - LaMa:  input único [1, 4, H, W] con RGB+máscara concatenada
   - MIGAN: inputs separados image [1, 3, H, W] + mask [1, 1, H, W]
            donde mask=255 significa "conocido" y mask=0 "eliminar"
   ═══════════════════════════════════════════════════════════════ */
function detectModelType(session) {
  var inputs = session.inputNames;
  var meta = session.inputMetadata || {};

  modelInfo = {
    type: 'unknown',
    inputImageName: null,
    inputMaskName: null,
    inputChannels: 3,
    fixedSize: null,
    maskInverted: false
  };

  var foundImage = null;
  var foundMask = null;
  var imageChannels = 3;
  var imageDims = null;

  for (var i = 0; i < inputs.length; i++) {
    var name = inputs[i];
    var info = meta[name];
    var dims = info && info.dimensions ? info.dimensions.slice() : null;
    if (!dims || dims.length !== 4) continue;

    var ch = dims[1];
    if ((ch === 4 || ch === 3) && !foundImage) {
      foundImage = name;
      imageChannels = ch;
      imageDims = dims;
    } else if (ch === 1 && !foundMask) {
      foundMask = name;
    }
  }

  // Fallback por nombre
  if (!foundImage) {
    for (var j = 0; j < inputs.length; j++) {
      var ln = inputs[j].toLowerCase();
      if (ln.indexOf('image') >= 0 || ln.indexOf('img') >= 0) {
        foundImage = inputs[j];
        var info2 = meta[inputs[j]];
        imageDims = info2 ? info2.dimensions : null;
        imageChannels = (imageDims && imageDims[1]) || 3;
        break;
      }
    }
  }
  if (!foundMask) {
    for (var k = 0; k < inputs.length; k++) {
      var ln2 = inputs[k].toLowerCase();
      if (ln2.indexOf('mask') >= 0) { foundMask = inputs[k]; break; }
    }
  }

  modelInfo.inputImageName = foundImage;
  modelInfo.inputMaskName = foundMask;
  modelInfo.inputChannels = imageChannels;

  // Detectar tamaño fijo
  if (imageDims && imageDims[2] > 0 && imageDims[3] > 0) {
    modelInfo.fixedSize = { h: imageDims[2], w: imageDims[3] };
  }

  // Determinar tipo
  if (imageChannels === 4 && !foundMask) {
    modelInfo.type = 'lama';
  } else if (imageChannels === 3 && foundMask) {
    modelInfo.type = 'migan';
    modelInfo.maskInverted = true;  // MIGAN: 255=conocido, 0=eliminar
  } else if (imageChannels === 3 && !foundMask) {
    // Modelo con imagen 3ch y sin máscara separada: asumimos LaMa con 3ch
    modelInfo.type = 'lama';
  } else {
    modelInfo.type = 'lama';
  }

  console.log('[worker] Modelo detectado: ' + modelInfo.type);
  console.log('  inputImage:', foundImage, 'channels=' + imageChannels, JSON.stringify(imageDims));
  console.log('  inputMask:', foundMask);
  console.log('  fixedSize:', JSON.stringify(modelInfo.fixedSize));
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
   Construcción de tensores para LaMa
   - 4 canales: RGB (0..1) con máscara a negro + canal máscara (0/1)
   - 3 canales: RGB (0..1) con máscara a negro
   ═══════════════════════════════════════════════════════════════ */
function buildLamaInput(imgPx, maskPx, width, height) {
  var size = width * height;
  var channels = modelInfo.inputChannels;

  if (channels === 4) {
    var f4 = new Float32Array(4 * size);
    for (var i = 0; i < size; i++) {
      var i4 = i * 4;
      var m = maskPx[i4] > 127 ? 1 : 0;
      if (m === 1) {
        f4[i] = 0; f4[size + i] = 0; f4[2 * size + i] = 0;
      } else {
        f4[i]            = imgPx[i4]     / 255;
        f4[size + i]     = imgPx[i4 + 1] / 255;
        f4[2 * size + i] = imgPx[i4 + 2] / 255;
      }
      f4[3 * size + i] = m;
    }
    return new ort.Tensor('float32', f4, [1, 4, height, width]);
  }

  var f3 = new Float32Array(3 * size);
  for (var j = 0; j < size; j++) {
    var j4 = j * 4;
    var mm = maskPx[j4] > 127 ? 1 : 0;
    if (mm === 1) {
      f3[j] = 0; f3[size + j] = 0; f3[2 * size + j] = 0;
    } else {
      f3[j]            = imgPx[j4]     / 255;
      f3[size + j]     = imgPx[j4 + 1] / 255;
      f3[2 * size + j] = imgPx[j4 + 2] / 255;
    }
  }
  return new ort.Tensor('float32', f3, [1, 3, height, width]);
}

/* ═══════════════════════════════════════════════════════════════
   Construcción de tensores para MIGAN
   - image: uint8 RGB [1, 3, H, W], valores 0..255
   - mask:  uint8 [1, 1, H, W], 255 = región conocida, 0 = eliminar
   
   IMPORTANTE: invertimos la máscara del usuario, porque en nuestra UI
   el blanco (255) significa "eliminar", pero en MIGAN 0 = eliminar.
   ═══════════════════════════════════════════════════════════════ */
function buildMiganInput(imgPx, maskPx, width, height) {
  var size = width * height;
  var imgData = new Uint8Array(3 * size);
  var maskData = new Uint8Array(size);

  for (var i = 0; i < size; i++) {
    var i4 = i * 4;
    imgData[i]            = imgPx[i4];
    imgData[size + i]     = imgPx[i4 + 1];
    imgData[2 * size + i] = imgPx[i4 + 2];

    // Invertir: blanco (255) del usuario → 0 en MIGAN (eliminar)
    //          negro (0) del usuario → 255 en MIGAN (conservar)
    maskData[i] = maskPx[i4] > 127 ? 0 : 255;
  }

  return {
    image: new ort.Tensor('uint8', imgData, [1, 3, height, width]),
    mask: new ort.Tensor('uint8', maskData, [1, 1, height, width])
  };
}

/* ═══════════════════════════════════════════════════════════════
   Cálculo de geometría del crop
   ═══════════════════════════════════════════════════════════════ */
function computeCropGeometry(bbox, imgW, imgH, targetSize) {
  var cropW, cropH;

  if (targetSize) {
    // Modelo de tamaño fijo (LaMa 512×512)
    var side = Math.max(bbox.w, bbox.h) + PAD_USER * 2;
    if (side < 64) side = 64;
    if (side > targetSize) side = targetSize;
    cropW = side;
    cropH = side;
  } else {
    // Modelo dinámico (MIGAN): crop = bbox + padding, alineado a 32
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

  // Alinear al múltiplo de 32 si es dinámico
  if (!targetSize) {
    realW = alignTo32(realW);
    realH = alignTo32(realH);
    if (cropX + realW > imgW) realW = Math.max(32, imgW - cropX);
    if (cropY + realH > imgH) realH = Math.max(32, imgH - cropY);
  }

  return { x: cropX, y: cropY, w: realW, h: realH };
}

/* ═══════════════════════════════════════════════════════════════
   Extrae crop de la imagen completa (con reflejo si sale de bordes)
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
  if (header[0] === 0x3C) {
    throw new Error('El buffer es HTML, no un ONNX.');
  }

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
  currentModelType = modelInfo.type;

  return {
    provider: usedProvider,
    modelType: modelInfo.type,
    inputs: {
      image: modelInfo.inputImageName,
      mask: modelInfo.inputMaskName,
      channels: modelInfo.inputChannels,
      fixedSize: modelInfo.fixedSize,
      names: session.inputNames.slice()
    }
  };
}

/* ═══════════════════════════════════════════════════════════════
   Inferencia sobre una región
   ═══════════════════════════════════════════════════════════════ */
async function runOnRegion(imageData, maskFull, imgW, imgH, bbox) {
  // 1) Calcular geometría del crop
  var targetSize = modelInfo.fixedSize ? modelInfo.fixedSize.w : null;
  var geom = computeCropGeometry(bbox, imgW, imgH, targetSize);

  console.log('[worker] Crop=' + geom.w + '×' + geom.h +
              ' en (' + geom.x + ',' + geom.y + ')' +
              ' · tipo=' + modelInfo.type);

  // 2) Extraer crop de imagen y máscara
  var cropImg = extractCrop(imageData, imgW, imgH, geom);
  var cropMask = extractMaskCrop(maskFull, imgW, imgH, geom);

  // 3) Construir tensores según tipo de modelo
  var feeds = {};
  var cropW = geom.w;
  var cropH = geom.h;

  if (modelInfo.type === 'migan') {
    var migan = buildMiganInput(cropImg, cropMask, cropW, cropH);
    feeds[modelInfo.inputImageName] = migan.image;
    feeds[modelInfo.inputMaskName] = migan.mask;
  } else {
    // LaMa
    var lama = buildLamaInput(cropImg, cropMask, cropW, cropH);
    feeds[modelInfo.inputImageName] = lama;
  }

  // Liberar referencias grandes antes de inferir
  cropImg = null;
  cropMask = null;

  // 4) Inferencia
  var t0 = Date.now();
  var results = await currentSession.run(feeds);
  var dt = ((Date.now() - t0) / 1000).toFixed(2);
  console.log('[worker] Inferencia OK en ' + dt + 's');

  feeds = null;

  // 5) Convertir salida
  var outKey = currentSession.outputNames[0] || Object.keys(results)[0];
  var outTensor = results[outKey];
  var outDims = outTensor.dims;
  var outH = outDims[outDims.length - 2];
  var outW = outDims[outDims.length - 1];

  // MIGAN devuelve la imagen ya "pasted" a las dimensiones del input.
  // LaMa también. Así que extraemos el resultado tal cual.
  var resultImg = tensorToImageData(outTensor, outW, outH);

  // 6) Ajustar tamaño: si la salida es mayor que el crop, recortar
  var finalW = Math.min(outW, cropW);
  var finalH = Math.min(outH, cropH);

  if (finalW !== outW || finalH !== outH) {
    // Recortar desde la esquina superior izquierda (para modelos que devuelven padding)
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
   Pegar resultado en la imagen completa
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
   Estrategia de ejecución
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

  console.log('[worker] Estrategia=' + strategy +
              ' | regiones=' + regions.length +
              ' | tipo=' + modelInfo.type);

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
    self.postMessage({
      id: id,
      type: 'error',
      error: errMsg
    });
  }
};