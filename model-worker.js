/* ═══════════════════════════════════════════════════════════════
   model-worker.js — ONNX Runtime Web Worker para LAMA Inpainting
   
   MODELO DINÁMICO: acepta H×W múltiplos de 32 (no fuerza 512×512).
   Procesa solo el área del bounding box de la máscara + padding.
   
   Ventajas del modelo dinámico:
   - Mucho menos memoria en Android (procesa solo lo necesario)
   - Mejor contexto (padding real de la imagen, no reflejo inventado)
   
   Estrategia de ejecución:
   - CASO 1: 1 región → recorte bbox+25px, alineado a múltiplo de 32
   - CASO 2: varias regiones pequeñas → cada una por separado
   - CASO 3: muchas regiones grandes → imagen completa alineada
   ═══════════════════════════════════════════════════════════════ */

// 1) Cargar ONNX Runtime al inicio
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');

// 2) Configurar WASM
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

// 3) Detección Android
const WORKER_IS_ANDROID = /Android/i.test(navigator.userAgent);
let forceWasm = WORKER_IS_ANDROID;
let useWebGPU = false;
let currentSession = null;
let currentModelId = null;

// Metadata del modelo (se detecta al cargar)
let inputImageName = null;
let inputMaskName = null;
let inputImageDims = null;
let inputMaskDims = null;

// Padding alrededor del bbox pintado por el usuario
const PAD_USER = 25;

console.log('[worker] Iniciado. Android=' + WORKER_IS_ANDROID);

/* ═══════════════════════════════════════════════════════════════
   Utilidad: serialización robusta de errores
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
   Detección de inputs del modelo
   ═══════════════════════════════════════════════════════════════ */
function detectInputInfo(session) {
  var inputs = session.inputNames;
  inputImageName = null;
  inputMaskName = null;
  inputImageDims = null;
  inputMaskDims = null;

  var meta = session.inputMetadata || {};

  for (var i = 0; i < inputs.length; i++) {
    var name = inputs[i];
    var info = meta[name];
    var dims = info && info.dimensions ? info.dimensions.slice() : null;

    if (dims && dims.length === 4) {
      var ch = dims[1];
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

  console.log('[worker] Inputs detectados:');
  console.log('  image:', inputImageName, JSON.stringify(inputImageDims));
  console.log('  mask: ', inputMaskName, JSON.stringify(inputMaskDims));
  console.log('  outputs:', session.outputNames.join(','));
}

/* ═══════════════════════════════════════════════════════════════
   Alineación a múltiplos de 32
   El modelo dinámico de LAMA requiere H y W múltiplos de 32.
   ═══════════════════════════════════════════════════════════════ */
function alignTo32(value) {
  return Math.max(32, Math.ceil(value / 32) * 32);
}

/* ═══════════════════════════════════════════════════════════════
   Construcción del tensor [1, 4, H, W]
   - Canales 0-2: imagen RGB con la zona enmascarada pintada de NEGRO
   - Canal 3: máscara binaria (0 o 1)
   ═══════════════════════════════════════════════════════════════ */
function buildInputTensor(imgData, maskData, width, height) {
  var size = width * height;
  var float = new Float32Array(4 * size);
  var imgPx = imgData.data;
  var maskPx = maskData.data;

  for (var i = 0; i < size; i++) {
    var i4 = i * 4;
    var m = maskPx[i4] > 127 ? 1 : 0;

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

/* ═══════════════════════════════════════════════════════════════
   Conversión de salida del modelo a ImageData
   ═══════════════════════════════════════════════════════════════ */
function tensorToImageData(tensor, width, height) {
  var out = new Uint8ClampedArray(width * height * 4);
  var data = tensor.data;
  var dims = tensor.dims;
  var size = width * height;

  var channels = dims.length >= 2 ? dims[1] : 3;
  var rOff, gOff, bOff;
  if (channels >= 3) {
    rOff = 0; gOff = size; bOff = 2 * size;
  } else {
    rOff = 0; gOff = 0; bOff = 0;
  }

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
   Cálculo de geometría del crop (dimensiones dinámicas)
   ═══════════════════════════════════════════════════════════════ */
function computeCropGeometry(bbox, imgW, imgH) {
  // Padding simétrico
  var cropW = bbox.w + PAD_USER * 2;
  var cropH = bbox.h + PAD_USER * 2;

  // Alinear a múltiplo de 32
  cropW = alignTo32(cropW);
  cropH = alignTo32(cropH);

  // Centro del bbox
  var userCx = bbox.x + bbox.w / 2;
  var userCy = bbox.y + bbox.h / 2;

  // Esquina superior izquierda, centrado
  var cropX = Math.round(userCx - cropW / 2);
  var cropY = Math.round(userCy - cropH / 2);

  // Ajustar para no salir de la imagen
  if (cropX < 0) cropX = 0;
  if (cropY < 0) cropY = 0;
  if (cropX + cropW > imgW) cropX = Math.max(0, imgW - cropW);
  if (cropY + cropH > imgH) cropY = Math.max(0, imgH - cropH);

  // Tamaño real (puede ser menor si la imagen es chica)
  var realW = Math.min(cropW, imgW - cropX);
  var realH = Math.min(cropH, imgH - cropY);

  // Alinear el tamaño real
  realW = alignTo32(realW);
  realH = alignTo32(realH);
  if (cropX + realW > imgW) realW = Math.max(32, imgW - cropX);
  if (cropY + realH > imgH) realH = Math.max(32, imgH - cropY);

  return {
    x: cropX,
    y: cropY,
    w: realW,
    h: realH
  };
}

/* ═══════════════════════════════════════════════════════════════
   Extrae un crop de la imagen, con reflejo si sale de los bordes
   ═══════════════════════════════════════════════════════════════ */
function extractCrop(imageData, imgW, imgH, geom) {
  var cropImg = new Uint8ClampedArray(geom.w * geom.h * 4);
  var src = imageData.data;

  for (var j = 0; j < geom.h; j++) {
    for (var i = 0; i < geom.w; i++) {
      var dstIdx = (j * geom.w + i) * 4;
      var sx = geom.x + i;
      var sy = geom.y + j;

      // Reflejo si está fuera
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

  return new ImageData(cropImg, geom.w, geom.h);
}

/* ═══════════════════════════════════════════════════════════════
   Extrae la máscara del crop
   ═══════════════════════════════════════════════════════════════ */
function extractMaskCrop(maskData, imgW, imgH, geom) {
  var cropMask = new Uint8ClampedArray(geom.w * geom.h * 4);
  var src = maskData;

  for (var j = 0; j < geom.h; j++) {
    for (var i = 0; i < geom.w; i++) {
      var dstIdx = (j * geom.w + i) * 4;
      var sx = geom.x + i;
      var sy = geom.y + j;

      if (sx < 0 || sx >= imgW || sy < 0 || sy >= imgH) {
        cropMask[dstIdx]     = 0;
        cropMask[dstIdx + 1] = 0;
        cropMask[dstIdx + 2] = 0;
        cropMask[dstIdx + 3] = 255;
        continue;
      }

      var srcIdx = (sy * imgW + sx) * 4;
      cropMask[dstIdx]     = src[srcIdx];
      cropMask[dstIdx + 1] = src[srcIdx + 1];
      cropMask[dstIdx + 2] = src[srcIdx + 2];
      cropMask[dstIdx + 3] = 255;
    }
  }

  return new ImageData(cropMask, geom.w, geom.h);
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
  var headerHex = Array.from(header).map(function (b) {
    return ('0' + b.toString(16)).slice(-2);
  }).join(' ');

  if (header[0] === 0x3C) {
    throw new Error('El buffer es HTML, no un ONNX. Header: ' + headerHex);
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
        graphOptimizationLevel: 'all'
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

  detectInputInfo(session);

  return {
    provider: usedProvider,
    inputs: {
      image: inputImageName,
      mask: inputMaskName,
      names: session.inputNames.slice()
    }
  };
}

/* ═══════════════════════════════════════════════════════════════
   Inferencia sobre una región (con tamaño dinámico)
   ═══════════════════════════════════════════════════════════════ */
async function runOnRegion(imageData, maskData, imgW, imgH, bbox) {
  // 1) Calcular geometría dinámica
  var geom = computeCropGeometry(bbox, imgW, imgH);

  console.log('[worker] Región: bbox=' + bbox.w + '×' + bbox.h +
              ' → crop=' + geom.w + '×' + geom.h +
              ' en (' + geom.x + ',' + geom.y + ')');

  // 2) Extraer crop de imagen y máscara
  var cropImg = extractCrop(imageData, imgW, imgH, geom);
  var cropMask = extractMaskCrop(maskData, imgW, imgH, geom);

  // 3) Construir tensor [1, 4, geom.h, geom.w]
  var inputTensor = buildInputTensor(cropImg, cropMask, geom.w, geom.h);

  // 4) Ejecutar inferencia
  var feeds = {};
  feeds[inputImageName] = inputTensor;

  // Si el modelo tiene input separado para máscara
  if (inputMaskName && inputMaskName !== inputImageName) {
    var size = geom.w * geom.h;
    var maskFloat = new Float32Array(size);
    var maskPx = cropMask.data;
    for (var k = 0; k < size; k++) {
      maskFloat[k] = maskPx[k * 4] > 127 ? 1 : 0;
    }
    feeds[inputMaskName] = new ort.Tensor('float32', maskFloat, [1, 1, geom.h, geom.w]);
  }

  var t0 = Date.now();
  var results = await currentSession.run(feeds);
  var dt = ((Date.now() - t0) / 1000).toFixed(2);
  console.log('[worker] Inferencia OK en ' + dt + 's');

  // 5) Convertir salida a ImageData
  var outKey = currentSession.outputNames[0] || Object.keys(results)[0];
  var outTensor = results[outKey];
  var outDims = outTensor.dims;
  var outH = outDims[outDims.length - 2];
  var outW = outDims[outDims.length - 1];

  var resultCrop = tensorToImageData(outTensor, outW, outH);

  return {
    imageData: resultCrop,
    pasteX: geom.x,
    pasteY: geom.y,
    pasteW: geom.w,
    pasteH: geom.h
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
              ' | ratio=' + maskRatio.toFixed(3));

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