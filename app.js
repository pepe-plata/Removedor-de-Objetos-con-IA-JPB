/* ═══════════════════════════════════════════════════════════════
   app.js — Removedor de Objetos con IA JPB
   Modelos: LAMA FP16 / LAMA INT8 (seleccionables por el usuario)
   Gestos: 2 dedos = pan O zoom (excluyentes)
   ═══════════════════════════════════════════════════════════════ */

'use strict';

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }

var IS_ANDROID = /Android/i.test(navigator.userAgent);
var IS_MOBILE = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);

/* ═══════════════════════════════════════════════════════════════
   Estado global
   ═══════════════════════════════════════════════════════════════ */
var state = {
  originalCanvas: null,
  currentCanvas:  null,
  maskCanvas:     null,
  originalCtx:    null,
  currentCtx:     null,
  maskCtx:        null,

  imgWidth: 0,
  imgHeight: 0,
  imgName: '',
  imgSizeBytes: 0,
  originalJpegBuffer: null,

  zoom: 1,
  offsetX: 0,
  offsetY: 0,
  showOriginal: false,

  activeTool: 'brush',
  brushSize: 30,
  maskMode: 'add',
  isDrawing: false,
  lastPoint: null,
  lassoPoints: [],
  rectStart: null,
  rectEnd: null,

  history: [],
  historyIndex: -1,
  MAX_HISTORY: 30,

  worker: null,
  workerReady: false,
  modelLoaded: false,
  modelLoadedId: null,
  modelProvider: null,
  modelInputSize: null,
  processing: false,

  wakeLock: null
};

/* ═══════════════════════════════════════════════════════════════
   Modelos disponibles
   
   - lama_fp16.onnx : ~107 MB · mejor calidad · WebGPU en PC
   - lama_int8.onnx : ~62 MB  · buena calidad · más rápido en CPU/WASM
   
   El usuario puede elegir cualquiera desde el selector.
   ═══════════════════════════════════════════════════════════════ */
var MODELS = [
  {
    id: 'lama-fp16',
    name: 'LAMA FP16 (Alta calidad)',
    description: 'Mejor calidad, ideal para WebGPU en PC',
    size: '~107 MB',
    urls: [
      'https://huggingface.co/g-ronimo/lama/resolve/main/lama_fp16.onnx',
      'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp16.onnx'
    ]
  },
  {
    id: 'lama-int8',
    name: 'LAMA INT8 (Rápido en móvil)',
    description: 'Menor tamaño y más rápido en CPU/WASM',
    size: '~62 MB',
    urls: [
      'https://huggingface.co/g-ronimo/lama/resolve/main/lama_int8.onnx',
      'https://huggingface.co/g-ronimo/lama/resolve/main/lama.onnx'
    ]
  }
];

function getModelById(id) {
  for (var i = 0; i < MODELS.length; i++) if (MODELS[i].id === id) return MODELS[i];
  return null;
}

/* ═══════════════════════════════════════════════════════════════
   Validación de buffer ONNX
   ═══════════════════════════════════════════════════════════════ */
function isValidOnnxBuffer(buffer) {
  if (!buffer) return { ok: false, reason: 'buffer es null/undefined' };
  if (!(buffer instanceof ArrayBuffer)) return { ok: false, reason: 'no es ArrayBuffer (tipo: ' + typeof buffer + ')' };
  if (buffer.byteLength < 1024) return { ok: false, reason: 'demasiado pequeño (' + buffer.byteLength + ' bytes)' };

  var header = new Uint8Array(buffer, 0, Math.min(16, buffer.byteLength));
  var headerStr = String.fromCharCode.apply(null, header);

  if (header[0] === 0x3C) {
    return { ok: false, reason: 'empieza con "<" → es HTML, no ONNX', header: headerStr };
  }
  if (headerStr.indexOf('<!') === 0 || headerStr.indexOf('<html') >= 0 || headerStr.indexOf('<?xml') >= 0) {
    return { ok: false, reason: 'es una página HTML/XML, no un modelo', header: headerStr };
  }
  if (headerStr.indexOf('Not Found') >= 0 || headerStr.indexOf('error') >= 0) {
    return { ok: false, reason: 'contiene mensaje de error en texto', header: headerStr };
  }

  return {
    ok: true,
    header: Array.from(header).map(function (b) { return ('0' + b.toString(16)).slice(-2); }).join(' ')
  };
}

/* ═══════════════════════════════════════════════════════════════
   Inicialización
   ═══════════════════════════════════════════════════════════════ */
function init() {
  console.log('[JPB] init() | Android=' + IS_ANDROID + ' | Mobile=' + IS_MOBILE);

  var splashEl = $('#splash');
  if (splashEl) setTimeout(function () { splashEl.classList.add('hide'); }, 2000);

  createInternalCanvases(1, 1);
  syncThemeSwitch();

  // Rellenar el selector de modelos con TODOS los modelos disponibles
  var modelSelect = $('#modelSelect');
  if (modelSelect) {
    modelSelect.innerHTML = '';
    for (var i = 0; i < MODELS.length; i++) {
      var opt = document.createElement('option');
      opt.value = MODELS[i].id;
      opt.textContent = MODELS[i].name;
      modelSelect.appendChild(opt);
    }
    // Modelo por defecto según plataforma (pero el usuario puede cambiar)
    modelSelect.value = IS_ANDROID ? 'lama-int8' : 'lama-fp16';
  }

  if (!window.JPBDB || !window.JPBDB.isAvailable || !window.JPBDB.isAvailable()) {
    console.warn('[JPB] IndexedDB no disponible');
    setStatus('Advertencia: almacenamiento local no disponible');
  } else {
    window.JPBDB.requestPersistence().catch(function () {});
  }

  bindUI();
  bindCanvasEvents();
  bindKeyboard();
  setupGlobalDelegation();
  registerSW();
  checkSharedFiles();

  if ('launchQueue' in window) {
    try {
      window.launchQueue.setConsumer(async function (launchParams) {
        if (launchParams.files && launchParams.files.length) {
          var fileHandle = launchParams.files[0];
          var file = await fileHandle.getFile();
          loadImageFile(file);
        }
      });
    } catch (e) {}
  }

  updateUIState();
  setStatus('Listo');
}

/* ═══════════════════════════════════════════════════════════════
   Delegación global para botones que abren el file input
   ═══════════════════════════════════════════════════════════════ */
function setupGlobalDelegation() {
  document.addEventListener('click', function (e) {
    var target = e.target;
    while (target && target !== document) {
      if (target.id === 'openBtn' || target.id === 'emptySelect' ||
          (target.dataset && target.dataset.action === 'open')) {
        e.preventDefault();
        e.stopPropagation();
        var fi = document.getElementById('fileInput');
        if (fi) fi.click();
        return;
      }
      target = target.parentNode;
    }
  }, true);
}

/* ═══════════════════════════════════════════════════════════════
   Canvas internos
   ═══════════════════════════════════════════════════════════════ */
function createInternalCanvases(w, h) {
  state.originalCanvas = document.createElement('canvas');
  state.currentCanvas  = document.createElement('canvas');
  state.maskCanvas     = document.createElement('canvas');

  state.originalCanvas.width = state.currentCanvas.width = state.maskCanvas.width = w;
  state.originalCanvas.height = state.currentCanvas.height = state.maskCanvas.height = h;

  state.originalCtx = state.originalCanvas.getContext('2d', { willReadFrequently: true });
  state.currentCtx  = state.currentCanvas.getContext('2d', { willReadFrequently: true });
  state.maskCtx     = state.maskCanvas.getContext('2d', { willReadFrequently: true });
}

function resizeInternalCanvases(w, h) {
  state.originalCanvas.width = state.currentCanvas.width = state.maskCanvas.width = w;
  state.originalCanvas.height = state.currentCanvas.height = state.maskCanvas.height = h;
  state.imgWidth = w;
  state.imgHeight = h;
}

/* ═══════════════════════════════════════════════════════════════
   Tema
   ═══════════════════════════════════════════════════════════════ */
function syncThemeSwitch() {
  var sw = $('#themeSwitch');
  if (!sw) return;
  var isDark = document.documentElement.classList.contains('dark');
  sw.setAttribute('aria-checked', isDark ? 'true' : 'false');
}

function toggleTheme() {
  var html = document.documentElement;
  var isDark = html.classList.contains('dark');
  if (isDark) {
    html.classList.remove('dark');
    html.classList.add('light');
    try { localStorage.setItem('jpb-theme', 'light'); } catch (e) {}
  } else {
    html.classList.remove('light');
    html.classList.add('dark');
    try { localStorage.setItem('jpb-theme', 'dark'); } catch (e) {}
  }
  syncThemeSwitch();
  redrawMainCanvas();
}

/* ═══════════════════════════════════════════════════════════════
   Bind UI
   ═══════════════════════════════════════════════════════════════ */
function bindUI() {
  var menuBtn = $('#menuBtn');
  var sidebar = $('#sidebar');
  var backdrop = $('#sidebarBackdrop');

  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', function () {
      if (window.matchMedia('(max-width: 720px)').matches) {
        sidebar.classList.toggle('open');
        if (backdrop) backdrop.classList.toggle('show', sidebar.classList.contains('open'));
      } else {
        sidebar.classList.toggle('collapsed');
      }
    });
  }
  if (backdrop && sidebar) {
    backdrop.addEventListener('click', function () {
      sidebar.classList.remove('open');
      backdrop.classList.remove('show');
    });
  }

  var themeSwitch = $('#themeSwitch');
  if (themeSwitch) {
    themeSwitch.addEventListener('click', toggleTheme);
    themeSwitch.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleTheme(); }
    });
  }

  var helpBtn = $('#helpBtn');
  var helpModal = $('#helpModal');
  if (helpBtn && helpModal) helpBtn.addEventListener('click', function () { openModal(helpModal); });

  var saveBtn = $('#saveBtn');
  var fileInput = $('#fileInput');

  if (saveBtn) saveBtn.addEventListener('click', function () { if (canSave()) openSaveModal(); });

  if (fileInput) {
    fileInput.addEventListener('change', function (e) {
      var f = e.target.files && e.target.files[0];
      if (f) loadImageFile(f);
      fileInput.value = '';
    });
  }

  $$('.sec-head').forEach(function (h) {
    h.addEventListener('click', function () {
      var parent = h.parentElement;
      if (parent) parent.classList.toggle('collapsed');
    });
  });

  $$('.tool-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('.tool-btn').forEach(function (x) { x.classList.remove('active'); });
      b.classList.add('active');
      state.activeTool = b.dataset.tool;
      updateBrushCursor();
    });
  });

  var brushSizeEl = $('#brushSize');
  var brushSizeVal = $('#brushSizeVal');
  if (brushSizeEl && brushSizeVal) {
    brushSizeEl.addEventListener('input', function () {
      state.brushSize = +brushSizeEl.value;
      brushSizeVal.textContent = state.brushSize;
      updateBrushCursor();
    });
  }

  var modeAdd = $('#modeAdd');
  var modeSub = $('#modeSub');
  if (modeAdd) modeAdd.addEventListener('click', function () { setMaskMode('add'); });
  if (modeSub) modeSub.addEventListener('click', function () { setMaskMode('sub'); });

  var maskClear = $('#maskClear');
  if (maskClear) maskClear.addEventListener('click', clearMask);

  var maskApply = $('#maskApply');
  if (maskApply) maskApply.addEventListener('click', runAI);

  var manageModelsBtn = $('#manageModelsBtn');
  if (manageModelsBtn) manageModelsBtn.addEventListener('click', openModelsModal);

  var undoBtn = $('#undoBtn');
  var redoBtn = $('#redoBtn');
  var fitBtn = $('#fitBtn');
  var toggleBtn = $('#toggleBtn');
  var copyBtn = $('#copyBtn');
  if (undoBtn) undoBtn.addEventListener('click', historyUndo);
  if (redoBtn) redoBtn.addEventListener('click', historyRedo);
  if (fitBtn) fitBtn.addEventListener('click', fitToViewport);
  if (toggleBtn) toggleBtn.addEventListener('click', toggleOriginal);
  if (copyBtn) copyBtn.addEventListener('click', copyToClipboard);

  var resetBtn = $('#resetBtn');
  if (resetBtn) resetBtn.addEventListener('click', resetAll);

  var saveFormat = $('#saveFormat');
  var qualityRow = $('#qualityRow');
  var qualityRange = $('#qualityRange');
  var qualityVal = $('#qualityVal');
  var exifRow = $('#exifRow');
  var confirmSave = $('#confirmSave');

  if (saveFormat) {
    saveFormat.addEventListener('change', function () {
      var f = saveFormat.value;
      if (qualityRow) qualityRow.classList.toggle('hidden', f !== 'jpeg' && f !== 'webp');
      if (exifRow) exifRow.classList.toggle('hidden', f !== 'jpeg');
    });
  }
  if (qualityRange && qualityVal) {
    qualityRange.addEventListener('input', function () {
      qualityVal.textContent = qualityRange.value;
    });
  }
  if (confirmSave) confirmSave.addEventListener('click', doSave);

  $$('[data-close]').forEach(function (b) {
    b.addEventListener('click', function () {
      var m = document.getElementById(b.dataset.close);
      if (m) closeModal(m);
    });
  });
  ['saveModal', 'modelsModal', 'helpModal'].forEach(function (id) {
    var m = document.getElementById(id);
    if (!m) return;
    m.addEventListener('click', function (e) { if (e.target === m) closeModal(m); });
  });

  $$('.tab').forEach(function (t) {
    t.addEventListener('click', function () {
      $$('.tab').forEach(function (x) { x.classList.remove('active'); });
      $$('.tab-panel').forEach(function (x) { x.classList.remove('active'); });
      t.classList.add('active');
      var panel = document.getElementById(t.dataset.tab);
      if (panel) panel.classList.add('active');
    });
  });

  setupDragDrop();
  window.addEventListener('paste', handlePaste);

  var modelSelect2 = $('#modelSelect');
  if (modelSelect2) {
    modelSelect2.addEventListener('change', function () {
      state.modelLoaded = false;
      state.modelLoadedId = null;
      setStatus('Modelo cambiado a: ' + (getModelById(modelSelect2.value) || {}).name);
    });
  }

  window.addEventListener('resize', function () {
    if (state.imgWidth) updateTransform();
  });
}

/* ═══════════════════════════════════════════════════════════════
   Canvas eventos (dibujo, pan, zoom, gestos táctiles)
   ═══════════════════════════════════════════════════════════════ */
function bindCanvasEvents() {
  var canvasWrap = $('#canvasWrap');
  if (!canvasWrap) return;

  var pointers = new Map();
  var isPanning = false;
  var panStart = null;
  var pendingDrawTimer = null;
  var spaceDown = false;

  // Estado para gestos de 2 dedos
  var gesture = {
    active: false,
    mode: null,          // 'pinch' | 'pan' | null
    startDist: 0,
    startZoom: 1,
    startMidX: 0,
    startMidY: 0,
    startOffsetX: 0,
    startOffsetY: 0,
    // Referencia inicial de cada dedo
    p1StartX: 0,
    p1StartY: 0,
    p2StartX: 0,
    p2StartY: 0
  };

  canvasWrap.addEventListener('pointerdown', onPointerDown);
  canvasWrap.addEventListener('pointermove', onPointerMove);
  canvasWrap.addEventListener('pointerup', onPointerUp);
  canvasWrap.addEventListener('pointercancel', onPointerUp);
  canvasWrap.addEventListener('wheel', onWheel, { passive: false });

  if (!window.matchMedia('(hover: none)').matches) {
    canvasWrap.addEventListener('mouseenter', function () { if (state.imgWidth) showBrushCursor(); });
    canvasWrap.addEventListener('mouseleave', hideBrushCursor);
    canvasWrap.addEventListener('mousemove', updateBrushCursorPos);
  }

  document.addEventListener('keydown', function (e) {
    if (e.code === 'Space' && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) {
      spaceDown = true;
      canvasWrap.style.cursor = 'grab';
    }
  });
  document.addEventListener('keyup', function (e) {
    if (e.code === 'Space') {
      spaceDown = false;
      canvasWrap.style.cursor = '';
    }
  });

  function onPointerDown(e) {
    try { canvasWrap.setPointerCapture(e.pointerId); } catch (err) {}
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointers.size === 2) {
      // Cancelar cualquier dibujo pendiente
      if (pendingDrawTimer) { clearTimeout(pendingDrawTimer); pendingDrawTimer = null; }
      cancelDrawing();
      startGesture();
      return;
    }

    if (pointers.size === 1) {
      if (spaceDown || e.button === 1) {
        isPanning = true;
        panStart = { x: e.clientX - state.offsetX, y: e.clientY - state.offsetY };
        canvasWrap.style.cursor = 'grabbing';
        return;
      }
      if (state.imgWidth) {
        pendingDrawTimer = setTimeout(function () {
          pendingDrawTimer = null;
          startDrawing(e);
        }, 90);
      } else {
        isPanning = true;
        panStart = { x: e.clientX - state.offsetX, y: e.clientY - state.offsetY };
      }
    }
  }

  function onPointerMove(e) {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointers.size === 2) {
      updateGesture();
      return;
    }

    if (isPanning && panStart) {
      state.offsetX = e.clientX - panStart.x;
      state.offsetY = e.clientY - panStart.y;
      updateTransform();
      return;
    }

    if (state.isDrawing) {
      continueDrawing(e);
      if (window.matchMedia('(hover: hover)').matches) updateBrushCursorPos(e);
    } else if (window.matchMedia('(hover: hover)').matches) {
      updateBrushCursorPos(e);
    }
  }

  function onPointerUp(e) {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) {
      gesture.active = false;
      gesture.mode = null;
    }

    if (pendingDrawTimer) { clearTimeout(pendingDrawTimer); pendingDrawTimer = null; }

    if (isPanning) {
      isPanning = false;
      panStart = null;
      canvasWrap.style.cursor = spaceDown ? 'grab' : '';
    }

    if (state.isDrawing) endDrawing(e);
  }

  function onWheel(e) {
    if (!state.imgWidth) return;
    e.preventDefault();
    if (e.shiftKey) { state.offsetX -= e.deltaY; updateTransform(); return; }
    if (e.altKey)   { state.offsetY -= e.deltaY; updateTransform(); return; }
    var delta = -e.deltaY * 0.0015;
    var newZoom = Math.max(0.05, Math.min(20, state.zoom * (1 + delta)));
    zoomAt(e.clientX, e.clientY, newZoom);
  }

  /* ─── Gestos de 2 dedos ─── */
  function startGesture() {
    var pts = Array.from(pointers.values());
    if (pts.length < 2) return;

    var dx = pts[0].x - pts[1].x;
    var dy = pts[0].y - pts[1].y;
    var dist = Math.hypot(dx, dy);

    gesture.active = true;
    gesture.mode = null;   // se decide en el primer move
    gesture.startDist = dist;
    gesture.startZoom = state.zoom;
    gesture.startMidX = (pts[0].x + pts[1].x) / 2;
    gesture.startMidY = (pts[0].y + pts[1].y) / 2;
    gesture.startOffsetX = state.offsetX;
    gesture.startOffsetY = state.offsetY;
    gesture.p1StartX = pts[0].x;
    gesture.p1StartY = pts[0].y;
    gesture.p2StartX = pts[1].x;
    gesture.p2StartY = pts[1].y;
  }

  function updateGesture() {
    if (!gesture.active) return;
    var pts = Array.from(pointers.values());
    if (pts.length < 2) return;

    var dx = pts[0].x - pts[1].x;
    var dy = pts[0].y - pts[1].y;
    var dist = Math.hypot(dx, dy);

    var midX = (pts[0].x + pts[1].x) / 2;
    var midY = (pts[0].y + pts[1].y) / 2;

    // Desplazamiento del centro (indica pan)
    var midDx = midX - gesture.startMidX;
    var midDy = midY - gesture.startMidY;
    var midMove = Math.hypot(midDx, midDy);

    // Cambio de distancia (indica zoom)
    var distDelta = Math.abs(dist - gesture.startDist);
    var distRatio = dist / gesture.startDist;

    // ─── DECISIÓN DE MODO (solo la primera vez) ───
    if (gesture.mode === null) {
      // Umbrales para decidir intención
      var ZOOM_THRESHOLD = 8;    // px de cambio en la distancia entre dedos
      var PAN_THRESHOLD = 8;     // px de movimiento del centro

      // ¿El usuario está claramente haciendo zoom?
      var isZoomIntent = distDelta > ZOOM_THRESHOLD && distDelta > midMove * 0.8;

      // ¿El usuario está claramente haciendo pan?
      var isPanIntent = midMove > PAN_THRESHOLD && distDelta < ZOOM_THRESHOLD * 0.6;

      if (isZoomIntent) {
        gesture.mode = 'pinch';
      } else if (isPanIntent) {
        gesture.mode = 'pan';
      }
      // Si no se decide aún, no hacemos nada (esperamos más movimiento)
    }

    // ─── EJECUTAR SOLO EL MODO ACTIVO (nunca ambos) ───
    if (gesture.mode === 'pinch') {
      var newZoom = Math.max(0.05, Math.min(20, gesture.startZoom * distRatio));

      var rect = canvasWrap.getBoundingClientRect();
      // Zoom centrado en la posición inicial del centro (más estable)
      var cx = gesture.startMidX - rect.left;
      var cy = gesture.startMidY - rect.top;

      var k = newZoom / gesture.startZoom;
      state.offsetX = cx - (cx - gesture.startOffsetX) * k;
      state.offsetY = cy - (cy - gesture.startOffsetY) * k;
      state.zoom = newZoom;
      updateTransform();
    } else if (gesture.mode === 'pan') {
      // Pan: mover usando el desplazamiento del centro de los dedos
      state.offsetX = gesture.startOffsetX + midDx;
      state.offsetY = gesture.startOffsetY + midDy;
      updateTransform();
    }
    // Si mode es null: no hacer nada (esperar a que se decida)
  }

  /* ─── Dibujo ─── */
  function startDrawing(e) {
    if (!state.imgWidth) return;
    if (!state.history.length) saveMaskSnapshot();
    state.isDrawing = true;

    var pt = screenToImage(e.clientX, e.clientY);

    if (state.activeTool === 'brush') {
      state.lastPoint = pt;
      drawBrushPoint(pt);
      redrawMainCanvas();
    } else if (state.activeTool === 'rect') {
      state.rectStart = pt;
      state.rectEnd = pt;
      redrawMainCanvas();
    } else if (state.activeTool === 'lasso') {
      state.lassoPoints = [pt];
    }
  }

  function continueDrawing(e) {
    if (!state.imgWidth) return;
    var pt = screenToImage(e.clientX, e.clientY);

    if (state.activeTool === 'brush') {
      drawBrushLine(state.lastPoint, pt);
      state.lastPoint = pt;
      redrawMainCanvas();
    } else if (state.activeTool === 'rect') {
      state.rectEnd = pt;
      redrawMainCanvas();
    } else if (state.activeTool === 'lasso') {
      state.lassoPoints.push(pt);
      redrawMainCanvas();
    }
  }

  function endDrawing(e) {
    var pt = screenToImage(e.clientX, e.clientY);

    if (state.activeTool === 'brush') {
      drawBrushLine(state.lastPoint, pt);
      state.lastPoint = null;
    } else if (state.activeTool === 'rect') {
      state.rectEnd = pt;
      applyRect();
      state.rectStart = state.rectEnd = null;
    } else if (state.activeTool === 'lasso') {
      state.lassoPoints.push(pt);
      applyLasso();
      state.lassoPoints = [];
    }

    state.isDrawing = false;
    saveMaskSnapshot();
    redrawMainCanvas();
  }

  function cancelDrawing() {
    state.isDrawing = false;
    state.lastPoint = null;
    state.rectStart = state.rectEnd = null;
    state.lassoPoints = [];
  }

  function drawBrushPoint(pt) {
    var ctx = state.maskCtx;
    var r = state.brushSize / 2;
    ctx.globalCompositeOperation = state.maskMode === 'add' ? 'source-over' : 'destination-out';
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }

  function drawBrushLine(a, b) {
    if (!a || !b) return;
    var ctx = state.maskCtx;
    ctx.globalCompositeOperation = state.maskMode === 'add' ? 'source-over' : 'destination-out';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = state.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
  }

  function applyRect() {
    if (!state.rectStart || !state.rectEnd) return;
    var x = Math.min(state.rectStart.x, state.rectEnd.x);
    var y = Math.min(state.rectStart.y, state.rectEnd.y);
    var w = Math.abs(state.rectEnd.x - state.rectStart.x);
    var h = Math.abs(state.rectEnd.y - state.rectStart.y);
    var ctx = state.maskCtx;
    ctx.globalCompositeOperation = state.maskMode === 'add' ? 'source-over' : 'destination-out';
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x, y, w, h);
    ctx.globalCompositeOperation = 'source-over';
  }

  function applyLasso() {
    if (state.lassoPoints.length < 3) return;
    var ctx = state.maskCtx;
    ctx.globalCompositeOperation = state.maskMode === 'add' ? 'source-over' : 'destination-out';
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(state.lassoPoints[0].x, state.lassoPoints[0].y);
    for (var i = 1; i < state.lassoPoints.length; i++) {
      ctx.lineTo(state.lassoPoints[i].x, state.lassoPoints[i].y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }
}

/* ═══════════════════════════════════════════════════════════════
   Coordenadas
   ═══════════════════════════════════════════════════════════════ */
function screenToImage(clientX, clientY) {
  var canvasWrap = $('#canvasWrap');
  if (!canvasWrap) return { x: 0, y: 0 };
  var rect = canvasWrap.getBoundingClientRect();
  return {
    x: (clientX - rect.left - state.offsetX) / state.zoom,
    y: (clientY - rect.top - state.offsetY) / state.zoom
  };
}

function updateTransform() {
  var canvasInner = $('#canvasInner');
  var ftZoom = $('#ftZoom');
  if (canvasInner) {
    canvasInner.style.transform = 'translate(' + state.offsetX + 'px, ' + state.offsetY + 'px) scale(' + state.zoom + ')';
  }
  if (ftZoom) ftZoom.textContent = Math.round(state.zoom * 100) + '%';
}

/* ═══════════════════════════════════════════════════════════════
   Render
   ═══════════════════════════════════════════════════════════════ */
function redrawMainCanvas() {
  var mainCanvas = $('#mainCanvas');
  if (!mainCanvas || !state.imgWidth) return;

  var w = state.imgWidth, h = state.imgHeight;
  if (mainCanvas.width !== w || mainCanvas.height !== h) {
    mainCanvas.width = w;
    mainCanvas.height = h;
  }
  var ctx = mainCanvas.getContext('2d');

  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  var base = state.showOriginal ? state.originalCanvas : state.currentCanvas;
  ctx.drawImage(base, 0, 0);

  var maskImg = state.maskCtx.getImageData(0, 0, w, h);
  var overlay = ctx.createImageData(w, h);
  var hasAny = false;
  for (var i = 0; i < maskImg.data.length; i += 4) {
    if (maskImg.data[i] > 127) {
      overlay.data[i] = 255;
      overlay.data[i + 1] = 105;
      overlay.data[i + 2] = 180;
      overlay.data[i + 3] = 115;
      hasAny = true;
    }
  }
  if (hasAny) {
    var tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').putImageData(overlay, 0, 0);
    ctx.drawImage(tmp, 0, 0);
  }

  if (state.rectStart && state.rectEnd) {
    var x = Math.min(state.rectStart.x, state.rectEnd.x);
    var y = Math.min(state.rectStart.y, state.rectEnd.y);
    var rw = Math.abs(state.rectEnd.x - state.rectStart.x);
    var rh = Math.abs(state.rectEnd.y - state.rectStart.y);
    ctx.save();
    ctx.strokeStyle = '#ff69b4';
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, rw, rh);
    ctx.restore();
  }

  if (state.lassoPoints.length > 1) {
    ctx.save();
    ctx.strokeStyle = '#ff69b4';
    ctx.lineWidth = 2 / state.zoom;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(state.lassoPoints[0].x, state.lassoPoints[0].y);
    for (var j = 1; j < state.lassoPoints.length; j++) {
      ctx.lineTo(state.lassoPoints[j].x, state.lassoPoints[j].y);
    }
    ctx.stroke();
    ctx.restore();
  }
}

/* ═══════════════════════════════════════════════════════════════
   Cursor pincel
   ═══════════════════════════════════════════════════════════════ */
var brushCursorEl = null;
function ensureBrushCursor() {
  if (brushCursorEl) return brushCursorEl;
  brushCursorEl = document.createElement('div');
  brushCursorEl.className = 'brush-cursor';
  document.body.appendChild(brushCursorEl);
  return brushCursorEl;
}
function showBrushCursor() { ensureBrushCursor().style.display = 'block'; }
function hideBrushCursor() { if (brushCursorEl) brushCursorEl.style.display = 'none'; }
function updateBrushCursor() {
  if (!brushCursorEl) return;
  var size = state.brushSize * state.zoom;
  brushCursorEl.style.width = size + 'px';
  brushCursorEl.style.height = size + 'px';
}
function updateBrushCursorPos(e) {
  if (!state.imgWidth) return;
  var el = ensureBrushCursor();
  el.style.display = 'block';
  el.style.left = e.clientX + 'px';
  el.style.top = e.clientY + 'px';
  updateBrushCursor();
}

/* ═══════════════════════════════════════════════════════════════
   Modo máscara
   ═══════════════════════════════════════════════════════════════ */
function setMaskMode(mode) {
  state.maskMode = mode;
  var modeAdd = $('#modeAdd');
  var modeSub = $('#modeSub');
  if (modeAdd) modeAdd.classList.toggle('active', mode === 'add');
  if (modeSub) modeSub.classList.toggle('active', mode === 'sub');
}

/* ═══════════════════════════════════════════════════════════════
   Historial
   ═══════════════════════════════════════════════════════════════ */
function saveMaskSnapshot() {
  if (!state.maskCanvas || !state.imgWidth) return;
  var snapshot = state.maskCtx.getImageData(0, 0, state.imgWidth, state.imgHeight);
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(snapshot);
  if (state.history.length > state.MAX_HISTORY) state.history.shift();
  state.historyIndex = state.history.length - 1;
  updateUndoRedoButtons();
}

function historyUndo() {
  if (state.historyIndex <= 0) return;
  state.historyIndex--;
  state.maskCtx.putImageData(state.history[state.historyIndex], 0, 0);
  redrawMainCanvas();
  updateUndoRedoButtons();
}

function historyRedo() {
  if (state.historyIndex >= state.history.length - 1) return;
  state.historyIndex++;
  state.maskCtx.putImageData(state.history[state.historyIndex], 0, 0);
  redrawMainCanvas();
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  var undoBtn = $('#undoBtn');
  var redoBtn = $('#redoBtn');
  if (undoBtn) undoBtn.disabled = state.historyIndex <= 0;
  if (redoBtn) redoBtn.disabled = state.historyIndex >= state.history.length - 1;
}

/* ═══════════════════════════════════════════════════════════════
   Limpiar máscara
   ═══════════════════════════════════════════════════════════════ */
function clearMask() {
  if (!state.imgWidth) return;
  state.maskCtx.clearRect(0, 0, state.imgWidth, state.imgHeight);
  state.history = [];
  state.historyIndex = -1;
  saveMaskSnapshot();
  redrawMainCanvas();
  updateUndoRedoButtons();
  setStatus('Máscara limpiada');
}

/* ═══════════════════════════════════════════════════════════════
   Carga de imagen
   ═══════════════════════════════════════════════════════════════ */
async function loadImageFile(file) {
  if (!file || !file.type || file.type.indexOf('image/') !== 0) {
    setStatus('Archivo no válido');
    return;
  }
  setStatus('Cargando imagen...');
  state.imgName = file.name || generateFileName();
  state.imgSizeBytes = file.size;
  var fileNameEl = $('#fileName');
  if (fileNameEl) fileNameEl.textContent = state.imgName;

  var buf;
  try { buf = await file.arrayBuffer(); }
  catch (e) { setStatus('Error leyendo archivo'); return; }

  if (file.type === 'image/jpeg' || /\.jpe?g$/i.test(file.name || '')) {
    state.originalJpegBuffer = buf.slice(0);
  } else {
    state.originalJpegBuffer = null;
  }

  var blob = new Blob([buf], { type: file.type });
  var url = URL.createObjectURL(blob);
  var img = new Image();
  img.onload = function () {
    URL.revokeObjectURL(url);
    setupImage(img);
    setStatus('Imagen cargada');
  };
  img.onerror = function () {
    URL.revokeObjectURL(url);
    setStatus('Error al cargar la imagen');
  };
  img.src = url;
}

function setupImage(img) {
  var w = img.naturalWidth, h = img.naturalHeight;
  resizeInternalCanvases(w, h);
  state.originalCtx.clearRect(0, 0, w, h);
  state.currentCtx.clearRect(0, 0, w, h);
  state.maskCtx.clearRect(0, 0, w, h);
  state.originalCtx.drawImage(img, 0, 0);
  state.currentCtx.drawImage(img, 0, 0);

  var mainCanvas = $('#mainCanvas');
  if (mainCanvas) { mainCanvas.width = w; mainCanvas.height = h; }

  var emptyState = $('#emptyState');
  var canvasInner = $('#canvasInner');
  if (emptyState) emptyState.style.display = 'none';
  if (canvasInner) canvasInner.hidden = false;

  state.showOriginal = false;
  state.history = [];
  state.historyIndex = -1;

  redrawMainCanvas();
  fitToViewport();
  saveMaskSnapshot();

  var ftDims = $('#ftDims');
  var ftSize = $('#ftSize');
  if (ftDims) ftDims.textContent = w + ' × ' + h + ' px';
  if (ftSize) ftSize.textContent = formatBytes(state.imgSizeBytes);

  updateUIState();
}

function generateFileName() {
  var d = new Date();
  function pad(n) { return String(n).padStart(2, '0'); }
  return 'foto_' + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '_' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
}

/* ═══════════════════════════════════════════════════════════════
   Vista
   ═══════════════════════════════════════════════════════════════ */
function fitToViewport() {
  if (!state.imgWidth) return;
  var canvasWrap = $('#canvasWrap');
  if (!canvasWrap) return;
  var rect = canvasWrap.getBoundingClientRect();
  var scale = Math.min(rect.width / state.imgWidth, rect.height / state.imgHeight) * 0.95;
  state.zoom = scale;
  state.offsetX = (rect.width - state.imgWidth * scale) / 2;
  state.offsetY = (rect.height - state.imgHeight * scale) / 2;
  updateTransform();
  updateBrushCursor();
}

function zoomAt(cx, cy, newZoom) {
  var canvasWrap = $('#canvasWrap');
  if (!canvasWrap) return;
  var rect = canvasWrap.getBoundingClientRect();
  var x = cx - rect.left;
  var y = cy - rect.top;
  var k = newZoom / state.zoom;
  state.offsetX = x - (x - state.offsetX) * k;
  state.offsetY = y - (y - state.offsetY) * k;
  state.zoom = newZoom;
  updateTransform();
  updateBrushCursor();
}

function toggleOriginal() {
  state.showOriginal = !state.showOriginal;
  redrawMainCanvas();
  setStatus(state.showOriginal ? 'Mostrando original' : 'Mostrando procesado');
}

/* ═══════════════════════════════════════════════════════════════
   Worker
   ═══════════════════════════════════════════════════════════════ */
function startWorker() {
  if (state.worker) return Promise.resolve();
  if (state.workerReady) return Promise.resolve();

  return new Promise(function (resolve, reject) {
    var w;
    try {
      w = new Worker('model-worker.js', { type: 'classic' });
    } catch (e) {
      try { w = new Worker('model-worker.js'); }
      catch (e2) { reject(new Error('No se pudo crear el worker: ' + e2.message)); return; }
    }

    var settled = false;
    var timeout = setTimeout(function () {
      if (settled) return;
      settled = true;
      reject(new Error('Timeout al iniciar el worker'));
    }, 15000);

    w.addEventListener('error', function (err) {
      console.error('[JPB] worker error:', err.message);
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(new Error('Worker error: ' + (err.message || 'desconocido')));
    });

    w.addEventListener('message', onWorkerMessage);

    setTimeout(function () {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      state.worker = w;
      state.workerReady = true;
      resolve();
    }, 500);
  });
}

var _workerReqId = 0;
var _workerPending = new Map();

function workerCall(action, payload, transfer) {
  return new Promise(function (resolve, reject) {
    if (!state.worker) { reject(new Error('Worker no inicializado')); return; }
    var id = ++_workerReqId;
    _workerPending.set(id, { resolve: resolve, reject: reject });
    var msg = { id: id, action: action, payload: payload || {} };
    try {
      state.worker.postMessage(msg, transfer || []);
    } catch (e) {
      _workerPending.delete(id);
      reject(e);
    }
  });
}

function onWorkerMessage(ev) {
  var msg = ev.data;
  if (!msg) return;

  if (typeof msg.id === 'undefined') {
    if (msg.type === 'progress' && msg.stage === 'region') {
      var progressInfo = $('#progressInfo');
      var progressFill = $('#progressFill');
      var dimText = '';
      if (msg.bboxW && msg.bboxH) {
        dimText = ' · Región: ' + msg.bboxW + '×' + msg.bboxH + ' px';
      }
      if (progressInfo) {
        progressInfo.textContent = 'Región ' + msg.current + ' de ' + msg.total + dimText;
      }
      if (progressFill) {
        progressFill.style.width = Math.round(msg.current / msg.total * 100) + '%';
      }
    } else if (msg.type === 'progress' && msg.stage === 'inference') {
      var pi = $('#progressInfo');
      var dims = '';
      if (msg.bboxW && msg.bboxH) {
        dims = ' · Área: ' + msg.bboxW + '×' + msg.bboxH + ' px';
      }
      if (pi) pi.textContent = msg.total + ' región(es) detectada(s)' + dims;
    }
    return;
  }

  var pending = _workerPending.get(msg.id);
  if (!pending) return;
  _workerPending.delete(msg.id);

  if (msg.type === 'error') {
    pending.reject(new Error(msg.error));
  } else if (msg.type === 'result') {
    pending.resolve(msg.result);
  } else {
    pending.resolve(msg);
  }
}

/* ═══════════════════════════════════════════════════════════════
   IA: Remover objetos
   ═══════════════════════════════════════════════════════════════ */
async function runAI() {
  if (!state.imgWidth) { setStatus('Carga una imagen primero'); return; }
  if (state.processing) return;

  var maskData = state.maskCtx.getImageData(0, 0, state.imgWidth, state.imgHeight);
  var hasMask = false;
  for (var i = 0; i < maskData.data.length; i += 4) {
    if (maskData.data[i] > 127) { hasMask = true; break; }
  }
  if (!hasMask) {
    setStatus('Pinta sobre los objetos a eliminar');
    return;
  }

  state.processing = true;
  updateUIState();

  try {
    await requestWakeLock();

    var modelSelect = $('#modelSelect');
    var modelId = (modelSelect && modelSelect.value) || MODELS[0].id;

    if (!window.JPBDB || !window.JPBDB.isAvailable || !window.JPBDB.isAvailable()) {
      throw new Error('IndexedDB no disponible: no se puede leer el modelo');
    }

    var hasIt = await window.JPBDB.hasModel(modelId);

    if (hasIt) {
      var checkBuf = await window.JPBDB.getModel(modelId);
      var validation = isValidOnnxBuffer(checkBuf);
      if (!validation.ok) {
        console.warn('[JPB] Modelo cacheado inválido: ' + validation.reason);
        await window.JPBDB.deleteModel(modelId);
        hasIt = false;
        state.modelLoaded = false;
        state.modelLoadedId = null;
      } else {
        console.log('[JPB] Modelo cacheado válido: ' + checkBuf.byteLength + ' bytes');
      }
    }

    if (!hasIt) {
      setStatus('Descargando modelo IA...');
      var buffer = await downloadModelWithFallback(modelId);

      var dlValidation = isValidOnnxBuffer(buffer);
      if (!dlValidation.ok) {
        throw new Error('El modelo descargado no es válido: ' + dlValidation.reason);
      }

      await window.JPBDB.saveModel(modelId, buffer, {
        name: getModelById(modelId).name,
        url: getModelById(modelId).urls[0]
      });
      state.modelLoaded = false;
      state.modelLoadedId = null;
    }

    if (!state.modelLoaded || state.modelLoadedId !== modelId) {
      setStatus('Iniciando worker...');
      await startWorker();

      setStatus('Cargando modelo en el worker...');
      var modelBuf = await window.JPBDB.getModel(modelId);
      if (!modelBuf) throw new Error('No se pudo leer el modelo de IndexedDB');

      var preValidation = isValidOnnxBuffer(modelBuf);
      if (!preValidation.ok) {
        await window.JPBDB.deleteModel(modelId);
        throw new Error('El modelo en IndexedDB está corrupto. Intenta de nuevo.');
      }

      var bufCopy = modelBuf.slice(0);

      var loadRes = await workerCall('load', {
        buffer: bufCopy,
        modelId: modelId
      });

      state.modelLoaded = true;
      state.modelLoadedId = modelId;
      state.modelProvider = loadRes.provider || 'wasm';
      state.modelInputSize = loadRes.inputSize || null;

      setStatus('Modelo listo (' + state.modelProvider + ')');
    }

    setStatus('Procesando con IA...');
    openProgressModal();

    var imgData = state.currentCtx.getImageData(0, 0, state.imgWidth, state.imgHeight);
    var imgCopy = new ImageData(
      new Uint8ClampedArray(imgData.data),
      imgData.width, imgData.height
    );
    var maskCopy = new ImageData(
      new Uint8ClampedArray(maskData.data),
      maskData.width, maskData.height
    );

    var t0 = performance.now();

    var res = await workerCall('infer', {
      imageData: imgCopy,
      maskData: maskCopy,
      width: state.imgWidth,
      height: state.imgHeight
    });

    var elapsed = (performance.now() - t0) / 1000;

    closeModal($('#progressModal'));

    if (res && res.imageData) {
      state.currentCtx.putImageData(res.imageData, 0, 0);

      // Limpiar la máscara tras remover
      state.maskCtx.clearRect(0, 0, state.imgWidth, state.imgHeight);
      state.history = [];
      state.historyIndex = -1;
      saveMaskSnapshot();

      redrawMainCanvas();
      setStatus('¡Objeto removido! (' + elapsed.toFixed(2) + 's · ' + (res.strategy || 'ok') + ')');
    } else {
      setStatus('Proceso completado en ' + elapsed.toFixed(2) + 's');
    }

  } catch (err) {
    console.error('[JPB runAI]', err);
    closeModal($('#progressModal'));
    setStatus('Error: ' + (err.message || err));
  } finally {
    state.processing = false;
    updateUIState();
    releaseWakeLock();
  }
}

/* ═══════════════════════════════════════════════════════════════
   Descarga del modelo
   ═══════════════════════════════════════════════════════════════ */
async function downloadModelWithFallback(modelId) {
  var model = getModelById(modelId);
  if (!model) throw new Error('Modelo desconocido: ' + modelId);

  var lastErr = null;
  for (var i = 0; i < model.urls.length; i++) {
    try {
      console.log('[JPB] Intentando descargar: ' + model.urls[i]);
      return await downloadWithProgress(model.urls[i], model.name);
    } catch (e) {
      lastErr = e;
      console.warn('[JPB] Falló ' + model.urls[i] + ': ' + (e.message || e));
    }
  }
  throw lastErr || new Error('No se pudo descargar el modelo');
}

async function downloadWithProgress(url, modelName) {
  var progressModal = $('#progressModal');
  var progressTitle = $('#progressTitle');
  var progressFill = $('#progressFill');
  var progressText = $('#progressText');
  var progressInfo = $('#progressInfo');

  openModal(progressModal);
  if (progressTitle) progressTitle.textContent = 'Descargando ' + (modelName || 'modelo');
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) progressText.textContent = '0%';
  if (progressInfo) progressInfo.textContent = 'Conectando...';

  var resp;
  try { resp = await fetch(url, { mode: 'cors', credentials: 'omit' }); }
  catch (e) { closeModal(progressModal); throw new Error('Error de red: ' + e.message); }

  if (!resp.ok) {
    closeModal(progressModal);
    throw new Error('HTTP ' + resp.status);
  }

  var contentType = (resp.headers.get('content-type') || '').toLowerCase();
  if (contentType.indexOf('text/html') >= 0) {
    closeModal(progressModal);
    throw new Error('La URL devolvió HTML, no un ONNX.');
  }

  var total = +(resp.headers.get('content-length') || 0);
  var reader = resp.body.getReader();
  var chunks = [];
  var received = 0;

  while (true) {
    var r = await reader.read();
    if (r.done) break;
    chunks.push(r.value);
    received += r.value.length;
    if (total) {
      var pct = Math.round(received / total * 100);
      if (progressFill) progressFill.style.width = pct + '%';
      if (progressText) progressText.textContent = pct + '%';
      if (progressInfo) progressInfo.textContent = formatBytes(received) + ' / ' + formatBytes(total);
    } else {
      if (progressInfo) progressInfo.textContent = formatBytes(received);
    }
  }

  var buffer = new Uint8Array(received);
  var offset = 0;
  for (var k = 0; k < chunks.length; k++) {
    buffer.set(chunks[k], offset);
    offset += chunks[k].length;
  }
  closeModal(progressModal);
  return buffer.buffer;
}

function openProgressModal() {
  var progressModal = $('#progressModal');
  var progressTitle = $('#progressTitle');
  var progressFill = $('#progressFill');
  var progressText = $('#progressText');
  var progressInfo = $('#progressInfo');

  openModal(progressModal);
  if (progressTitle) progressTitle.textContent = 'Removiendo objetos';
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) progressText.textContent = '';
  if (progressInfo) progressInfo.textContent = 'Analizando máscara...';
}

/* ═══════════════════════════════════════════════════════════════
   Wake Lock
   ═══════════════════════════════════════════════════════════════ */
async function requestWakeLock() {
  if ('wakeLock' in navigator) {
    try { state.wakeLock = await navigator.wakeLock.request('screen'); } catch (e) {}
  }
}
function releaseWakeLock() {
  if (state.wakeLock) {
    state.wakeLock.release().catch(function () {});
    state.wakeLock = null;
  }
}

/* ═══════════════════════════════════════════════════════════════
   Guardar
   ═══════════════════════════════════════════════════════════════ */
function canSave() {
  if (!state.imgWidth) { setStatus('No hay imagen para guardar'); return false; }
  return true;
}

function openSaveModal() {
  var saveModal = $('#saveModal');
  var saveFormat = $('#saveFormat');
  var qualityRow = $('#qualityRow');
  var exifRow = $('#exifRow');
  if (!saveModal) return;
  openModal(saveModal);
  if (saveFormat) {
    var f = saveFormat.value;
    if (qualityRow) qualityRow.classList.toggle('hidden', f !== 'jpeg' && f !== 'webp');
    if (exifRow) exifRow.classList.toggle('hidden', f !== 'jpeg');
  }
}

async function doSave() {
  var saveFormat = $('#saveFormat');
  var qualityRange = $('#qualityRange');
  var keepExif = $('#keepExif');
  var saveModal = $('#saveModal');

  var format = (saveFormat && saveFormat.value) || 'png';
  var quality = qualityRange ? +qualityRange.value / 100 : 0.92;
  var src = state.currentCanvas;
  var mime = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp', bmp: 'image/bmp' }[format] || 'image/png';

  var blob;
  if (format === 'bmp') blob = await canvasToBMP(src);
  else blob = await new Promise(function (res) { src.toBlob(res, mime, quality); });

  if (!blob) { setStatus('Error al exportar'); return; }
  if (format === 'jpeg' && keepExif && keepExif.checked && state.originalJpegBuffer) {
    blob = await injectExif(blob, state.originalJpegBuffer);
  }

  var baseName = (state.imgName || 'imagen').replace(/\.[^.]+$/, '');
  var ext = { png: 'png', jpeg: 'jpg', webp: 'webp', bmp: 'bmp' }[format];
  var suggested = baseName + '_jpb.' + ext;

  if (window.showSaveFilePicker) {
    try {
      var handle = await window.showSaveFilePicker({
        suggestedName: suggested,
        types: [{ description: 'Imagen', accept: {} }]
      });
      var writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      setStatus('Guardado: ' + handle.name);
      closeModal(saveModal);
      return;
    } catch (e) { if (e.name !== 'AbortError') console.warn(e); }
  }

  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = suggested;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);

  closeModal(saveModal);
  setStatus('Descargado: ' + suggested);
}

async function canvasToBMP(canvas) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  var imgData = ctx.getImageData(0, 0, w, h);
  var data = imgData.data;
  var rowSize = Math.floor((24 * w + 31) / 32) * 4;
  var pixelArraySize = rowSize * h;
  var fileSize = 54 + pixelArraySize;
  var buffer = new ArrayBuffer(fileSize);
  var view = new DataView(buffer);

  view.setUint8(0, 0x42); view.setUint8(1, 0x4D);
  view.setUint32(2, fileSize, true);
  view.setUint32(10, 54, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, w, true);
  view.setInt32(22, h, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 24, true);
  view.setUint32(34, pixelArraySize, true);
  view.setInt32(38, 2835, true);
  view.setInt32(42, 2835, true);

  var offset = 54;
  for (var y = h - 1; y >= 0; y--) {
    for (var x = 0; x < w; x++) {
      var i = (y * w + x) * 4;
      view.setUint8(offset++, data[i + 2]);
      view.setUint8(offset++, data[i + 1]);
      view.setUint8(offset++, data[i]);
    }
    for (var p = w * 3; p < rowSize; p++) view.setUint8(offset++, 0);
  }
  return new Blob([buffer], { type: 'image/bmp' });
}

async function injectExif(newJpegBlob, originalBuffer) {
  try {
    var orig = new Uint8Array(originalBuffer);
    if (orig[0] !== 0xFF || orig[1] !== 0xD8) return newJpegBlob;
    var i = 2;
    var app1 = null;
    while (i < orig.length - 1) {
      if (orig[i] !== 0xFF) break;
      var marker = orig[i + 1];
      if (marker === 0xDA) break;
      if (marker === 0xE1) {
        var len = (orig[i + 2] << 8) | orig[i + 3];
        app1 = orig.slice(i, i + 2 + len);
        break;
      }
      var len2 = (orig[i + 2] << 8) | orig[i + 3];
      i += 2 + len2;
    }
    if (!app1) return newJpegBlob;
    var newJpeg = new Uint8Array(await newJpegBlob.arrayBuffer());
    var out = new Uint8Array(2 + app1.length + (newJpeg.length - 2));
    out[0] = 0xFF; out[1] = 0xD8;
    out.set(app1, 2);
    out.set(newJpeg.slice(2), 2 + app1.length);
    return new Blob([out], { type: 'image/jpeg' });
  } catch (e) {
    console.warn('EXIF inject failed', e);
    return newJpegBlob;
  }
}

async function copyToClipboard() {
  if (!state.imgWidth) return;
  try {
    var blob = await new Promise(function (r) { state.currentCanvas.toBlob(r, 'image/png'); });
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    setStatus('Copiado al portapapeles');
  } catch (e) {
    setStatus('No se pudo copiar: ' + e.message);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Modales
   ═══════════════════════════════════════════════════════════════ */
function openModal(m) { if (m) m.hidden = false; }
function closeModal(m) { if (m) m.hidden = true; }

async function openModelsModal() {
  var modelsModal = $('#modelsModal');
  var modelsList = $('#modelsList');
  if (!modelsModal || !modelsList) return;

  openModal(modelsModal);
  modelsList.innerHTML = '<li class="models-empty">Cargando...</li>';

  if (!window.JPBDB || !window.JPBDB.isAvailable || !window.JPBDB.isAvailable()) {
    modelsList.innerHTML = '<li class="models-empty">IndexedDB no disponible en este navegador</li>';
    return;
  }

  var metas = [];
  try { metas = await window.JPBDB.listModels(); } catch (e) { metas = []; }

  var metaMap = {};
  for (var i = 0; i < metas.length; i++) metaMap[metas[i].id] = metas[i];

  modelsList.innerHTML = '';

  MODELS.forEach(function (model) {
    var meta = metaMap[model.id];
    var downloaded = !!meta;

    var li = document.createElement('li');
    li.className = 'model-item';

    var row = document.createElement('div');
    row.className = 'model-row';

    var info = document.createElement('div');
    info.className = 'model-info';

    var name = document.createElement('div');
    name.className = 'model-name';
    name.textContent = model.name;

    var desc = document.createElement('div');
    desc.className = 'model-meta';
    desc.textContent = model.description;

    var metaLine = document.createElement('div');
    metaLine.className = 'model-meta';
    if (downloaded) {
      metaLine.textContent = 'Descargado · ' + formatBytes(meta.size) + ' · ' + new Date(meta.date).toLocaleDateString();
    } else {
      metaLine.textContent = 'Tamaño aprox. ' + model.size;
    }

    info.appendChild(name);
    info.appendChild(desc);
    info.appendChild(metaLine);

    var badge = document.createElement('span');
    badge.className = 'badge ' + (downloaded ? 'badge-downloaded' : 'badge-missing');
    badge.textContent = downloaded ? 'Descargado' : 'No descargado';

    row.appendChild(info);
    row.appendChild(badge);

    var actions = document.createElement('div');
    actions.className = 'model-actions';

    if (downloaded) {
      var delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'mini-btn mini-btn-del';
      delBtn.textContent = 'Eliminar';
      delBtn.addEventListener('click', async function () {
        if (!confirm('¿Eliminar el modelo "' + model.name + '"?')) return;
        try {
          await window.JPBDB.deleteModel(model.id);
          state.modelLoaded = false;
          state.modelLoadedId = null;
          setStatus('Modelo eliminado');
          openModelsModal();
        } catch (e) {
          setStatus('Error al eliminar: ' + e.message);
        }
      });
      actions.appendChild(delBtn);
    } else {
      var dlBtn = document.createElement('button');
      dlBtn.type = 'button';
      dlBtn.className = 'mini-btn mini-btn-dl';
      dlBtn.textContent = 'Descargar';
      dlBtn.addEventListener('click', async function () {
        dlBtn.disabled = true;
        dlBtn.textContent = 'Descargando...';
        badge.className = 'badge badge-downloading';
        badge.textContent = 'Descargando';
        try {
          var buffer = await downloadModelWithFallback(model.id);
          var v = isValidOnnxBuffer(buffer);
          if (!v.ok) throw new Error('Modelo inválido: ' + v.reason);
          await window.JPBDB.saveModel(model.id, buffer, { name: model.name, url: model.urls[0] });
          setStatus('Modelo descargado');
          openModelsModal();
        } catch (e) {
          setStatus('Error: ' + e.message);
          dlBtn.disabled = false;
          dlBtn.textContent = 'Reintentar';
          badge.className = 'badge badge-missing';
          badge.textContent = 'No descargado';
        }
      });
      actions.appendChild(dlBtn);
    }

    li.appendChild(row);
    li.appendChild(actions);
    modelsList.appendChild(li);
  });
}

/* ═══════════════════════════════════════════════════════════════
   Reset
   ═══════════════════════════════════════════════════════════════ */
function resetAll() {
  if (!confirm('¿Reiniciar la aplicación? Se perderán los cambios.')) return;

  state.imgWidth = 0;
  state.imgHeight = 0;
  state.imgName = '';
  state.imgSizeBytes = 0;
  state.originalJpegBuffer = null;
  state.showOriginal = false;
  state.history = [];
  state.historyIndex = -1;

  resizeInternalCanvases(1, 1);

  var mainCanvas = $('#mainCanvas');
  if (mainCanvas) {
    mainCanvas.width = 1; mainCanvas.height = 1;
    mainCanvas.getContext('2d').clearRect(0, 0, 1, 1);
  }
  var emptyState = $('#emptyState');
  var canvasInner = $('#canvasInner');
  if (emptyState) emptyState.style.display = '';
  if (canvasInner) canvasInner.hidden = true;

  var fileNameEl = $('#fileName');
  var ftDims = $('#ftDims');
  var ftSize = $('#ftSize');
  var ftZoom = $('#ftZoom');
  if (fileNameEl) fileNameEl.textContent = 'Sin imagen';
  if (ftDims) ftDims.textContent = '—';
  if (ftSize) ftSize.textContent = '—';
  if (ftZoom) ftZoom.textContent = '100%';

  updateUIState();
  setStatus('Listo');
}

/* ═══════════════════════════════════════════════════════════════
   UI state
   ═══════════════════════════════════════════════════════════════ */
function updateUIState() {
  var hasImage = state.imgWidth > 0;
  var saveBtn = $('#saveBtn');
  var maskApply = $('#maskApply');
  var copyBtn = $('#copyBtn');
  var toggleBtn = $('#toggleBtn');
  var fitBtn = $('#fitBtn');

  if (saveBtn) saveBtn.disabled = !hasImage || state.processing;
  if (maskApply) maskApply.disabled = !hasImage || state.processing;
  if (copyBtn) copyBtn.disabled = !hasImage;
  if (toggleBtn) toggleBtn.disabled = !hasImage;
  if (fitBtn) fitBtn.disabled = !hasImage;

  updateUndoRedoButtons();
}

function setStatus(msg) {
  var ftStatus = $('#ftStatus');
  var aiStatus = $('#aiStatus');
  if (ftStatus) ftStatus.textContent = msg;
  if (aiStatus) aiStatus.textContent = msg;
  console.log('[JPB]', msg);
}

/* ═══════════════════════════════════════════════════════════════
   Utilidades
   ═══════════════════════════════════════════════════════════════ */
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / (1024 * 1024)).toFixed(2) + ' MB';
}

/* ═══════════════════════════════════════════════════════════════
   Drag & drop / Paste
   ═══════════════════════════════════════════════════════════════ */
function setupDragDrop() {
  var canvasWrap = $('#canvasWrap');
  if (!canvasWrap) return;

  ['dragenter', 'dragover'].forEach(function (ev) {
    window.addEventListener(ev, function (e) {
      e.preventDefault();
      canvasWrap.style.outline = '2px dashed var(--primary)';
    });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    window.addEventListener(ev, function (e) {
      e.preventDefault();
      canvasWrap.style.outline = '';
    });
  });
  window.addEventListener('drop', function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadImageFile(f);
  });
}

function handlePaste(e) {
  var items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (var i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image/') === 0) {
      var f = items[i].getAsFile();
      if (f) { loadImageFile(f); break; }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   Atajos
   ═══════════════════════════════════════════════════════════════ */
function bindKeyboard() {
  window.addEventListener('keydown', function (e) {
    var ctrl = e.ctrlKey || e.metaKey;

    if (ctrl && e.key === 'o') { e.preventDefault(); var fi = $('#fileInput'); if (fi) fi.click(); }
    else if (ctrl && e.key === 's') { e.preventDefault(); if (canSave()) openSaveModal(); }
    else if (ctrl && e.key === 'c' && !window.getSelection().toString()) { e.preventDefault(); copyToClipboard(); }
    else if (ctrl && e.key === 'z' && !e.shiftKey) { e.preventDefault(); historyUndo(); }
    else if (ctrl && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) { e.preventDefault(); historyRedo(); }
    else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomAt(window.innerWidth / 2, window.innerHeight / 2, Math.min(20, state.zoom * 1.25)); }
    else if (e.key === '-') { e.preventDefault(); zoomAt(window.innerWidth / 2, window.innerHeight / 2, Math.max(0.05, state.zoom / 1.25)); }
    else if (e.key === '0') { e.preventDefault(); fitToViewport(); }
    else if (e.key === '1') { e.preventDefault(); zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1); }
    else if (e.key === '[') {
      e.preventDefault();
      state.brushSize = Math.max(1, state.brushSize - 5);
      var bs = $('#brushSize'); var bsv = $('#brushSizeVal');
      if (bs) bs.value = state.brushSize;
      if (bsv) bsv.textContent = state.brushSize;
      updateBrushCursor();
    }
    else if (e.key === ']') {
      e.preventDefault();
      state.brushSize = Math.min(100, state.brushSize + 5);
      var bs2 = $('#brushSize'); var bsv2 = $('#brushSizeVal');
      if (bs2) bs2.value = state.brushSize;
      if (bsv2) bsv2.textContent = state.brushSize;
      updateBrushCursor();
    }
    else if (e.key === 'Escape') {
      ['saveModal', 'modelsModal', 'helpModal'].forEach(function (id) {
        var m = document.getElementById(id);
        if (m && !m.hidden) closeModal(m);
      });
    }
  });
}

/* ═══════════════════════════════════════════════════════════════
   Service Worker + Share Target
   ═══════════════════════════════════════════════════════════════ */
function registerSW() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(function (e) { console.warn('SW error', e); });
  }
}

async function checkSharedFiles() {
  var url = new URL(location.href);
  if (url.searchParams.get('shared') === '1') {
    try {
      var cache = await caches.open('jpb-shared');
      var resp = await cache.match('shared-files');
      if (resp) {
        var data = await resp.json();
        if (data.files && data.files.length) {
          var f = data.files[0];
          var bin = atob(f.data);
          var arr = new Uint8Array(bin.length);
          for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          var blob = new Blob([arr], { type: f.type });
          var file = new File([blob], f.name, { type: f.type });
          loadImageFile(file);
        }
        await cache.delete('shared-files');
        history.replaceState({}, '', location.pathname);
      }
    } catch (e) { console.warn('Share target load error', e); }
  }
}

/* ═══════════════════════════════════════════════════════════════
   Arranque
   ═══════════════════════════════════════════════════════════════ */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

window.addEventListener('resize', function () {
  if (state.imgWidth) updateTransform();
});

window.addEventListener('orientationchange', function () {
  var sidebar = $('#sidebar');
  var backdrop = $('#sidebarBackdrop');
  if (sidebar) sidebar.classList.remove('open');
  if (backdrop) backdrop.classList.remove('show');
});