/* ═══════════════════════════════════════════════════════════════
   db.js — Wrapper de IndexedDB para cachear modelos ONNX
   DB_NAME: 'Obj-remover-db', DB_VERSION: 2
   Stores: 'models' (ArrayBuffer), 'models-meta' (metadata)
   SIN módulos: todo se expone en window.JPBDB.
   ═══════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  const DB_NAME = 'Obj-remover-db';
  const DB_VERSION = 2;
  const STORE_MODELS = 'models';
  const STORE_META = 'models-meta';

  let _dbPromise = null;

  function isAvailable() {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  }

  function openDB() {
    if (!isAvailable()) {
      return Promise.reject(new Error('IndexedDB no está disponible en este navegador'));
    }
    if (_dbPromise) return _dbPromise;

    _dbPromise = new Promise((resolve, reject) => {
      let req;
      try {
        req = indexedDB.open(DB_NAME, DB_VERSION);
      } catch (e) {
        reject(e);
        return;
      }

      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_MODELS)) {
          db.createObjectStore(STORE_MODELS);
        }
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: 'id' });
        }
      };

      req.onsuccess = () => {
        const db = req.result;
        // Manejo de cierre inesperado
        db.onclose = () => { _dbPromise = null; };
        resolve(db);
      };
      req.onerror = () => {
        _dbPromise = null;
        reject(req.error || new Error('No se pudo abrir IndexedDB'));
      };
      req.onblocked = () => {
        console.warn('IndexedDB bloqueado por otra pestaña');
      };
    });
    return _dbPromise;
  }

  /** Guarda el ArrayBuffer de un modelo + sus metadatos. */
  async function saveModel(id, arrayBuffer, meta) {
    meta = meta || {};
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction([STORE_MODELS, STORE_META], 'readwrite');
      t.objectStore(STORE_MODELS).put(arrayBuffer, id);
      t.objectStore(STORE_META).put({
        id: id,
        name: meta.name || id,
        size: arrayBuffer.byteLength,
        url: meta.url || '',
        date: Date.now()
      });
      t.oncomplete = () => resolve(true);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }

  /** Recupera el ArrayBuffer de un modelo. Devuelve null si no existe. */
  async function getModel(id) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE_MODELS, 'readonly');
      const req = t.objectStore(STORE_MODELS).get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  /** Recupera los metadatos de un modelo concreto. */
  async function getModelMeta(id) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE_META, 'readonly');
      const req = t.objectStore(STORE_META).get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  /** Recupera todos los metadatos de modelos guardados. */
  async function listModels() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE_META, 'readonly');
      const req = t.objectStore(STORE_META).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  /** Elimina un modelo y sus metadatos. */
  async function deleteModel(id) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction([STORE_MODELS, STORE_META], 'readwrite');
      t.objectStore(STORE_MODELS).delete(id);
      t.objectStore(STORE_META).delete(id);
      t.oncomplete = () => resolve(true);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }

  /** Devuelve true si el modelo existe en la DB. */
  async function hasModel(id) {
    try {
      const m = await getModel(id);
      return !!m;
    } catch (e) {
      return false;
    }
  }

  /** Pide persistencia al navegador. */
  async function requestPersistence() {
    if (navigator.storage && navigator.storage.persist) {
      try { return await navigator.storage.persist(); }
      catch (e) { return false; }
    }
    return false;
  }

  /** Estima el espacio usado / disponible. */
  async function estimateStorage() {
    if (navigator.storage && navigator.storage.estimate) {
      try { return await navigator.storage.estimate(); }
      catch (e) { return null; }
    }
    return null;
  }

  // Exponer al ámbito global
  window.JPBDB = {
    isAvailable: isAvailable,
    saveModel: saveModel,
    getModel: getModel,
    getModelMeta: getModelMeta,
    listModels: listModels,
    deleteModel: deleteModel,
    hasModel: hasModel,
    requestPersistence: requestPersistence,
    estimateStorage: estimateStorage
  };

  console.log('[JPBDB] IndexedDB wrapper listo:', isAvailable());
})();