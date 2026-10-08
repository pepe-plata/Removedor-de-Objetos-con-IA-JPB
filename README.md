# 🎨 Removedor de Objetos con IA JPB

PWA para **remover objetos** de imágenes usando IA ONNX **100% local**.

## ✅ Últimos ajustes

- ❌ Eliminada la sección "Fondo del Lienzo" del sidebar.
- ✅ Secciones laterales expandidas por defecto.
- ✅ Editor de máscara **siempre activo** (sin botón "Cancelar" que lo desactive).
- ✅ Al pulsar "Remover Objetos": si el modelo **no está descargado, se descarga automáticamente**.
- ✅ Modal "Gestionar modelos" muestra estado (Descargado / No descargado), botón **Descargar** y **Eliminar**.
- ✅ Corregido "IndexedDB no disponible": `db.js` ya no usa `export` (funciona con `<script src>` normal).
- ✅ Corregido botón "Seleccionar imagen" del estado vacío (listener robusto + `defer`).

## 📁 Estructura
