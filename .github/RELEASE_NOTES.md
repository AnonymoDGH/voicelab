Cambiador de voz con IA en tiempo real, solo con la CPU.

## Instalar

1. Instala [VB-Cable](https://vb-audio.com/Cable/) (gratis) y reinicia Windows.
2. Descarga **`VoiceLab_*_x64-setup.exe`** de abajo y ejecútalo.
   Windows puede avisar con SmartScreen porque el instalador aún no está firmado:
   pulsa *Más información → Ejecutar de todas formas*.
3. Abre VoiceLab: la primera vez descarga los modelos de IA (~800 MB) desde este mismo release.
   Si vienes de la 0.1, solo baja lo nuevo (~90 MB).
4. Elige una voz, pulsa **EN VIVO** y en Discord/OBS elige **CABLE Output** como micrófono.

## Archivos

| Archivo | Qué es |
|---|---|
| `VoiceLab_*_x64-setup.exe` | Instalador de la app (Windows 10/11, x64) |
| `voicelab-cli-windows-x64.zip` | Línea de comandos `voicelab` con las voces incluidas |
| `*.onnx`, `manifest.json` | Modelos que la app descarga sola; no hace falta bajarlos a mano |

Con la línea de comandos: `voicelab download-models`, después `voicelab bench` y `voicelab run --voice carlos`.

## Cambios en esta versión

- **Modo Ultra (48 kHz).** El modelo da 4 pasos en vez de 2 y una segunda red (AP-BWE) reconstruye
  los agudos de 8 a 24 kHz: sibilantes y aire, lo más realista. Pide 2 hilos de CPU; añade ~70 ms.
  Elígelo en **Ajustes → Motor**.
- **Efectos:** Robot, Radio, Demonio, Ardilla, Eco y Cueva, sobre la voz IA o la tuya, y **tono**
  de -12 a +12 semitonos. Todo en vivo desde el panel de la izquierda.
- **Reducir ruido:** RNNoise limpia el micrófono (teclado, ventilador) antes del modelo.
- **Cambia de voz desde cualquier app** con `Ctrl+Alt+1`…`9` (se puede desactivar en Ajustes).
- **Retratos:** cada voz tiene su personaje ilustrado; pon la foto que quieras a cualquier voz con
  el botón de la cámara de su tarjeta, o al clonarla.
- **8 voces nuevas** donadas por voluntarios (CC0): Valeria, Mateo, Rafael, Iván, Álvaro, Clara,
  Irene y Noelia. Ya son 16.
- **Menos latencia:** en los silencios la salida descarta el búfer que no hizo falta.
- Consola rediseñada para que todo quepa: teclas con piloto en lugar de interruptores.

## Qué incluye VoiceLab

- Conversión de voz en streaming con MeanVC2 sobre ONNX Runtime: modos Rápido (~225 ms), Calidad (~305 ms)
  y Ultra (~375 ms, 48 kHz).
- Clonado de voz desde un archivo o grabándote (10–20 s).
- 16 voces incluidas con retrato, efectos, reducción de ruido, seis temas y atajos globales.

Si algo falla, abre un issue con la salida de `voicelab devices` y `voicelab bench`.
