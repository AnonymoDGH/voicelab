Cambiador de voz con IA en tiempo real, solo con la CPU.

## Instalar

1. Instala [VB-Cable](https://vb-audio.com/Cable/) (gratis) y reinicia Windows.
2. Descarga **`VoiceLab_*_x64-setup.exe`** de abajo y ejecútalo.
   Windows puede avisar con SmartScreen porque el instalador aún no está firmado:
   pulsa *Más información → Ejecutar de todas formas*.
3. Abre VoiceLab: la primera vez descarga los modelos de IA (~700 MB) desde este mismo release.
4. Elige una voz, pulsa **EN VIVO** y en Discord/OBS elige **CABLE Output** como micrófono.

## Archivos

| Archivo | Qué es |
|---|---|
| `VoiceLab_*_x64-setup.exe` | Instalador de la app (Windows 10/11, x64) |
| `voicelab-cli-windows-x64.zip` | Línea de comandos `voicelab` con las voces incluidas |
| `*.onnx`, `manifest.json` | Modelos que la app descarga sola; no hace falta bajarlos a mano |

Con la línea de comandos: `voicelab download-models`, después `voicelab bench` y `voicelab run --voice carlos`.

## Cambios en esta versión

- Los microcortes que informa Windows ya no salen como error rojo: se cuentan en la barra de estado.
- Aviso cuando el micrófono elegido no envía sonido (micro equivocado, apagado o bloqueado por la
  privacidad de Windows), con qué revisar.
- La lectura de CPU ya no parpadea a 0 %.
- Se prefiere «CABLE Input» (estéreo) frente a «CABLE In 16ch» al elegir el micrófono virtual.
- Las descargas usan los certificados de Windows (funciona tras proxies corporativos o antivirus
  que inspeccionan HTTPS).

## Qué incluye VoiceLab

- Conversión de voz en streaming con MeanVC2 sobre ONNX Runtime: modo Rápido (~225 ms) y Calidad (~305 ms).
- Clonado de voz desde un archivo o grabándote (10–20 s).
- 8 voces incluidas, seis temas y atajo global `Ctrl+Alt+V`.

Si algo falla, abre un issue con la salida de `voicelab devices` y `voicelab bench`.
