# Componentes de terceros

| Componente | Uso en VoiceLab | Licencia | Origen |
|---|---|---|---|
| MeanVC2 (DiT mean-flow + GTM) | Modelo de conversión de voz (`dit_*.onnx`, `gtm_*.onnx`) y referencia del pipeline streaming | Apache-2.0 | [ASLP-lab/MeanVC2](https://github.com/ASLP-lab/MeanVC2), [HF](https://huggingface.co/ASLP-lab/MeanVC2) |
| Fast-U2++ (WeNet) | Encoder ASR de contenido (`asr_*.onnx`) | Apache-2.0 | incluido en ASLP-lab/MeanVC2 |
| Vocos | Vocoder (`vocos.onnx`) | MIT | incluido en ASLP-lab/MeanVC2 |
| WavLM-Large + ECAPA-TDNN | Encoder de hablante para clonar voces (`spk_encoder.int8.onnx`) | MIT (Microsoft) | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm), [UniSpeech](https://github.com/microsoft/UniSpeech) |
| VCTK Corpus | Audio de referencia de las 8 voces incluidas (`voices/*.vlvoice`) | CC-BY-4.0 | [CSTR, Univ. de Edimburgo](https://datashare.ed.ac.uk/handle/10283/3443) vía [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) |
| ONNX Runtime | Inferencia en CPU | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| Geist, Geist Mono | Tipografías de la interfaz | OFL-1.1 | [vercel/geist-font](https://github.com/vercel/geist-font) |
| VB-Cable | Micrófono virtual (no se incluye; lo instala el usuario) | Donationware | [vb-audio.com](https://vb-audio.com/Cable/) |

Las voces incluidas son huellas de voz (embeddings de 256 valores) calculadas de clips de VCTK;
no contienen el audio original. Atribución: *CSTR VCTK Corpus, University of Edinburgh, CC-BY-4.0*.

Las dependencias de Rust y npm conservan sus propias licencias (MIT/Apache-2.0 en su mayoría);
ver `Cargo.lock` y `app/package-lock.json`.
