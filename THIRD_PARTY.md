# Componentes de terceros

| Componente | Uso en VoiceLab | Licencia | Origen |
|---|---|---|---|
| MeanVC2 (DiT mean-flow + GTM) | Modelo de conversión de voz (`dit_*.onnx`, `gtm_*.onnx`) y referencia del pipeline streaming | Apache-2.0 | [ASLP-lab/MeanVC2](https://github.com/ASLP-lab/MeanVC2), [HF](https://huggingface.co/ASLP-lab/MeanVC2) |
| Fast-U2++ (WeNet) | Encoder ASR de contenido (`asr_*.onnx`) | Apache-2.0 | incluido en ASLP-lab/MeanVC2 |
| Vocos | Vocoder (`vocos.onnx`) | MIT | incluido en ASLP-lab/MeanVC2 |
| WavLM-Large + ECAPA-TDNN | Encoder de hablante para clonar voces (`spk_encoder.int8.onnx`) | MIT (Microsoft) | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm), [UniSpeech](https://github.com/microsoft/UniSpeech) |
| VCTK Corpus | Audio de referencia de 8 voces incluidas (Carlos, Diego, Elena, Jorge, Lucía, Marta, Pablo, Sofía) | CC-BY-4.0 | [CSTR, Univ. de Edimburgo](https://datashare.ed.ac.uk/handle/10283/3443) vía [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) |
| Unmute Voice Donation Project | Audio de referencia de 7 voces incluidas (Valeria, Mateo, Rafael, Iván, Álvaro, Clara, Irene), donadas por voluntarios para clonación | CC0-1.0 | [Kyutai](https://unmute.sh/voice-donation) vía [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) |
| Grabaciones propias de Kyutai para unmute.sh | Audio de referencia de la voz Noelia | CC0-1.0 | [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) `unmute-prod-website/` |
| AP-BWE | Extensión de banda 16 → 48 kHz del modo Ultra (`bwe_16k_48k.int8.onnx`) | MIT | [yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) |
| RNNoise (nnnoiseless) | Reducción de ruido del micrófono | BSD-3-Clause | [xiph/rnnoise](https://github.com/xiph/rnnoise), [nnnoiseless](https://github.com/jneem/nnnoiseless) |
| ONNX Runtime | Inferencia en CPU | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| Notionists (Zoish) | Ilustraciones de los retratos de las voces incluidas (`voices/*.svg`) | CC0-1.0 | [heyzoish.gumroad.com](https://heyzoish.gumroad.com/l/notionists) |
| DiceBear | Genera esos retratos (`npm run portraits`; solo desarrollo, no se distribuye) | MIT | [dicebear/dicebear](https://github.com/dicebear/dicebear) |
| Geist, Geist Mono | Tipografías de la interfaz | OFL-1.1 | [vercel/geist-font](https://github.com/vercel/geist-font) |
| VB-Cable | Micrófono virtual (no se incluye; lo instala el usuario) | Donationware | [vb-audio.com](https://vb-audio.com/Cable/) |

Las voces incluidas son huellas de voz (embeddings de 256 valores) calculadas de clips de VCTK y
de donaciones de voz CC0; no contienen el audio original. Atribución: *CSTR VCTK Corpus,
University of Edinburgh, CC-BY-4.0*; *Unmute Voice Donation Project (Volhejn, 2025), CC0*.
Los nombres de las voces son de personaje, no los de quienes las grabaron.

Sus retratos son personajes ficticios, no las personas que grabaron las voces: ilustraciones del estilo
*Notionists* de Zoish (dominio público, CC0 1.0) combinadas con DiceBear.

Las dependencias de Rust y npm conservan sus propias licencias (MIT/Apache-2.0 en su mayoría);
ver `Cargo.lock` y `app/package-lock.json`.
