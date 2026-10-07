<script lang="ts">
  import type { Api, VoiceInfo } from "./api";
  import Icon from "./Icon.svelte";
  import Portrait from "./Portrait.svelte";
  import { portraitFromFile } from "./portrait";

  let {
    api,
    modelsReady,
    onCreated,
  }: { api: Api; modelsReady: boolean; onCreated: (v: VoiceInfo, photo: string | null) => void } = $props();

  let mode = $state<"file" | "record">("file");
  let file = $state<string | null>(null);
  let name = $state("");
  let description = $state("");
  /** Optional picture, already cropped and shrunk; saved once the voice exists. */
  let photo = $state<string | null>(null);
  let photoPicker: HTMLInputElement;
  let consent = $state(false);
  let seconds = $state(12);
  let busy = $state(false);
  let countdown = $state(0);
  let error = $state<string | null>(null);

  const fileName = $derived(file?.split(/[\\/]/).pop() ?? null);
  const sourceReady = $derived(mode === "record" || file !== null);
  const ready = $derived(modelsReady && consent && name.trim().length > 0 && !busy && sourceReady);

  async function pick() {
    const f = await api.pickAudioFile();
    if (f) {
      file = f;
      if (!name) name = (f.split(/[\\/]/).pop() ?? "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
    }
  }

  async function photoPicked() {
    const f = photoPicker.files?.[0];
    photoPicker.value = "";
    if (!f) return;
    error = null;
    try {
      photo = await portraitFromFile(f);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function create() {
    busy = true;
    error = null;
    let timer: ReturnType<typeof setInterval> | undefined;
    try {
      let v: VoiceInfo;
      if (mode === "file") {
        v = await api.cloneFromFile(file!, name, description);
      } else {
        countdown = seconds;
        timer = setInterval(() => (countdown = Math.max(0, countdown - 1)), 1000);
        v = await api.recordAndClone(seconds, name, description);
      }
      const picture = photo;
      name = "";
      description = "";
      file = null;
      photo = null;
      consent = false;
      onCreated(v, picture);
    } catch (e) {
      error = String(e);
    } finally {
      clearInterval(timer);
      countdown = 0;
      busy = false;
    }
  }
</script>

<section class="page">
  <header>
    <h1>Clonar una voz</h1>
    <p class="sub">
      Con 10–20 segundos de habla limpia, VoiceLab aprende el timbre de una persona. Lo que dices y cómo lo dices
      sigue siendo tuyo; solo cambia a quién suena.
    </p>
  </header>

  <div class="cols">
    <div class="steps">
      <div class="step">
        <div class="num mono" class:done={sourceReady}>01</div>
        <div class="body">
          <h2>Audio de referencia</h2>
          <div class="seg" role="tablist">
            <button role="tab" aria-selected={mode === "file"} class:on={mode === "file"} onclick={() => (mode = "file")}>
              <Icon name="upload" size={15} /> Archivo
            </button>
            <button role="tab" aria-selected={mode === "record"} class:on={mode === "record"} onclick={() => (mode = "record")}>
              <Icon name="mic" size={15} /> Grabarme
            </button>
          </div>
          {#if mode === "file"}
            <button class="drop" onclick={pick} disabled={busy}>
              {#if fileName}
                <span class="file mono">{fileName}</span>
                <span class="hint">Cambiar archivo</span>
              {:else}
                <span class="file">Elegir audio…</span>
                <span class="hint mono">wav · mp3 · flac · ogg</span>
              {/if}
            </button>
          {:else}
            <div class="rec" class:live={countdown > 0}>
              <span class="rec-dot"></span>
              {#if countdown > 0}
                <span class="mono big">{String(countdown).padStart(2, "0")} s</span>
                <span class="hint">Habla con naturalidad…</span>
              {:else}
                <span>Duración</span>
                <input type="range" min="5" max="30" step="1" bind:value={seconds} aria-label="Duración de la grabación" />
                <span class="mono">{seconds} s</span>
              {/if}
            </div>
          {/if}
        </div>
      </div>

      <div class="step">
        <div class="num mono" class:done={name.trim().length > 0}>02</div>
        <div class="body">
          <h2>Nombre</h2>
          <div class="fields">
            <input bind:value={name} maxlength="40" placeholder="Nombre de la voz" aria-label="Nombre" />
            <input bind:value={description} maxlength="60" placeholder="Descripción (opcional)" aria-label="Descripción" />
          </div>
          <div class="photo">
            <div class="thumb"><Portrait src={photo} name={name} lit={photo !== null} /></div>
            <div class="photo-text">
              <span>Foto <span class="opt">(opcional)</span></span>
              <span class="hint">Para reconocerla de un vistazo.</span>
            </div>
            <div class="photo-actions">
              {#if photo}
                <button class="small" onclick={() => (photo = null)} disabled={busy}>Quitar</button>
              {/if}
              <button class="small" onclick={() => photoPicker.click()} disabled={busy}>
                <Icon name="image" size={14} />
                {photo ? "Cambiar" : "Elegir imagen…"}
              </button>
            </div>
            <input class="picker" type="file" accept="image/*" bind:this={photoPicker} onchange={photoPicked} tabindex="-1" aria-hidden="true" />
          </div>
        </div>
      </div>

      <div class="step">
        <div class="num mono" class:done={consent}>03</div>
        <div class="body">
          <h2>Permiso</h2>
          <label class="consent">
            <input type="checkbox" bind:checked={consent} />
            <span>Es mi voz o tengo permiso de la persona. No la usaré para suplantar a nadie ni para engañar.</span>
          </label>
        </div>
      </div>

      {#if !modelsReady}<p class="warn">Primero descarga los modelos en Ajustes.</p>{/if}
      {#if error}<p class="error">{error}</p>{/if}

      <div class="actions">
        <button class="signal create" disabled={!ready} onclick={create}>
          {#if busy}
            {mode === "record" && countdown > 0 ? "Grabando…" : "Analizando la voz…"}
          {:else}
            Crear voz
          {/if}
        </button>
      </div>
    </div>

    <aside class="tips">
      <span class="label">Para que suene bien</span>
      <ul>
        <li>Una sola persona, sin música ni efectos.</li>
        <li>10–20 segundos bastan; más no mejora mucho.</li>
        <li>Micrófono cerca y sala sin eco.</li>
        <li>Habla normal: ni susurros ni gritos.</li>
      </ul>
      <span class="label">Privacidad</span>
      <p>
        La voz se guarda solo en tu PC como un archivo <span class="mono">.vlvoice</span> de 1 KB, con su foto si eliges una.
        Nada se sube a internet.
      </p>
    </aside>
  </div>
</section>

<style>
  .page {
    padding: 26px 28px 40px;
    max-width: 980px;
  }
  h1 {
    font-size: 26px;
  }
  .sub {
    color: var(--text-2);
    margin: 6px 0 24px;
    max-width: 620px;
    line-height: 1.5;
  }
  .cols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 250px;
    gap: 28px;
    align-items: start;
  }
  .steps {
    display: grid;
    gap: 6px;
  }
  .step {
    display: grid;
    grid-template-columns: 40px 1fr;
    gap: 14px;
    padding: 16px 0;
    border-bottom: 1px solid var(--line);
  }
  .num {
    font-size: 13px;
    color: var(--text-3);
    border: 1px solid var(--line-strong);
    border-radius: 8px;
    height: 30px;
    display: grid;
    place-items: center;
  }
  .num.done {
    color: var(--signal-ink);
    background: var(--signal);
    border-color: var(--signal);
  }
  .body {
    display: grid;
    gap: 12px;
  }
  h2 {
    font-size: 15px;
    line-height: 30px;
  }
  .seg {
    display: flex;
    gap: 2px;
    padding: 3px;
    width: fit-content;
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--surface);
  }
  .seg button {
    display: flex;
    align-items: center;
    gap: 6px;
    border: none;
    background: transparent;
    color: var(--text-2);
    padding: 5px 12px;
    border-radius: 7px;
  }
  .seg button.on {
    background: var(--surface-3);
    color: var(--text);
  }
  .drop {
    display: grid;
    gap: 2px;
    justify-items: start;
    padding: 16px;
    border-style: dashed;
    background: var(--surface);
    text-align: left;
  }
  .file {
    font-weight: 500;
  }
  .hint {
    color: var(--text-3);
    font-size: 12px;
  }
  .rec {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--surface);
  }
  .rec input {
    flex: 1;
  }
  .rec-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--bad);
    flex: none;
  }
  .rec.live .rec-dot {
    animation: pulse 1s infinite;
  }
  .big {
    font-size: 18px;
  }
  @keyframes pulse {
    50% {
      box-shadow: 0 0 0 6px color-mix(in srgb, var(--bad) 25%, transparent);
    }
  }
  .fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .photo {
    display: grid;
    grid-template-columns: 64px minmax(0, 1fr) auto;
    align-items: center;
    gap: 14px;
    padding: 8px 10px 8px 8px;
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--surface);
  }
  .thumb {
    aspect-ratio: 4 / 3;
    border-radius: 5px;
  }
  .photo-text {
    display: grid;
    gap: 2px;
  }
  .opt {
    color: var(--text-3);
  }
  .photo-actions {
    display: flex;
    gap: 6px;
  }
  .small {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 12px;
    font-size: 13px;
    white-space: nowrap;
  }
  .picker {
    display: none;
  }
  .consent {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    color: var(--text-2);
    line-height: 1.5;
    cursor: pointer;
  }
  .consent input {
    margin-top: 4px;
  }
  .actions {
    padding: 16px 0 0 54px;
  }
  .create {
    padding: 10px 26px;
  }
  .warn {
    color: var(--warn);
    padding-left: 54px;
  }
  .error {
    color: var(--bad);
    padding-left: 54px;
  }
  .tips {
    display: grid;
    gap: 10px;
    padding: 16px;
    border-radius: var(--r-lg);
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--text-2);
    font-size: 13px;
  }
  .tips ul {
    margin: 0 0 8px;
    padding-left: 16px;
    line-height: 1.7;
  }
  .tips p {
    line-height: 1.5;
  }
</style>
