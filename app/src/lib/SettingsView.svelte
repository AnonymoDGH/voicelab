<script lang="ts">
  import type { Api, DownloadProgress, Overview, Settings } from "./api";
  import { THEMES, applyTheme } from "../theme";
  import Icon from "./Icon.svelte";

  let {
    api,
    overview,
    onSave,
    onModelsReady,
  }: {
    api: Api;
    overview: Overview;
    onSave: (s: Settings) => Promise<void>;
    onModelsReady: () => void;
  } = $props();

  // Local editable copy; saved on each change.
  // svelte-ignore state_referenced_locally
  let s = $state<Settings>({ ...overview.settings });
  let saveError = $state<string | null>(null);
  let progress = $state<DownloadProgress | null>(null);
  let downloadError = $state<string | null>(null);

  const gateOn = $derived(s.gate_db > -100);
  const cable = $derived(overview.virtual_cable);
  const toDb = (g: number) => Math.round(20 * Math.log10(g));
  const fromDb = (db: number) => Math.pow(10, db / 20);
  const signed = (db: number) => `${db > 0 ? "+" : ""}${db} dB`;
  const mb = (b: number) => (b / 1e6).toFixed(0);

  async function save() {
    saveError = null;
    try {
      await onSave($state.snapshot(s));
    } catch (e) {
      saveError = String(e);
    }
  }

  function pickTheme(id: string) {
    s.theme = id;
    applyTheme(id);
    save();
  }

  async function download() {
    downloadError = null;
    progress = { done: 0, total: 1, file: "" };
    try {
      await api.downloadModels((p) => (progress = p));
      onModelsReady();
    } catch (e) {
      downloadError = String(e);
    } finally {
      progress = null;
    }
  }
</script>

<section class="page">
  <h1>Ajustes</h1>

  <div class="section">
    <div class="side">
      <span class="label">Apariencia</span>
      <p>El tema cambia al instante. «Automático» sigue el modo claro u oscuro de Windows.</p>
    </div>
    <div class="themes">
      {#each THEMES as t (t.id)}
        <button class="theme" class:on={s.theme === t.id} onclick={() => pickTheme(t.id)} aria-pressed={s.theme === t.id}>
          {#if t.id === "auto"}
            <span class="swatch split">
              <span class="half" data-theme="papel"><span class="mini"><i></i><b></b></span></span>
              <span class="half" data-theme="estudio"><span class="mini"><i></i><b></b></span></span>
            </span>
          {:else}
            <span class="swatch" data-theme={t.id}><span class="mini"><i></i><b></b><em></em></span></span>
          {/if}
          <span class="tname">{t.name}</span>
          <span class="thint">{t.hint}</span>
        </button>
      {/each}
    </div>
  </div>

  <div class="section">
    <div class="side">
      <span class="label">Audio</span>
      <p>Tu micrófono entra, la voz convertida sale por el micrófono virtual.</p>
    </div>
    <div class="fields">
      <label class="field">
        <span>Micrófono</span>
        <select bind:value={s.input} onchange={save}>
          <option value={null}>Predeterminado del sistema</option>
          {#each overview.inputs as d (d.id)}
            <option value={d.id}>{d.name}</option>
          {/each}
        </select>
      </label>
      <label class="field">
        <span>Salida</span>
        <select bind:value={s.output} onchange={save}>
          <option value={null}>{cable ? `Automática · ${cable.name}` : "Automática (VB-Cable no instalado)"}</option>
          {#each overview.outputs as d (d.id)}
            <option value={d.id}>{d.name}{d.is_virtual_cable ? " · micrófono virtual" : ""}</option>
          {/each}
        </select>
      </label>
      <div class="note" class:ok={cable}>
        <Icon name={cable ? "check" : "alert"} size={15} />
        {#if cable}
          <span>Listo. En Discord, OBS o tu juego elige <b>CABLE Output</b> como micrófono.</span>
        {:else}
          <span>Instala el micrófono virtual gratuito VB-Cable para usar tu voz en otras apps.</span>
          <button onclick={() => api.openUrl(overview.vb_cable_url)}>Descargar <Icon name="external" size={13} /></button>
        {/if}
      </div>
      <label class="field">
        <span>Escucharme por</span>
        <select bind:value={s.monitor} onchange={save}>
          <option value={null}>Salida predeterminada</option>
          {#each overview.outputs.filter((d) => !d.is_virtual_cable) as d (d.id)}
            <option value={d.id}>{d.name}</option>
          {/each}
        </select>
      </label>
    </div>
  </div>

  <div class="section">
    <div class="side">
      <span class="label">Motor</span>
      <p>Rápido responde antes; Calidad pronuncia mejor y gasta la mitad de CPU.</p>
    </div>
    <div class="fields">
      <div class="modes">
        <button class="mode" class:on={s.variant === "40ms"} onclick={() => ((s.variant = "40ms"), save())} aria-pressed={s.variant === "40ms"}>
          <span class="mname">Rápido</span>
          <span class="mono mstat">~225 ms · CPU ×2</span>
          <span class="mdesc">Para hablar en directo</span>
        </button>
        <button class="mode" class:on={s.variant === "120ms"} onclick={() => ((s.variant = "120ms"), save())} aria-pressed={s.variant === "120ms"}>
          <span class="mname">Calidad</span>
          <span class="mono mstat">~305 ms · CPU ×1</span>
          <span class="mdesc">PCs justos, grabaciones</span>
        </button>
      </div>
      <label class="field">
        <span>Hilos de CPU</span>
        <select bind:value={s.threads} onchange={save}>
          {#each [1, 2, 3, 4] as n (n)}
            <option value={n}>{n}{n === 1 ? " · recomendado" : ""}</option>
          {/each}
        </select>
      </label>
    </div>
  </div>

  <div class="section">
    <div class="side">
      <span class="label">Sonido</span>
      <p>La puerta silencia la salida cuando no hablas, para que el ruido de fondo no se convierta.</p>
    </div>
    <div class="fields">
      <div class="slider">
        <label class="check">
          <input type="checkbox" checked={gateOn} onchange={(e) => ((s.gate_db = e.currentTarget.checked ? -50 : -100), save())} />
          Puerta de ruido
        </label>
        <input type="range" min="-80" max="-20" step="1" value={gateOn ? s.gate_db : -50} disabled={!gateOn}
          oninput={(e) => (s.gate_db = +e.currentTarget.value)} onchange={save} aria-label="Umbral de la puerta" />
        <span class="mono val">{gateOn ? `${s.gate_db} dB` : "apagada"}</span>
      </div>
      <div class="slider">
        <span>Ganancia de entrada</span>
        <input type="range" min="-12" max="12" step="1" value={toDb(s.input_gain)}
          oninput={(e) => (s.input_gain = fromDb(+e.currentTarget.value))} onchange={save} aria-label="Ganancia de entrada" />
        <span class="mono val">{signed(toDb(s.input_gain))}</span>
      </div>
      <div class="slider">
        <span>Ganancia de salida</span>
        <input type="range" min="-12" max="12" step="1" value={toDb(s.output_gain)}
          oninput={(e) => (s.output_gain = fromDb(+e.currentTarget.value))} onchange={save} aria-label="Ganancia de salida" />
        <span class="mono val">{signed(toDb(s.output_gain))}</span>
      </div>
      <label class="field">
        <span>Atajo global · voz IA / tu voz</span>
        <input class="mono" bind:value={s.hotkey} onchange={save} placeholder="CommandOrControl+Alt+V" />
      </label>
    </div>
  </div>

  <div class="section">
    <div class="side">
      <span class="label">Modelos</span>
      <p>Se descargan una sola vez y todo funciona sin conexión.</p>
    </div>
    <div class="fields">
      {#if overview.models_ready}
        <div class="note ok"><Icon name="check" size={15} /><span>Instalados en <span class="mono path">{overview.models_dir}</span></span></div>
      {:else if progress}
        <div class="progress"><div style="width:{(100 * progress.done) / Math.max(progress.total, 1)}%"></div></div>
        <span class="mono small">{mb(progress.done)} / {mb(progress.total)} MB · {progress.file}</span>
      {:else}
        <p class="text2">~700 MB desde <span class="mono">{overview.model_repo}</span></p>
        <button class="signal dl" onclick={download}><Icon name="download" size={15} /> Descargar modelos</button>
      {/if}
      {#if downloadError}
        <p class="error">{downloadError}</p>
        <p class="text2 small">También puedes copiarlos a mano en <span class="mono">{overview.models_dir}</span>.</p>
      {/if}
    </div>
  </div>

  {#if saveError}<p class="error">{saveError}</p>{/if}
</section>

<style>
  .page {
    padding: 26px 28px 48px;
    max-width: 980px;
  }
  h1 {
    font-size: 26px;
    margin-bottom: 8px;
  }
  .section {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr);
    gap: 28px;
    padding: 22px 0;
    border-bottom: 1px solid var(--line);
  }
  .side p {
    color: var(--text-2);
    font-size: 12.5px;
    line-height: 1.5;
    margin-top: 6px;
  }
  .fields {
    display: grid;
    gap: 12px;
  }
  .field {
    display: grid;
    gap: 6px;
  }
  .field > span,
  .slider > span:first-child,
  .check {
    font-size: 13px;
    color: var(--text-2);
  }
  .themes {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
  }
  .theme {
    display: grid;
    align-content: start;
    gap: 4px;
    padding: 8px 8px 10px;
    text-align: left;
    background: var(--surface);
    border-radius: var(--r-lg);
  }
  .theme.on {
    border-color: var(--signal);
    box-shadow: inset 0 0 0 1px var(--signal);
  }
  .swatch {
    display: block;
    height: 64px;
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg);
    border: 1px solid var(--line);
    margin-bottom: 6px;
  }
  .swatch.split {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
  .half {
    background: var(--bg);
  }
  .mini {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    padding: 9px;
    height: 100%;
  }
  .mini i {
    grid-row: span 2;
    border-radius: 5px;
    background: var(--surface);
    border: 1px solid var(--line);
  }
  .mini b {
    border-radius: 4px;
    background: var(--signal);
  }
  .mini em {
    border-radius: 4px;
    background: var(--text-3);
    opacity: 0.6;
  }
  .split .mini {
    grid-template-columns: 1fr;
  }
  .split .mini i {
    grid-row: auto;
  }
  .tname {
    font-weight: 600;
    padding: 0 2px;
  }
  .thint {
    font-size: 11.5px;
    color: var(--text-3);
    padding: 0 2px;
    line-height: 1.35;
  }
  .note {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: var(--r);
    font-size: 13px;
    color: var(--warn);
    border: 1px solid color-mix(in srgb, var(--warn) 40%, transparent);
    background: color-mix(in srgb, var(--warn) 8%, transparent);
  }
  .note.ok {
    color: var(--ok);
    border-color: color-mix(in srgb, var(--ok) 35%, transparent);
    background: color-mix(in srgb, var(--ok) 7%, transparent);
  }
  .note span {
    color: var(--text);
  }
  .note button {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    white-space: nowrap;
  }
  .modes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .mode {
    display: grid;
    gap: 3px;
    text-align: left;
    padding: 12px 14px;
    background: var(--surface);
    border-radius: var(--r-lg);
  }
  .mode.on {
    border-color: var(--signal);
    box-shadow: inset 0 0 0 1px var(--signal);
    background: var(--signal-soft);
  }
  .mname {
    font-weight: 600;
  }
  .mstat {
    font-size: 11.5px;
    color: var(--text-2);
  }
  .mdesc {
    font-size: 12px;
    color: var(--text-3);
  }
  .slider {
    display: grid;
    grid-template-columns: 170px 1fr 70px;
    align-items: center;
    gap: 12px;
  }
  .check {
    display: flex;
    gap: 8px;
    align-items: center;
    cursor: pointer;
  }
  .val {
    text-align: right;
    font-size: 12px;
    color: var(--text-2);
  }
  .progress {
    height: 6px;
    border-radius: 6px;
    background: var(--meter-off);
    overflow: hidden;
  }
  .progress div {
    height: 100%;
    background: var(--signal);
    transition: width 0.2s;
  }
  .dl {
    display: flex;
    align-items: center;
    gap: 8px;
    width: fit-content;
  }
  .text2 {
    color: var(--text-2);
  }
  .small {
    font-size: 12px;
  }
  .path {
    font-size: 12px;
    word-break: break-all;
  }
  .error {
    color: var(--bad);
  }
</style>
