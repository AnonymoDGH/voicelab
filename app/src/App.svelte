<script lang="ts">
  import { onMount } from "svelte";
  import { api as apiPromise, type Api, type LiveInfo, type LiveStatus, type Overview, type Settings, type VoiceInfo } from "./lib/api";
  import Clone from "./lib/Clone.svelte";
  import Deck from "./lib/Deck.svelte";
  import Icon from "./lib/Icon.svelte";
  import SettingsView from "./lib/SettingsView.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import Voices from "./lib/Voices.svelte";
  import { portraitFromFile } from "./lib/portrait";
  import { applyTheme } from "./theme";

  type View = "voices" | "clone" | "settings";

  let api = $state<Api | null>(null);
  let ov = $state<Overview | null>(null);
  let live = $state<LiveStatus | null>(null);
  let info = $state<LiveInfo | null>(null);
  let view = $state<View>("voices");
  let starting = $state(false);
  let toast = $state<{ text: string; bad: boolean } | null>(null);
  let toastTimer: ReturnType<typeof setTimeout> | undefined;

  const running = $derived(live?.running ?? ov?.running ?? false);
  const voice = $derived(ov?.voices.find((v) => v.id === ov?.settings.voice) ?? null);

  function notify(text: string, bad = false) {
    toast = { text, bad };
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), bad ? 6000 : 2800);
  }

  async function refresh() {
    ov = await api!.overview();
    info = ov.info;
  }

  let lastError: string | null = null;
  async function poll() {
    if (!api) return;
    live = await api.liveStatus();
    if (live.error && live.error !== lastError) notify(live.error, true);
    lastError = live.error;
  }

  onMount(() => {
    let stop = false;
    let unlisten: (() => void) | undefined;
    (async () => {
      api = await apiPromise;
      await refresh();
      applyTheme(ov!.settings.theme);
      if (!ov!.models_ready) view = "settings";
      unlisten = await api.onEnabledChanged(() => poll());
      const loop = async () => {
        if (stop) return;
        await poll();
        setTimeout(loop, running ? 70 : 600);
      };
      loop();
    })();
    return () => {
      stop = true;
      unlisten?.();
    };
  });

  async function power() {
    if (!api || !ov) return;
    if (running) {
      await api.stop();
      info = null;
      await poll();
      return;
    }
    if (!ov.models_ready) {
      view = "settings";
      return notify("Descarga los modelos para empezar.", true);
    }
    if (!ov.settings.voice) {
      view = "voices";
      return notify("Elige una voz primero.", true);
    }
    starting = true;
    try {
      info = await api.start();
      await poll();
      if (!info.output_is_virtual_cable) notify("La salida no es un micrófono virtual: instala VB-Cable para usarla en Discord u OBS.");
    } catch (e) {
      notify(String(e), true);
    } finally {
      starting = false;
    }
  }

  async function selectVoice(id: string) {
    if (!api || !ov || ov.settings.voice === id) return;
    await api.setVoice(id);
    ov.settings.voice = id;
    const v = ov.voices.find((x) => x.id === id);
    if (v && running) notify(`Ahora suenas como ${v.name}`);
  }

  async function deleteVoice(v: VoiceInfo) {
    if (!confirm(`¿Borrar la voz «${v.name}»? No se puede deshacer.`)) return;
    try {
      await api!.deleteVoice(v.id);
      await refresh();
    } catch (e) {
      notify(String(e), true);
    }
  }

  async function setPortrait(v: VoiceInfo, file: File | null) {
    try {
      if (file) await api!.setVoicePortrait(v.id, await portraitFromFile(file));
      else await api!.clearVoicePortrait(v.id);
      await refresh();
    } catch (e) {
      notify(e instanceof Error ? e.message : String(e), true);
    }
  }

  async function saveSettings(s: Settings) {
    await api!.updateSettings(s);
    ov!.settings = s;
    if (running) info = (await api!.overview()).info;
  }

  async function voiceCreated(v: VoiceInfo, photo: string | null) {
    // The voice exists already: if its picture fails to save, it can be set again from its tile.
    const photoError = photo ? await api!.setVoicePortrait(v.id, photo).then(() => null, String) : null;
    await refresh();
    await selectVoice(v.id);
    view = "voices";
    if (photoError) notify(`Voz «${v.name}» creada, pero no se guardó la foto: ${photoError}`, true);
    else notify(`Voz «${v.name}» creada`);
  }

  async function setFlag(flags: { enabled?: boolean; muted?: boolean; monitor?: boolean }) {
    await api!.setFlags(flags);
    if (flags.monitor !== undefined) ov!.settings.monitor_enabled = flags.monitor;
    await poll();
  }

  const tabs: { id: View; label: string }[] = [
    { id: "voices", label: "Voces" },
    { id: "clone", label: "Clonar" },
    { id: "settings", label: "Ajustes" },
  ];
</script>

{#if api && ov}
  <div class="shell">
    <header class="top">
      <div class="brand">
        <span class="mark" class:live={running}></span>
        <span class="word">voicelab</span>
      </div>
      <nav aria-label="Secciones">
        {#each tabs as t (t.id)}
          <button class:on={view === t.id} aria-current={view === t.id ? "page" : undefined} onclick={() => (view = t.id)}>
            {t.label}
            {#if t.id === "settings" && !ov.models_ready}<span class="pip"></span>{/if}
          </button>
        {/each}
      </nav>
    </header>

    <div class="body">
      <Deck
        {running}
        {starting}
        {live}
        {info}
        {voice}
        settings={ov.settings}
        onPower={power}
        onEnabled={(v) => setFlag({ enabled: v })}
        onMonitor={(v) => setFlag({ monitor: v })}
        onMuted={(v) => setFlag({ muted: v })}
      />

      <main>
        {#if !ov.models_ready && view !== "settings"}
          <div class="banner">
            <Icon name="download" size={15} />
            <span>Falta descargar los modelos de IA (una sola vez, ~700 MB).</span>
            <button onclick={() => (view = "settings")}>Ir a Ajustes</button>
          </div>
        {:else if !ov.virtual_cable && view === "voices"}
          <div class="banner">
            <Icon name="alert" size={15} />
            <span>Para usar tu voz en Discord, OBS o juegos instala el micrófono virtual gratuito VB-Cable.</span>
            <button onclick={() => api!.openUrl(ov!.vb_cable_url)}>Descargar</button>
          </div>
        {/if}

        {#if view === "voices"}
          <Voices
            voices={ov.voices}
            selected={ov.settings.voice}
            live={running}
            onSelect={selectVoice}
            onDelete={deleteVoice}
            onPortrait={setPortrait}
            onClone={() => (view = "clone")}
          />
        {:else if view === "clone"}
          <Clone {api} modelsReady={ov.models_ready} onCreated={voiceCreated} />
        {:else}
          <SettingsView {api} overview={ov} onSave={saveSettings} onModelsReady={() => refresh().then(() => notify("Modelos listos"))} />
        {/if}
      </main>
    </div>

    <StatusBar overview={ov} {info} {live} />
  </div>

  {#if toast}
    <div class="toast" class:bad={toast.bad} role="status">{toast.text}</div>
  {/if}
{/if}

<style>
  .shell {
    display: grid;
    grid-template-rows: 52px 1fr 28px;
    height: 100vh;
  }
  .top {
    display: flex;
    align-items: center;
    gap: 32px;
    padding: 0 16px 0 20px;
    border-bottom: 1px solid var(--line);
    background: var(--surface);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
    width: 264px;
  }
  .mark {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    background: var(--text);
    transition: background 0.2s, box-shadow 0.2s;
  }
  .mark.live {
    background: var(--signal);
    box-shadow: var(--glow);
  }
  .word {
    font-family: var(--mono);
    font-weight: 600;
    font-size: 15px;
    letter-spacing: -0.02em;
  }
  nav {
    display: flex;
    gap: 4px;
    height: 100%;
  }
  nav button {
    position: relative;
    border: none;
    border-radius: 0;
    background: transparent;
    color: var(--text-2);
    padding: 0 14px;
    height: 100%;
    font-weight: 500;
    border-bottom: 2px solid transparent;
  }
  nav button:hover:not(:disabled) {
    background: transparent;
    color: var(--text);
  }
  nav button.on {
    color: var(--text);
    border-bottom-color: var(--signal);
  }
  .pip {
    position: absolute;
    top: 14px;
    right: 6px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--warn);
  }
  .body {
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    min-height: 0;
  }
  main {
    overflow-y: auto;
    min-width: 0;
  }
  .banner {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 16px 28px 0;
    padding: 10px 12px 10px 14px;
    border-radius: var(--r);
    border: 1px solid var(--line-strong);
    background: var(--surface);
    font-size: 13px;
    color: var(--text-2);
  }
  .banner :global(svg) {
    color: var(--signal);
    flex: none;
  }
  .banner button {
    margin-left: auto;
    padding: 5px 12px;
    white-space: nowrap;
  }
  .toast {
    position: fixed;
    bottom: 44px;
    left: calc(50% + 150px);
    transform: translateX(-50%);
    padding: 10px 16px;
    border-radius: var(--r);
    background: var(--text);
    color: var(--bg);
    font-weight: 500;
    box-shadow: var(--shadow);
    max-width: 60vw;
  }
  .toast.bad {
    background: var(--bad);
    color: #fff;
  }
</style>
