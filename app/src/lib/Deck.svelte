<script lang="ts">
  import type { LiveInfo, LiveStatus, Settings, VoiceInfo } from "./api";
  import LedMeter from "./LedMeter.svelte";
  import Portrait from "./Portrait.svelte";
  import Switch from "./Switch.svelte";
  import VoicePrint from "./VoicePrint.svelte";

  let {
    running,
    starting,
    live,
    info,
    voice,
    settings,
    onPower,
    onEnabled,
    onMonitor,
    onMuted,
  }: {
    running: boolean;
    starting: boolean;
    live: LiveStatus | null;
    info: LiveInfo | null;
    voice: VoiceInfo | null;
    settings: Settings;
    onPower: () => void;
    onEnabled: (v: boolean) => void;
    onMonitor: (v: boolean) => void;
    onMuted: (v: boolean) => void;
  } = $props();

  const enabled = $derived(live?.enabled ?? true);
  const muted = $derived(live?.muted ?? false);
  const stats = $derived(live?.stats ?? null);
  const onAir = $derived(running && enabled && !muted);
  const hotkey = $derived(settings.hotkey.replace("CommandOrControl", "Ctrl"));
  const cpu = $derived(stats ? Math.round(stats.load * 100) : null);
</script>

<aside class="deck">
  <button class="power" class:live={running} disabled={starting} onclick={onPower} aria-pressed={running}>
    <span class="lamp" class:on={running} class:blink={starting}></span>
    <span class="power-text">
      <span class="state mono">{starting ? "Cargando" : running ? "En vivo" : "Apagado"}</span>
      <span class="sub">
        {#if starting}
          Preparando el modelo…
        {:else if running}
          Pulsa para detener
        {:else}
          Pulsa para empezar
        {/if}
      </span>
    </span>
  </button>

  <div class="lcd">
    <div class="lcd-head">
      <span class="label">Voz</span>
      {#if running}
        <span class="badge mono" class:off={!enabled}>{enabled ? "IA" : "Original"}</span>
      {/if}
    </div>
    <div class="ident">
      <div class="pic">
        <Portrait src={voice?.portrait ?? null} name={voice?.name ?? ""} lit={onAir} />
      </div>
      <div class="who">
        <div class="voice-name">{voice?.name ?? "Sin voz"}</div>
        <div class="voice-desc">{voice?.description || "Elige una voz de la biblioteca"}</div>
      </div>
    </div>
    <div class="print">
      <VoicePrint print={voice?.print ?? []} height={26} lit={onAir} animate={onAir && (stats?.gate_open ?? false)} />
    </div>
    <div class="readouts mono">
      <div>
        <span class="label">Latencia</span>
        <span class="value">{#if running && info}{Math.round(info.latency_ms)}<small> ms</small>{:else}—{/if}</span>
      </div>
      <div>
        <span class="label">CPU</span>
        <span class="value" class:warn={cpu !== null && cpu > 70} class:bad={cpu !== null && cpu > 95}>
          {cpu ?? "—"}<small>{cpu !== null ? " %" : ""}</small>
        </span>
      </div>
      <div>
        <span class="label">Modo</span>
        <span class="value">{settings.variant === "40ms" ? "Rápido" : "Calidad"}</span>
      </div>
    </div>
  </div>

  <div class="meters">
    <LedMeter label="Ent" peak={stats?.input_peak ?? 0} active={running} />
    <LedMeter label="Sal" peak={stats?.output_peak ?? 0} active={running} />
    <div class="scale mono" aria-hidden="true">
      <span>0</span><span>-12</span><span>-24</span><span>-36</span><span>-48</span>
    </div>
    <div class="gate">
      <span class="dot" class:open={running && (stats?.gate_open ?? false)}></span>
      <span class="label">Puerta</span>
    </div>
  </div>

  {#if running && stats?.input_silent}
    <div class="mic-warn" role="alert">
      <strong>Tu micrófono no envía sonido</strong>
      <span>Elige otro en <b>Ajustes → Audio</b>, o activa el acceso en Windows: Configuración → Privacidad → Micrófono.</span>
    </div>
  {/if}

  <div class="switches">
    <Switch label="Voz IA" hint={hotkey} checked={enabled} disabled={!running} onchange={onEnabled} />
    <Switch label="Escucharme" hint="por tus auriculares" checked={settings.monitor_enabled} onchange={onMonitor} />
    <Switch label="Silenciar salida" checked={muted} disabled={!running} onchange={onMuted} />
  </div>
</aside>

<style>
  .deck {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 14px;
    background: var(--surface);
    border-right: 1px solid var(--line);
    min-height: 0;
    overflow-y: auto;
  }
  .power {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 11px 16px;
    border-radius: var(--r-lg);
    text-align: left;
    background: var(--surface-2);
    border: 1px solid var(--line-strong);
  }
  .power.live {
    background: var(--signal);
    border-color: var(--signal);
    color: var(--signal-ink);
    box-shadow: var(--glow);
  }
  .power.live:hover:not(:disabled) {
    background: var(--signal);
    filter: brightness(1.05);
  }
  .lamp {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex: none;
    background: var(--meter-off);
    border: 2px solid var(--line-strong);
  }
  .lamp.on {
    background: var(--signal-ink);
    border-color: var(--signal-ink);
    box-shadow: 0 0 0 3px rgb(255 255 255 / 0.25);
  }
  .lamp.blink {
    animation: blink 0.8s steps(2) infinite;
    background: var(--signal);
  }
  @keyframes blink {
    50% {
      opacity: 0.2;
    }
  }
  .power-text {
    display: grid;
  }
  .state {
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .sub {
    font-size: 12px;
    opacity: 0.75;
  }
  .lcd {
    background: var(--lcd);
    color: var(--lcd-text);
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
    padding: 12px 14px;
    display: grid;
    gap: 3px;
  }
  .lcd .label {
    color: var(--lcd-text-2);
  }
  .lcd-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .badge {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--signal);
    color: var(--signal-ink);
  }
  .badge.off {
    background: transparent;
    color: var(--lcd-text-2);
    border: 1px solid var(--lcd-text-2);
  }
  .ident {
    display: grid;
    grid-template-columns: 72px minmax(0, 1fr);
    align-items: center;
    gap: 12px;
    margin-top: 6px;
  }
  .pic {
    aspect-ratio: 4 / 3;
    border-radius: 7px;
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--lcd-text-2) 30%, transparent);
  }
  .who {
    min-width: 0;
  }
  .voice-name {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .voice-desc {
    font-size: 12px;
    color: var(--lcd-text-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .print {
    margin: 8px 0 4px;
  }
  .readouts {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border-top: 1px dashed color-mix(in srgb, var(--lcd-text-2) 40%, transparent);
    padding-top: 10px;
    gap: 6px;
  }
  .readouts div {
    display: grid;
    gap: 2px;
  }
  .value {
    font-size: 15px;
    font-weight: 500;
  }
  .value small {
    font-size: 10.5px;
    color: var(--lcd-text-2);
  }
  .value.warn {
    color: var(--warn);
  }
  .value.bad {
    color: var(--bad);
  }
  .meters {
    display: grid;
    grid-template-columns: auto auto auto 1fr;
    align-items: start;
    gap: 10px;
    padding: 10px 14px;
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
  }
  .scale {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 109px;
    font-size: 9.5px;
    color: var(--text-3);
  }
  .gate {
    justify-self: end;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--meter-off);
  }
  .dot.open {
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
  }
  .mic-warn {
    display: grid;
    gap: 3px;
    padding: 10px 12px;
    border-radius: var(--r);
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-2);
    border: 1px solid color-mix(in srgb, var(--warn) 45%, transparent);
    background: color-mix(in srgb, var(--warn) 9%, transparent);
  }
  .mic-warn strong {
    color: var(--warn);
    font-size: 13px;
  }
  .switches {
    display: grid;
    gap: 2px;
    margin: 0 -4px;
  }
</style>
