<script lang="ts">
  import type { LiveInfo, LiveStatus, Overview } from "./api";

  let { overview, info, live }: { overview: Overview; info: LiveInfo | null; live: LiveStatus | null } = $props();

  const stats = $derived(live?.stats ?? null);
  const cable = $derived(overview.virtual_cable);
</script>

<footer class="status mono">
  {#if info && live?.running}
    <span title="Micrófono">Ent · {info.input}</span>
    <span title="Salida">Sal · {info.output}</span>
    <span>{(info.output_rate / 1000).toFixed(1)} kHz</span>
    <span>bloque {Math.round(info.block_ms)} ms</span>
    <span>búfer {Math.round(stats?.output_buffer_ms ?? 0)} ms</span>
    <span class:warn={(stats?.underruns ?? 0) > 0}>cortes {stats?.underruns ?? 0}</span>
  {:else}
    <span>Detenido</span>
    <span class:ok={!!cable} class:warn={!cable}>{cable ? `Micrófono virtual: ${cable.name}` : "VB-Cable no instalado"}</span>
    <span>{overview.models_ready ? "Modelos listos" : "Modelos sin descargar"}</span>
  {/if}
</footer>

<style>
  .status {
    display: flex;
    gap: 18px;
    align-items: center;
    padding: 0 16px;
    height: 28px;
    font-size: 11px;
    color: var(--text-3);
    background: var(--surface);
    border-top: 1px solid var(--line);
    white-space: nowrap;
    overflow: hidden;
  }
  span {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  span:first-child,
  span:nth-child(2) {
    max-width: 260px;
  }
  .warn {
    color: var(--warn);
  }
  .ok {
    color: var(--ok);
  }
</style>
