<script lang="ts">
  import type { VoiceInfo } from "./api";
  import Icon from "./Icon.svelte";
  import VoicePrint from "./VoicePrint.svelte";

  let {
    voices,
    selected,
    live,
    onSelect,
    onDelete,
    onClone,
  }: {
    voices: VoiceInfo[];
    selected: string | null;
    live: boolean;
    onSelect: (id: string) => void;
    onDelete: (v: VoiceInfo) => void;
    onClone: () => void;
  } = $props();

  type Filter = "all" | "mine" | "builtin";
  let filter = $state<Filter>("all");
  let query = $state("");

  // User voices first: they are the ones people switch between most.
  const ordered = $derived([...voices.filter((v) => !v.builtin), ...voices.filter((v) => v.builtin)]);
  const norm = (s: string) => s.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase();
  const shown = $derived(
    ordered
      .map((v, i) => ({ v, slot: i + 1 }))
      .filter(({ v }) => filter === "all" || (filter === "mine") === !v.builtin)
      .filter(({ v }) => norm(`${v.name} ${v.description}`).includes(norm(query.trim()))),
  );

  // Keys 1–9 pick the voice in that slot (when not typing in a field).
  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement;
    if (t.tagName === "INPUT" || t.tagName === "SELECT" || e.ctrlKey || e.altKey || e.metaKey) return;
    const n = Number(e.key);
    if (n >= 1 && n <= 9 && ordered[n - 1]) onSelect(ordered[n - 1].id);
  }

  const filters: [Filter, string][] = [
    ["all", "Todas"],
    ["mine", "Mías"],
    ["builtin", "Incluidas"],
  ];
</script>

<svelte:window onkeydown={onKey} />

<section class="page">
  <header class="head">
    <div>
      <h1>Voces</h1>
      <p class="sub">Elige una voz; puedes cambiarla mientras hablas. Teclas <kbd>1</kbd>–<kbd>9</kbd> para las primeras.</p>
    </div>
    <div class="tools">
      <div class="seg" role="tablist" aria-label="Filtrar voces">
        {#each filters as [id, label] (id)}
          <button role="tab" aria-selected={filter === id} class:on={filter === id} onclick={() => (filter = id)}>{label}</button>
        {/each}
      </div>
      <label class="search">
        <Icon name="search" size={15} />
        <input placeholder="Buscar" bind:value={query} aria-label="Buscar voz" />
      </label>
    </div>
  </header>

  <div class="grid">
    {#each shown as { v, slot } (v.id)}
      {@const active = v.id === selected}
      <div
        class="tile"
        class:active
        role="button"
        tabindex="0"
        aria-pressed={active}
        onclick={() => onSelect(v.id)}
        onkeydown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(v.id))}
        title={v.source ? `Origen: ${v.source}${v.license ? ` · ${v.license}` : ""}` : v.name}
      >
        <div class="tile-head">
          <span class="slot mono">{String(slot).padStart(2, "0")}</span>
          {#if active}
            <span class="tag on mono">{live ? "En uso" : "Elegida"}</span>
          {:else if !v.builtin}
            <span class="tag mono">Tuya</span>
          {/if}
        </div>
        <VoicePrint print={v.print} height={34} lit={active} />
        <div class="tile-foot">
          <div class="name">{v.name}</div>
          <div class="desc">{v.description || (v.builtin ? "Voz incluida" : "Voz clonada")}</div>
        </div>
        {#if !v.builtin}
          <button
            class="del"
            aria-label="Borrar {v.name}"
            title="Borrar voz"
            onclick={(e) => {
              e.stopPropagation();
              onDelete(v);
            }}
          >
            <Icon name="trash" size={14} />
          </button>
        {/if}
      </div>
    {/each}

    {#if filter !== "builtin" && !query}
      <button class="tile new" onclick={onClone}>
        <span class="plus"><Icon name="plus" size={18} /></span>
        <span class="name">Clonar una voz</span>
        <span class="desc">Desde un audio o grabándote</span>
      </button>
    {/if}
  </div>

  {#if !shown.length && query}
    <p class="empty">Ninguna voz coincide con «{query}».</p>
  {/if}
</section>

<style>
  .page {
    padding: 26px 28px 40px;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  h1 {
    font-size: 26px;
  }
  .sub {
    color: var(--text-2);
    margin-top: 4px;
  }
  kbd {
    font-family: var(--mono);
    font-size: 11px;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid var(--line-strong);
    background: var(--surface-2);
  }
  .tools {
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .seg {
    display: flex;
    padding: 3px;
    gap: 2px;
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--surface);
  }
  .seg button {
    border: none;
    background: transparent;
    color: var(--text-2);
    padding: 5px 12px;
    border-radius: 7px;
    font-size: 13px;
  }
  .seg button.on {
    background: var(--surface-3);
    color: var(--text);
  }
  .search {
    position: relative;
    width: 190px;
    color: var(--text-3);
  }
  .search :global(svg) {
    position: absolute;
    left: 11px;
    top: 50%;
    transform: translateY(-50%);
  }
  .search input {
    padding: 7px 10px 7px 32px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(178px, 1fr));
    gap: 12px;
  }
  .tile {
    position: relative;
    display: grid;
    gap: 12px;
    padding: 12px 14px 14px;
    border-radius: var(--r-lg);
    border: 1px solid var(--line);
    background: var(--surface);
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
  }
  .tile:hover {
    background: var(--surface-2);
    border-color: var(--line-strong);
  }
  .tile.active {
    border-color: var(--signal-line);
    background: var(--signal-soft);
    box-shadow: inset 0 0 0 1px var(--signal-line);
  }
  .tile-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 18px;
  }
  .slot {
    font-size: 12px;
    color: var(--text-3);
    letter-spacing: 0.06em;
  }
  .active .slot {
    color: var(--signal);
  }
  .tag {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--text-2);
    border: 1px solid var(--line-strong);
  }
  .tag.on {
    background: var(--signal);
    border-color: var(--signal);
    color: var(--signal-ink);
  }
  .name {
    font-weight: 600;
    font-size: 15px;
  }
  .desc {
    color: var(--text-2);
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .del {
    position: absolute;
    right: 8px;
    bottom: 8px;
    padding: 6px;
    border: none;
    background: transparent;
    color: var(--text-3);
    opacity: 0;
  }
  .tile:hover .del,
  .del:focus-visible {
    opacity: 1;
  }
  .del:hover:not(:disabled) {
    color: var(--bad);
    background: transparent;
  }
  .new {
    border-style: dashed;
    background: transparent;
    align-content: center;
    justify-items: start;
    gap: 4px;
    min-height: 132px;
  }
  .plus {
    width: 30px;
    height: 30px;
    display: grid;
    place-items: center;
    border-radius: 8px;
    color: var(--signal);
    border: 1px solid var(--signal-line);
    margin-bottom: 8px;
  }
  .empty {
    color: var(--text-2);
    margin-top: 24px;
  }
</style>
