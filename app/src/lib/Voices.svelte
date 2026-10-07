<script lang="ts">
  import type { VoiceInfo } from "./api";
  import Icon from "./Icon.svelte";
  import Portrait from "./Portrait.svelte";

  let {
    voices,
    selected,
    live,
    onSelect,
    onDelete,
    onPortrait,
    onClone,
  }: {
    voices: VoiceInfo[];
    selected: string | null;
    live: boolean;
    onSelect: (id: string) => void;
    onDelete: (v: VoiceInfo) => void;
    /** A picked image file to use as the voice's picture, or null to remove the user's own. */
    onPortrait: (v: VoiceInfo, file: File | null) => void;
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
    if (e.key === "Escape") menu = null;
    const t = e.target as HTMLElement;
    if (t.tagName === "INPUT" || t.tagName === "SELECT" || e.ctrlKey || e.altKey || e.metaKey) return;
    const n = Number(e.key);
    if (n >= 1 && n <= 9 && ordered[n - 1]) onSelect(ordered[n - 1].id);
  }

  // Picture menu of one tile, and the hidden file input it opens.
  let menu = $state<string | null>(null);
  let picker: HTMLInputElement;
  let pickingFor: VoiceInfo | null = null;

  function choosePhoto(v: VoiceInfo) {
    menu = null;
    pickingFor = v;
    picker.value = "";
    picker.click();
  }

  function picked() {
    const file = picker.files?.[0];
    if (file && pickingFor) onPortrait(pickingFor, file);
    pickingFor = null;
  }

  const filters: [Filter, string][] = [
    ["all", "Todas"],
    ["mine", "Mías"],
    ["builtin", "Incluidas"],
  ];
</script>

<svelte:window onkeydown={onKey} onclick={() => (menu = null)} />

<input class="picker" type="file" accept="image/*" bind:this={picker} onchange={picked} tabindex="-1" aria-hidden="true" />

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
        onkeydown={(e) =>
          e.target === e.currentTarget && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(v.id))}
        title={v.source ? `Origen: ${v.source}${v.license ? ` · ${v.license}` : ""}` : v.name}
      >
        <div class="screen">
          <Portrait src={v.portrait} name={v.name} lit={active} />
          <span class="slot mono">{String(slot).padStart(2, "0")}</span>
          {#if active}
            <span class="tag on mono">{live ? "En uso" : "Elegida"}</span>
          {:else if !v.builtin}
            <span class="tag mono">Tuya</span>
          {/if}
          <button
            class="photo"
            class:open={menu === v.id}
            aria-label="Foto de {v.name}"
            aria-haspopup="menu"
            aria-expanded={menu === v.id}
            title="Cambiar foto"
            onclick={(e) => {
              e.stopPropagation();
              menu = menu === v.id ? null : v.id;
            }}
          >
            <Icon name="camera" size={14} />
          </button>
          {#if menu === v.id}
            <!-- Its items are buttons; the click handler only keeps clicks from reaching the tile. -->
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <div class="menu" role="menu" tabindex="-1" onclick={(e) => e.stopPropagation()}>
              <button role="menuitem" onclick={() => choosePhoto(v)}>
                <Icon name="image" size={14} /> Cambiar foto…
              </button>
              {#if v.custom_portrait}
                <button
                  role="menuitem"
                  onclick={() => {
                    menu = null;
                    onPortrait(v, null);
                  }}
                >
                  <Icon name="close" size={14} /> Quitar foto
                </button>
              {/if}
            </div>
          {/if}
        </div>
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
        <span class="screen"><span class="plus"><Icon name="plus" size={18} /></span></span>
        <span class="tile-foot">
          <span class="name">Clonar una voz</span>
          <span class="desc">Desde un audio o grabándote</span>
        </span>
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
    align-content: start;
    gap: 11px;
    padding: 7px 7px 13px;
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
  /* The picture, framed like a monitor with on-screen labels. */
  .screen {
    position: relative;
    aspect-ratio: 4 / 3;
    border-radius: calc(var(--r-lg) - 7px);
    box-shadow: 0 0 0 1px var(--line);
  }
  .tile:hover .screen {
    --portrait-ink: color-mix(in srgb, var(--lcd-text-2) 55%, var(--lcd-text));
  }
  .active .screen {
    box-shadow: 0 0 0 1px var(--signal-line);
  }
  /* Keeps the on-screen labels readable over light pictures. */
  .screen::after {
    content: "";
    position: absolute;
    inset: 0 0 auto;
    z-index: 2;
    height: 34%;
    border-radius: inherit;
    background: linear-gradient(rgb(0 0 0 / 0.55), transparent);
    pointer-events: none;
  }
  .new .screen::after {
    content: none;
  }
  .slot {
    position: absolute;
    z-index: 3;
    left: 9px;
    top: 7px;
    font-size: 11.5px;
    color: var(--lcd-text-2);
    letter-spacing: 0.06em;
  }
  .active .slot {
    color: var(--signal);
  }
  .tag {
    position: absolute;
    z-index: 3;
    right: 7px;
    top: 6px;
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--lcd-text-2);
    border: 1px solid color-mix(in srgb, var(--lcd-text-2) 50%, transparent);
  }
  .tag.on {
    background: var(--signal);
    border-color: var(--signal);
    color: var(--signal-ink);
  }
  .photo {
    position: absolute;
    z-index: 3;
    right: 6px;
    bottom: 6px;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border-radius: 7px;
    color: var(--lcd-text);
    background: color-mix(in srgb, var(--lcd) 75%, transparent);
    border: 1px solid color-mix(in srgb, var(--lcd-text-2) 45%, transparent);
    opacity: 0;
  }
  .tile:hover .photo,
  .photo:focus-visible,
  .photo.open {
    opacity: 1;
  }
  .photo:hover:not(:disabled),
  .photo.open {
    background: var(--lcd);
    border-color: var(--lcd-text-2);
    color: var(--signal);
  }
  .menu {
    position: absolute;
    z-index: 10;
    right: 0;
    top: calc(100% + 6px);
    display: grid;
    gap: 2px;
    min-width: 170px;
    padding: 4px;
    border-radius: var(--r);
    border: 1px solid var(--line-strong);
    background: var(--surface-2);
    box-shadow: var(--shadow);
    cursor: default;
  }
  .menu button {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 10px;
    border: none;
    border-radius: 7px;
    background: transparent;
    font-size: 13px;
    text-align: left;
    white-space: nowrap;
  }
  .menu button:hover:not(:disabled) {
    background: var(--surface-3);
  }
  .menu :global(svg) {
    color: var(--text-3);
  }
  .picker {
    display: none;
  }
  .tile-foot {
    display: grid;
    gap: 1px;
    min-width: 0;
    padding: 0 7px;
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
  }
  .new .screen {
    display: grid;
    place-items: center;
    box-shadow: none;
    border: 1px dashed var(--line-strong);
  }
  .plus {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    color: var(--signal);
    border: 1px solid var(--signal-line);
  }
  .empty {
    color: var(--text-2);
    margin-top: 24px;
  }
</style>
