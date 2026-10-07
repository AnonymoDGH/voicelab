<script lang="ts">
  // A latching console key: lamp + label, lit when on.
  let {
    label,
    hint = "",
    checked,
    disabled = false,
    onchange,
  }: { label: string; hint?: string; checked: boolean; disabled?: boolean; onchange: (v: boolean) => void } = $props();
</script>

<button class="key" class:on={checked} role="switch" aria-checked={checked} {disabled} onclick={() => onchange(!checked)}>
  <span class="lamp"></span>
  <span class="name">{label}</span>
  {#if hint}<span class="hint">{hint}</span>{/if}
</button>

<style>
  .key {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
    align-content: center;
    column-gap: 8px;
    row-gap: 2px;
    min-height: 46px;
    padding: 8px 10px;
    text-align: left;
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: var(--r);
  }
  .key:hover:not(:disabled) {
    border-color: var(--line-strong);
  }
  .lamp {
    grid-row: 1 / 3;
    align-self: center;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--meter-off);
    border: 1px solid var(--line-strong);
    transition: background 0.15s, box-shadow 0.15s;
  }
  .name {
    font-size: 12.5px;
    font-weight: 500;
    line-height: 1.2;
  }
  .hint {
    grid-column: 2;
    font-family: var(--mono);
    font-size: 9.5px;
    color: var(--text-3);
    letter-spacing: 0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .key.on {
    border-color: var(--signal-line);
    background: var(--signal-soft);
  }
  .key.on .lamp {
    background: var(--signal);
    border-color: var(--signal);
    box-shadow: 0 0 8px var(--signal);
  }
</style>
