<script lang="ts">
  let {
    label,
    hint = "",
    checked,
    disabled = false,
    onchange,
  }: { label: string; hint?: string; checked: boolean; disabled?: boolean; onchange: (v: boolean) => void } = $props();
</script>

<button class="row" role="switch" aria-checked={checked} {disabled} onclick={() => onchange(!checked)}>
  <span class="text">
    <span class="name">{label}</span>
    {#if hint}<span class="hint">{hint}</span>{/if}
  </span>
  <span class="track" class:on={checked}><span class="knob"></span></span>
</button>

<style>
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--r);
    text-align: left;
  }
  .row:hover:not(:disabled) {
    background: var(--surface-2);
    border-color: transparent;
  }
  .text {
    display: grid;
    gap: 1px;
  }
  .name {
    font-weight: 500;
  }
  .hint {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-3);
    letter-spacing: 0.02em;
  }
  .track {
    flex: none;
    width: 36px;
    height: 20px;
    border-radius: 20px;
    background: var(--meter-off);
    border: 1px solid var(--line-strong);
    position: relative;
    transition: background 0.18s, border-color 0.18s;
  }
  .knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-2);
    transition: transform 0.18s, background 0.18s;
  }
  .track.on {
    background: var(--signal);
    border-color: var(--signal);
  }
  .track.on .knob {
    transform: translateX(16px);
    background: var(--signal-ink);
  }
</style>
