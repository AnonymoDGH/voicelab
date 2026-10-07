<script lang="ts">
  import type { Effect } from "./api";

  let {
    effect,
    pitch,
    onEffect,
    onPitch,
  }: {
    effect: Effect;
    pitch: number;
    onEffect: (e: Effect) => void;
    /** `commit` is false while dragging (apply live, save on release). */
    onPitch: (semitones: number, commit: boolean) => void;
  } = $props();

  const effects: { id: Effect; label: string }[] = [
    { id: "none", label: "Limpio" },
    { id: "robot", label: "Robot" },
    { id: "radio", label: "Radio" },
    { id: "demon", label: "Demonio" },
    { id: "chipmunk", label: "Ardilla" },
    { id: "echo", label: "Eco" },
    { id: "cave", label: "Cueva" },
  ];
  const signed = (n: number) => (n > 0 ? `+${n}` : `${n}`);
</script>

<div class="fx">
  <div class="head">
    <span class="label">Efectos</span>
    <span class="mono hint">se suman a la voz IA</span>
  </div>
  <div class="pads" role="radiogroup" aria-label="Efecto">
    {#each effects as e (e.id)}
      <button class="pad mono" class:on={effect === e.id} role="radio" aria-checked={effect === e.id} onclick={() => onEffect(e.id)}>
        {e.label}
      </button>
    {/each}
  </div>
  <div class="pitch">
    <span class="label">Tono</span>
    <input
      type="range"
      min="-12"
      max="12"
      step="1"
      value={pitch}
      oninput={(e) => onPitch(+e.currentTarget.value, false)}
      onchange={(e) => onPitch(+e.currentTarget.value, true)}
      aria-label="Tono en semitonos"
    />
    <button class="mono val" class:moved={pitch !== 0} title="Volver a 0" onclick={() => onPitch(0, true)}>
      {signed(pitch)}<small> st</small>
    </button>
  </div>
</div>

<style>
  .fx {
    display: grid;
    gap: 8px;
    padding: 10px 12px;
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .hint {
    font-size: 10.5px;
    color: var(--text-3);
  }
  .pads {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
  }
  .pad {
    padding: 6px 2px;
    font-size: 10px;
    letter-spacing: 0.02em;
    text-align: center;
    text-transform: uppercase;
    color: var(--text-2);
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    white-space: nowrap;
    overflow: hidden;
  }
  .pad:hover {
    color: var(--text);
    border-color: var(--line-strong);
  }
  .pad.on {
    color: var(--signal-ink);
    border-color: var(--signal);
    background: var(--signal);
  }
  .pitch {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 10px;
  }
  .val {
    min-width: 52px;
    padding: 3px 6px;
    font-size: 13px;
    text-align: right;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-2);
  }
  .val.moved {
    color: var(--signal);
  }
  .val small {
    font-size: 10px;
    color: var(--text-3);
  }
</style>
