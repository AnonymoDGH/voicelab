<script lang="ts">
  import { untrack } from "svelte";

  // Vertical segmented level meter (dBFS, -48..0) with peak hold, like a console channel strip.
  let { label, peak = 0, active = true }: { label: string; peak?: number; active?: boolean } = $props();

  const SEGMENTS = 16;
  const FLOOR_DB = -48;
  const level = (p: number) => Math.min(1, Math.max(0, (20 * Math.log10(Math.max(p, 1e-6)) - FLOOR_DB) / -FLOOR_DB));

  let lit = $state(0);
  let hold = $state(0);
  let holdAge = 0;

  $effect(() => {
    const target = Math.round(level(active ? peak : 0) * SEGMENTS);
    untrack(() => {
      lit = target >= lit ? target : Math.max(target, lit - 1);
      if (target >= hold) {
        hold = target;
        holdAge = 0;
      } else if (++holdAge > 12) {
        hold = Math.max(target, hold - 1);
      }
    });
  });

  // Segment i (0 = bottom) color zone: top 2 red, next 3 amber, rest green.
  const zone = (i: number) => (i >= SEGMENTS - 2 ? "bad" : i >= SEGMENTS - 5 ? "warn" : "ok");
</script>

<div class="strip" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={SEGMENTS} aria-valuenow={lit}>
  <div class="leds">
    {#each { length: SEGMENTS } as _, k}
      {@const i = SEGMENTS - 1 - k}
      <span class="led {zone(i)}" class:on={i < lit} class:hold={i === hold - 1 && hold > lit}></span>
    {/each}
  </div>
  <span class="label">{label}</span>
</div>

<style>
  .strip {
    display: grid;
    justify-items: center;
    gap: 8px;
  }
  .leds {
    display: grid;
    gap: 3px;
    width: 18px;
  }
  .led {
    height: 4px;
    border-radius: 2px;
    background: var(--meter-off);
    transition: background 60ms linear;
  }
  .led.on.ok,
  .led.hold.ok {
    background: var(--ok);
  }
  .led.on.warn,
  .led.hold.warn {
    background: var(--warn);
  }
  .led.on.bad,
  .led.hold.bad {
    background: var(--bad);
  }
</style>
