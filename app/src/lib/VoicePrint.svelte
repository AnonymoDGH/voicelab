<script lang="ts">
  // The voice's fingerprint: bars computed from its speaker embedding, mirrored around the
  // center line. Same voice -> same picture; similar voices look alike.
  let {
    print,
    height = 32,
    lit = false,
    animate = false,
  }: { print: number[]; height?: number; lit?: boolean; animate?: boolean } = $props();

  const bars = $derived(print.length ? print : Array(32).fill(0.3));
</script>

<div class="print" class:lit class:animate style="height:{height}px" aria-hidden="true">
  {#each bars as v, i}
    <span style="height:{Math.max(8, v * 100)}%; animation-delay:{(i % 8) * 70}ms"></span>
  {/each}
</div>

<style>
  .print {
    display: flex;
    align-items: center;
    gap: 2px;
    width: 100%;
  }
  span {
    flex: 1;
    min-width: 2px;
    border-radius: 2px;
    background: var(--text-3);
    opacity: 0.55;
    transition: background 0.2s, opacity 0.2s;
  }
  .lit span {
    background: var(--signal);
    opacity: 1;
  }
  .animate span {
    animation: breathe 1.1s ease-in-out infinite alternate;
    transform-origin: center;
  }
  @keyframes breathe {
    from {
      transform: scaleY(0.7);
    }
    to {
      transform: scaleY(1);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .animate span {
      animation: none;
    }
  }
</style>
