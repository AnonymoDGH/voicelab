<script lang="ts">
  import { initials } from "./portrait";

  // A voice's picture shown like a small monitor: the image is turned into a duotone of the
  // screen color and an "ink" (muted, or the signal color when lit) under scanlines, so photos
  // and the built-in illustrations look alike in every theme. No picture -> initials.
  // The parent sizes it (any aspect: the image is center-cropped) and may set --portrait-ink.
  let { src, name, lit = false }: { src: string | null; name: string; lit?: boolean } = $props();
</script>

<div class="portrait" class:lit aria-hidden="true">
  {#if src}
    <img {src} alt="" draggable="false" />
  {:else}
    <span class="initials mono">{initials(name)}</span>
  {/if}
</div>

<style>
  .portrait {
    --ink: var(--portrait-ink, color-mix(in srgb, var(--lcd-text-2) 80%, var(--lcd-text)));
    position: relative;
    isolation: isolate;
    overflow: hidden;
    width: 100%;
    height: 100%;
    border-radius: inherit;
    /* Studio backdrop: a soft light behind the head, so black hair stands out from the screen. */
    background: radial-gradient(circle at 50% 38%, color-mix(in srgb, var(--lcd-text) 36%, var(--lcd)), var(--lcd) 72%);
    container-type: size;
  }
  .lit {
    --ink: var(--signal);
  }
  img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: grayscale(1) contrast(1.1);
  }
  .initials {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    font-size: 34cqmin;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #fff;
  }
  /* Tint: light parts take the ink, dark parts keep the screen color. */
  .portrait::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 1;
    background: var(--ink);
    mix-blend-mode: multiply;
    transition: background 0.2s;
  }
  /* Glass: scanlines and a soft vignette. */
  .portrait::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 2;
    background:
      radial-gradient(130% 100% at 50% 35%, transparent 55%, rgb(0 0 0 / 0.4)),
      repeating-linear-gradient(to bottom, transparent 0 2px, rgb(0 0 0 / 0.12) 2px 3px);
    pointer-events: none;
  }
</style>
