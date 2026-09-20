/**
 * <jarvis-bots-button> — the Bots entry point for the Jarvis navigation.
 *
 * A framework-free custom element, so it drops into whatever the Jarvis app
 * already uses for navigation: a bottom tab bar, a sidebar, or a header.
 * It owns three things and nothing else:
 *
 *   1. An accessible button with an icon and a label.
 *   2. A badge showing how many bots want attention right now.
 *   3. A status poll, so the badge is true without the page being open.
 *
 * It does not navigate by itself. It emits `jarvis-bots:open`, and the app
 * decides what that means — push a route, open a sheet, swap a panel. That
 * keeps it compatible with a router this component has never heard of.
 *
 * Usage
 * -----
 *   <script type="module" src="./bots-button.js"></script>
 *   <jarvis-bots-button
 *       label="Bots"
 *       status-url="/api/bots/status"
 *       poll-ms="30000"></jarvis-bots-button>
 *
 *   document.addEventListener('jarvis-bots:open', () => router.go('/bots'));
 *
 * With no `status-url` it renders a quiet button with no badge, which is the
 * correct look before any bot is configured.
 *
 * Attributes
 * ----------
 *   label       button text (default "Bots"); always rendered, never icon-only
 *   status-url  GET endpoint returning { attention: <int>, state: "ok"|"warn"|"error" }
 *   poll-ms     poll interval, floored at 5000 and paused while the tab is hidden
 *   compact     boolean; icon over label, for a bottom tab bar
 *
 * Properties
 * ----------
 *   .status = { attention, state }   set directly if the app already has the data,
 *                                    which is cheaper than a second poll
 */

const FLOOR_MS = 5000;

const ICON = `
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <rect x="4" y="7.5" width="16" height="12" rx="3.5"
        fill="none" stroke="currentColor" stroke-width="1.7"/>
  <path d="M12 7.5V4.2" fill="none" stroke="currentColor"
        stroke-width="1.7" stroke-linecap="round"/>
  <circle cx="12" cy="3.1" r="1.4" fill="currentColor"/>
  <circle cx="9.2" cy="13" r="1.35" fill="currentColor"/>
  <circle cx="14.8" cy="13" r="1.35" fill="currentColor"/>
  <path d="M9.6 16.4h4.8" fill="none" stroke="currentColor"
        stroke-width="1.5" stroke-linecap="round"/>
</svg>`;

const STYLE = `
:host {
  /* Tokens fall back to sensible values, but inherit the app's if it sets
     them, so the button matches whatever Jarvis already looks like. */
  --bb-fg: var(--jarvis-nav-fg, #d7d3cb);
  --bb-fg-strong: var(--jarvis-nav-fg-strong, #f5f2ec);
  --bb-bg: var(--jarvis-nav-bg, transparent);
  --bb-hover: var(--jarvis-nav-hover, rgba(255,255,255,.08));
  --bb-accent: var(--jarvis-accent, #e0a72a);
  --bb-warn: var(--jarvis-warn, #e0a72a);
  --bb-error: var(--jarvis-error, #d9553d);
  --bb-badge-fg: var(--jarvis-badge-fg, #16171b);
  display: inline-block;
}
@media (prefers-color-scheme: light) {
  :host(:not([data-theme="dark"])) {
    --bb-fg: var(--jarvis-nav-fg, #3f3a33);
    --bb-fg-strong: var(--jarvis-nav-fg-strong, #17150f);
    --bb-hover: var(--jarvis-nav-hover, rgba(0,0,0,.06));
    --bb-accent: var(--jarvis-accent, #9a6508);
    --bb-warn: var(--jarvis-warn, #9a6508);
    --bb-error: var(--jarvis-error, #b23a25);
    --bb-badge-fg: var(--jarvis-badge-fg, #fffdf8);
  }
}
button {
  /* 44px minimum in both directions: this is a phone control. */
  min-height: 44px;
  min-width: 44px;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  padding: 8px 14px;
  position: relative;
  background: var(--bb-bg);
  color: var(--bb-fg);
  border: 0;
  border-radius: 10px;
  font: inherit;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: .01em;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
}
:host([compact]) button {
  flex-direction: column;
  gap: 3px;
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .02em;
}
button:hover { background: var(--bb-hover); color: var(--bb-fg-strong); }
button:focus-visible {
  outline: 2px solid var(--bb-accent);
  outline-offset: 2px;
  color: var(--bb-fg-strong);
}
button[aria-current="page"] { color: var(--bb-fg-strong); }
svg { width: 22px; height: 22px; display: block; flex: none; }
:host([compact]) svg { width: 24px; height: 24px; }

.badge {
  position: absolute;
  top: 4px;
  left: 26px;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  box-sizing: border-box;
  border-radius: 9px;
  background: var(--bb-accent);
  color: var(--bb-badge-fg);
  font-size: 11px;
  font-weight: 700;
  line-height: 18px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
:host([compact]) .badge { left: 50%; top: 2px; margin-left: 2px; }
.badge[data-state="error"] { background: var(--bb-error); color: #fff; }
.badge[data-state="warn"]  { background: var(--bb-warn); }
.badge[hidden] { display: none; }

/* The badge appearing is the one moment motion is worth it: it means
   something changed while you were not looking. */
@media (prefers-reduced-motion: no-preference) {
  .badge { animation: pop .18s ease-out; }
  @keyframes pop { from { transform: scale(.6); opacity: 0 } to { transform: none; opacity: 1 } }
}
.sr {
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap;
}
`;

class JarvisBotsButton extends HTMLElement {
  static observedAttributes = ['label', 'status-url', 'poll-ms', 'compact'];

  #root;
  #button;
  #badge;
  #live;
  #timer = null;
  #status = { attention: 0, state: 'ok' };
  #abort = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = STYLE;

    this.#button = document.createElement('button');
    this.#button.type = 'button';
    this.#button.innerHTML = ICON;

    const label = document.createElement('span');
    label.className = 'label';
    this.#button.append(label);

    this.#badge = document.createElement('span');
    this.#badge.className = 'badge';
    this.#badge.hidden = true;
    this.#button.append(this.#badge);

    // A polite live region: the badge count is announced when it changes,
    // rather than the whole button being re-read.
    this.#live = document.createElement('span');
    this.#live.className = 'sr';
    this.#live.setAttribute('aria-live', 'polite');

    this.#button.addEventListener('click', () => this.#open());
    this.#root.append(style, this.#button, this.#live);
  }

  connectedCallback() {
    this.#render();
    this.#schedule();
    document.addEventListener('visibilitychange', this.#onVisibility);
  }

  disconnectedCallback() {
    this.#stop();
    document.removeEventListener('visibilitychange', this.#onVisibility);
  }

  attributeChangedCallback() {
    this.#render();
    this.#schedule();
  }

  get status() { return { ...this.#status }; }

  set status(value) {
    const attention = Math.max(0, Number(value?.attention) || 0);
    const state = ['ok', 'warn', 'error'].includes(value?.state) ? value.state : 'ok';
    const changed = attention !== this.#status.attention || state !== this.#status.state;
    this.#status = { attention, state };
    this.#render();
    if (changed) this.#announce();
  }

  /** Mark this button as the current page, for the app's router to call. */
  setCurrent(isCurrent) {
    if (isCurrent) this.#button.setAttribute('aria-current', 'page');
    else this.#button.removeAttribute('aria-current');
  }

  #open() {
    this.dispatchEvent(new CustomEvent('jarvis-bots:open', {
      bubbles: true,
      composed: true,
      detail: { status: this.status },
    }));
  }

  #render() {
    const label = this.getAttribute('label') || 'Bots';
    this.#root.querySelector('.label').textContent = label;

    const { attention, state } = this.#status;
    if (attention > 0) {
      this.#badge.hidden = false;
      this.#badge.textContent = attention > 99 ? '99+' : String(attention);
      this.#badge.dataset.state = state;
    } else {
      this.#badge.hidden = true;
      this.#badge.removeAttribute('data-state');
    }

    // The accessible name carries the count and the state, so the badge is
    // never the only way to know something needs attention.
    const parts = [label];
    if (attention > 0) {
      parts.push(`${attention} ${attention === 1 ? 'bot needs' : 'bots need'} attention`);
    }
    if (state === 'error') parts.push('one or more bots are failing');
    else if (state === 'warn' && attention === 0) parts.push('a bot is paused');
    this.#button.setAttribute('aria-label', parts.join(', '));
  }

  #announce() {
    const { attention, state } = this.#status;
    if (attention === 0 && state === 'ok') { this.#live.textContent = ''; return; }
    this.#live.textContent = attention > 0
      ? `${attention} ${attention === 1 ? 'bot needs' : 'bots need'} attention`
      : (state === 'error' ? 'A bot is failing' : 'A bot is paused');
  }

  #onVisibility = () => {
    // Polling a hidden tab burns the owner's battery for a badge nobody can
    // see. Stop while hidden, and refresh once on return.
    if (document.hidden) this.#stop();
    else this.#schedule({ immediate: true });
  };

  #interval() {
    const raw = Number(this.getAttribute('poll-ms'));
    return Number.isFinite(raw) && raw > 0 ? Math.max(FLOOR_MS, raw) : 30000;
  }

  #stop() {
    if (this.#timer !== null) { clearTimeout(this.#timer); this.#timer = null; }
    this.#abort?.abort();
    this.#abort = null;
  }

  #schedule({ immediate = true } = {}) {
    this.#stop();
    if (!this.getAttribute('status-url') || document.hidden || !this.isConnected) return;
    const tick = async () => {
      await this.#poll();
      if (this.isConnected && !document.hidden) {
        this.#timer = setTimeout(tick, this.#interval());
      }
    };
    if (immediate) tick();
    else this.#timer = setTimeout(tick, this.#interval());
  }

  async #poll() {
    const url = this.getAttribute('status-url');
    if (!url) return;
    this.#abort?.abort();
    const controller = new AbortController();
    this.#abort = controller;
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { accept: 'application/json' },
        credentials: 'same-origin',
      });
      if (!response.ok) { this.status = { attention: 0, state: 'error' }; return; }
      const data = await response.json();
      this.status = { attention: data.attention, state: data.state };
    } catch (error) {
      // An aborted poll is a normal part of teardown, not a failure worth
      // showing. Anything else means the app cannot reach its own backend.
      if (controller.signal.aborted) return;
      this.status = { attention: 0, state: 'error' };
    }
  }
}

if (!customElements.get('jarvis-bots-button')) {
  customElements.define('jarvis-bots-button', JarvisBotsButton);
}

export { JarvisBotsButton };
