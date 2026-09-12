// CoreBank Teller client script (D026).
// Display-only except confirmPayment(). The session countdown never
// redirects, so the deterministic session-timeout trigger (D018) is
// unaffected.

var SESSION_SECONDS = 15 * 60;

function startSessionTimer() {
  var el = document.querySelector('.session-timer');
  if (!el) {
    return;
  }
  var remaining = SESSION_SECONDS;
  function tick() {
    var m = Math.floor(remaining / 60);
    var s = remaining % 60;
    el.innerHTML = 'Session expires in ' + m + ':' + (s < 10 ? '0' : '') + s;
    if (remaining > 0) {
      remaining = remaining - 1;
    }
  }
  tick();
  setInterval(tick, 1000);
}

// Native confirm() before the irreversible payment submit.
function confirmPayment() {
  return window.confirm('You are about to submit this payment. This action cannot be undone. Continue?');
}

window.onload = startSessionTimer;
