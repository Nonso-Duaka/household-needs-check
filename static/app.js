// Results page: grow-in animations + the "flag if over" slider.
(function () {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!reduce) {
    // Bars grow from 0.
    document.querySelectorAll(".dbar, .bbar").forEach((el) => {
      const target = el.style.width;
      el.style.width = "0";
      requestAnimationFrame(() => requestAnimationFrame(() => { el.style.width = target; }));
    });
    // Gauges sweep in from empty.
    const CIRC = 326.726;
    document.querySelectorAll(".gauge-value").forEach((el) => {
      const target = el.getAttribute("stroke-dashoffset");
      el.setAttribute("stroke-dashoffset", CIRC);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        el.style.strokeDashoffset = target;
      }));
    });
  }

  const card = document.querySelector("[data-prob]");
  if (!card) return;
  const prob = parseFloat(card.dataset.prob) * 100;
  const slider = document.getElementById("thr");
  const thrval = document.getElementById("thrval");
  const flagline = document.getElementById("flagline");
  if (!slider) return;

  function render() {
    const t = parseInt(slider.value, 10);
    thrval.textContent = t;
    const flagged = prob >= t;
    flagline.textContent = flagged
      ? "Flagged for a closer look"
      : "Not flagged";
    flagline.className = "flagline " + (flagged ? "flag-yes" : "flag-no");
  }
  slider.addEventListener("input", render);
  render();
})();
