
(function () {
  const card = document.querySelector("[data-prob]");
  if (!card) return;
  const prob = parseFloat(card.dataset.prob);          // 0..1
  const slider = document.getElementById("thr");        // percent 5..95
  const thrval = document.getElementById("thrval");
  const flagline = document.getElementById("flagline");

  function render() {
    const t = parseInt(slider.value, 10);               // percent
    thrval.textContent = t;
    const flagged = prob * 100 >= t;
    flagline.textContent = flagged
      ? `At a ${t}% threshold: flagged — suggested for review.`
      : `At a ${t}% threshold: not flagged.`;
    flagline.className = "flagline " + (flagged ? "flag-yes" : "flag-no");
  }

  slider.addEventListener("input", render);
  render();
})();
