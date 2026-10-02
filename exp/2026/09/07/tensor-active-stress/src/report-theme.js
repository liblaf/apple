(() => {
  "use strict";
  const key = "face-activation-followup-theme";
  const root = document.documentElement;

  function storage(action) {
    try {
      return action(window.localStorage);
    } catch (error) {
      // Browser privacy settings may disable optional preference persistence.
      if (!["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
    }
  }

  const saved = storage((store) => store.getItem(key));
  const preference = saved === "light" || saved === "dark" ? saved : "system";

  function apply(value) {
    if (value === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", value);
  }

  // This script runs in the head so a saved preference applies before paint.
  apply(preference);
  document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("report-theme");
    select.value = preference;
    select.disabled = false;
    select.addEventListener("change", () => {
      apply(select.value);
      storage((store) => {
        if (select.value === "system") store.removeItem(key);
        else store.setItem(key, select.value);
      });
    });
  });
})();
