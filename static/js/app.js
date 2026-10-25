// Renders every <canvas data-chart="ID"> from the JSON in <script id="ID"> (Django json_script).
(function () {
  var palette = ["#2f6fed", "#20c997", "#fd7e14", "#6f42c1", "#dc3545", "#0dcaf0", "#ffc107"];
  document.querySelectorAll("canvas[data-chart]").forEach(function (cv) {
    var node = document.getElementById(cv.dataset.chart);
    if (!node || typeof Chart === "undefined") { return; }
    var cfg = JSON.parse(node.textContent);
    cfg.datasets.forEach(function (d, i) {
      d.backgroundColor = palette[i % palette.length];
      d.borderColor = palette[i % palette.length];
      if (cfg.type === "line") { d.fill = false; d.tension = 0.25; }
    });
    new Chart(cv, {
      type: cfg.type || "bar",
      data: { labels: cfg.labels, datasets: cfg.datasets },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: cfg.datasets.length > 1 } },
                 scales: { y: { beginAtZero: true } } }
    });
  });

  // Click-to-sort tables: <table data-sortable> with <th class="sortable">.
  document.querySelectorAll("table[data-sortable]").forEach(function (table) {
    table.querySelectorAll("th.sortable").forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var asc = th.dataset.asc !== "true";
        th.dataset.asc = asc;
        var rows = Array.from(table.tBodies[0].rows);
        rows.sort(function (a, b) {
          var x = a.cells[idx].dataset.sort || a.cells[idx].textContent.trim();
          var y = b.cells[idx].dataset.sort || b.cells[idx].textContent.trim();
          var nx = parseFloat(x), ny = parseFloat(y);
          var cmp = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
          return asc ? cmp : -cmp;
        });
        rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
      });
    });
  });

  // Training page: filter datasets' columns / algorithms by problem type.
  var meta = document.getElementById("dataset-meta");
  var dsSel = document.getElementById("id_dataset");
  if (meta && dsSel) {
    var info = JSON.parse(meta.textContent);
    var target = document.getElementById("id_target_column");
    var ptype = document.getElementById("id_problem_type");
    var refreshAlgos = function () {
      document.querySelectorAll("[data-algo-type]").forEach(function (el) {
        el.style.display = el.dataset.algoType === ptype.value ? "" : "none";
        if (el.dataset.algoType !== ptype.value) { el.querySelector("input").checked = false; }
      });
    };
    var refreshTarget = function () {
      var d = info[dsSel.value];
      var keep = target.value;
      target.innerHTML = "";
      if (!d) { return; }
      d.columns.forEach(function (c) {
        var o = document.createElement("option"); o.value = c; o.textContent = c; target.appendChild(o);
      });
      target.value = keep && d.columns.indexOf(keep) >= 0 ? keep : (d.target || d.columns[d.columns.length - 1]);
      if (d.problem_type) { ptype.value = d.problem_type; }
      refreshAlgos();
    };
    dsSel.addEventListener("change", refreshTarget);
    ptype.addEventListener("change", refreshAlgos);
    refreshTarget();
  }
})();
