/* Bitino website: theme toggle + model-zoo table (search, filters, sort). No dependencies. */
(function () {
  var root = document.documentElement;
  var btn = document.querySelector("button.theme");
  if (btn) {
    btn.addEventListener("click", function () {
      var dark = root.dataset.theme ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("bitino-theme", root.dataset.theme); } catch (e) {}
    });
  }
})();

var BitinoZoo = (function () {
  var rows = [], sortKey = "id", sortDir = 1;
  var MARK = { clean: "", throttled: " ⚠", "unknown-caps": " †" };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function acc(v, metric) {
    if (typeof v !== "number") return "–";
    if (metric && metric.indexOf("PSNR") >= 0) return v.toFixed(2) + " dB";
    if (metric && metric.indexOf("cosine") >= 0) return v.toFixed(5);
    return (100 * v).toFixed(1) + "%";
  }
  function ms(x) {
    if (x == null) return "–";
    return x < 10 ? x.toPrecision(3) : x.toFixed(1);
  }
  function val(m, k) {
    switch (k) {
      case "runtime": return typeof m.runtime === "number" ? m.runtime : -1;
      case "int8d": return m.int8 && typeof m.int8.delta === "number" ? m.int8.delta : -999;
      case "s22t1": return m.s22 && m.s22.t1 ? m.s22.t1.ms : 1e9;
      case "s22t4": return m.s22 && m.s22.t4 ? m.s22.t4.ms : 1e9;
      case "x86": return m.x86_t1 == null ? 1e9 : m.x86_t1;
      case "insn_opt": return m.insn_opt == null ? 1e12 : m.insn_opt;
      case "release": return m.release ? 0 : 1;
      default: return String(m[k] || "").toLowerCase();
    }
  }
  function render() {
    var q = document.getElementById("q").value.trim().toLowerCase();
    var task = document.getElementById("task").value, fw = document.getElementById("fw").value;
    var rel = document.getElementById("rel").checked, spd = document.getElementById("spd").checked;
    var list = rows.filter(function (m) {
      if (task && m.task !== task) return false;
      if (fw && m.framework !== fw) return false;
      if (rel && !m.release) return false;
      if (spd && !(m.s22 && (m.s22.t1 || m.s22.t4))) return false;
      if (q) {
        var hay = [m.id, m.name, m.task, m.family, m.framework, m.precision, m.paper].join(" ").toLowerCase();
        return q.split(/\s+/).every(function (w) { return hay.indexOf(w) >= 0; });
      }
      return true;
    });
    list.sort(function (a, b) {
      var x = val(a, sortKey), y = val(b, sortKey);
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
    });
    var html = list.map(function (m) {
      var i8 = m.int8 && typeof m.int8.delta === "number"
        ? (m.int8.delta > 0 ? "+" : "") + m.int8.delta.toFixed(2) + " pt<span class=sub>" + esc(m.int8.set) + ", n=" + esc(m.int8.n) + "</span>" : "–";
      var t1 = m.s22 && m.s22.t1, t4 = m.s22 && m.s22.t4;
      var label = function (f) { return f.indexOf("log2") >= 0 ? "int8 log2" : f.indexOf("int8") >= 0 ? "int8" : "float"; };
      var full = m.licence_weights || "?";
      var short = full.split(/ \(|;|,/)[0];
      if (short.length > 34) short = short.slice(0, 32) + "…";
      var lic = '<span class="sub" title="' + esc(full) + '">' + esc(short) + "</span>" + (m.release
        ? m.assets.map(function (a) {
            return '<a href="' + esc(a.url) + '" title="' + esc(a.asset) + " · sha256 " + esc(a.sha256) + '">' + label(a.file) + "</a>";
          }).join(" · ")
        : "");
      return "<tr>" +
        "<td><span class=id>" + esc(m.id) + "</span><span class=sub>" + esc(m.name) + "</span>" +
        "<span class=sub>recipe: <span class='badge " + (m.recipe === "verified" ? "verified" : "") + "'>" + esc(m.recipe) + "</span> · env " + esc(m.env) + "</span></td>" +
        "<td>" + esc(m.task) + "<span class=sub>" + esc(m.precision) + "</span></td>" +
        "<td>" + esc(m.framework) + "</td>" +
        "<td class=num>" + acc(m.reference, m.metric) + " → " + acc(m.runtime, m.metric) + "<span class=sub>" + esc(m.eval_set) + (m.n ? ", n=" + esc(m.n) : "") + "</span></td>" +
        "<td class=num>" + i8 + "</td>" +
        "<td class=num>" + (m.insn_opt != null ? m.insn_opt.toFixed(1) : "–") + " / " + (m.insn_int8 != null ? m.insn_int8.toFixed(1) : "–") + "</td>" +
        "<td class=num>" + (t1 ? ms(t1.ms) + (MARK[t1.validity] || "") + "<span class=sub>" + esc(t1.stage) + ", " + esc(t1.date) + "</span>" : "–") + "</td>" +
        "<td class=num>" + (t4 ? ms(t4.ms) + (MARK[t4.validity] || "") + "<span class=sub>" + esc(t4.stage) + ", " + esc(t4.date) + "</span>" : "–") + "</td>" +
        "<td class=num>" + (m.x86_t1 != null ? ms(m.x86_t1) + " ‡" : "–") + "</td>" +
        "<td>" + (m.release ? "<span class='badge ok'>release</span><br>" : "") + lic +
        "<span class=sub><code>bitino fetch " + esc(m.id) + "</code></span></td>" +
        "</tr>";
    }).join("");
    document.querySelector("#zoo tbody").innerHTML = html || "<tr><td colspan=10>No model matches.</td></tr>";
    document.getElementById("count").textContent = list.length + " of " + rows.length + " models";
  }
  function fill(id, values) {
    var sel = document.getElementById(id);
    values.forEach(function (v) {
      var o = document.createElement("option");
      o.value = v; o.textContent = v; sel.appendChild(o);
    });
  }
  function init() {
    rows = JSON.parse(document.getElementById("zoo-data").textContent);
    var uniq = function (k) { return rows.map(function (m) { return m[k]; }).filter(function (v, i, a) { return v && a.indexOf(v) === i; }).sort(); };
    fill("task", uniq("task"));
    fill("fw", uniq("framework"));
    ["q", "task", "fw", "rel", "spd"].forEach(function (id) {
      document.getElementById(id).addEventListener(id === "q" ? "input" : "change", render);
    });
    document.querySelectorAll("#zoo th").forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.dataset.k;
        sortDir = sortKey === k ? -sortDir : 1;
        sortKey = k;
        document.querySelectorAll("#zoo th").forEach(function (t) { t.classList.remove("sorted-asc", "sorted-desc"); });
        th.classList.add(sortDir > 0 ? "sorted-asc" : "sorted-desc");
        render();
      });
    });
    var h = decodeURIComponent((location.hash || "").slice(1));
    if (h) document.getElementById("q").value = h;
    render();
  }
  return { init: init };
})();
