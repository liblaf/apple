const data = await (await fetch("viewer-manifest.json")).json(),
  canvas = document.querySelector("#view"),
  select = document.querySelector("#case"),
  state = document.querySelector("#state"),
  step = document.querySelector("#step"),
  stepLabel = document.querySelector("#stepLabel"),
  cutaway = document.querySelector("#cutaway"),
  material = document.querySelector("#material"),
  targetOverlay = document.querySelector("#targetOverlay"),
  fiber = document.querySelector("#fiber"),
  fiberRegion = document.querySelector("#fiberRegion"),
  hint = document.querySelector("#hint");
document.querySelector("#title").textContent = data.title;
for (const c of data.cases) select.add(new Option(c.label, c.id));
if (data.fibers)
  for (const r of data.fibers.regions.filter((x) => x !== "all"))
    fiberRegion.add(new Option("MuscleId " + r, r));
else fiber.parentElement.hidden = true;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0xf7f5ef);
const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 2));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(4, -5, 6);
scene.add(light);
const camera = new THREE.PerspectiveCamera(36, 1, 0.001, 1e9),
  cache = new Map();
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
let mesh,
  overlay,
  fiberLines,
  loadNumber = 0;
const palette = [0xd9a486, 0xa54b59, 0xd8b17d, 0x4f8f8b, 0x9b8dc2, 0x77a866, 0xe0a458, 0x7095c8];
function dispose(x) {
  x.geometry.dispose();
  if (Array.isArray(x.material)) x.material.forEach((m) => m.dispose());
  else x.material.dispose();
}
async function json(path) {
  if (!cache.has(path))
    cache.set(
      path,
      fetch(path).then((r) => {
        if (!r.ok) throw Error(path + " " + r.status);
        return r.json();
      }),
    );
  return cache.get(path);
}
async function geometry(path) {
  const s = await json(path),
    t = await json("geometry/" + s.topology);
  return { ...t, ...s };
}
function makeMesh(m, overlay = false) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(m.positions, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(m.normals, 3));
  g.setIndex(m.indices);
  let mats;
  if (!overlay && material.checked && m.material_classes) {
    const groups = [];
    let last = -1,
      start = 0;
    for (let i = 0; i < m.material_classes.length; i++) {
      const k = m.material_classes[i];
      if (i && k !== last) {
        groups.push([start, i - start, last]);
        start = i;
      }
      last = k;
    }
    groups.push([start, m.material_classes.length - start, last]);
    for (const [a, n, k] of groups) g.addGroup(3 * a, 3 * n, k);
    mats = palette.map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.8, metalness: 0 }),
    );
  } else
    mats = new THREE.MeshStandardMaterial({
      color: overlay ? 0xd65368 : 0xd9a486,
      roughness: 0.85,
      transparent: overlay,
      opacity: overlay ? 0.55 : 1,
      depthWrite: !overlay,
    });
  return new THREE.Mesh(g, mats);
}
function preset(name) {
  const p = data.cameras[name];
  camera.position.fromArray(p.position);
  controls.target.fromArray(p.focal_point);
  camera.up.fromArray(p.view_up);
  camera.near = Math.max(+p.parallel_scale / 1000, 1e-6);
  camera.far = +p.parallel_scale * 100;
  camera.updateProjectionMatrix();
  controls.update();
}
function selectedCase() {
  return data.cases.find((x) => x.id === select.value);
}
function status(message) {
  hint.textContent = message;
}
function syncHistory() {
  const c = selectedCase(),
    h = c.history || [],
    historyOption = state.querySelector('option[value="history"]'),
    targetOption = state.querySelector('option[value="target_skin"]');
  historyOption.disabled = !h.length;
  targetOption.disabled = !c.states.target_skin;
  if (
    (state.value === "history" && !h.length) ||
    (state.value === "target_skin" && !c.states.target_skin)
  )
    state.value = "endpoint";
  step.max = Math.max(h.length - 1, 0);
  step.disabled = state.value !== "history" || !h.length;
  stepLabel.textContent = h.length ? "optimization step " + h[+step.value].step : "—";
}
async function load() {
  const ticket = ++loadNumber,
    c = selectedCase(),
    history = c.history || [];
  syncHistory();
  let pointer;
  if (state.value === "history") {
    const frame = history[+step.value];
    if (!frame) {
      state.value = "endpoint";
      return load();
    }
    pointer = cutaway.checked && frame.cutaway_file ? frame.cutaway_file : frame.file;
  } else {
    pointer = c.states[state.value];
    if (cutaway.checked && state.value !== "target_skin")
      pointer = c.states[state.value + "_cutaway"] || pointer;
  }
  if (!pointer) {
    status("Selected saved geometry is unavailable");
    return;
  }
  status("Loading saved geometry…");
  try {
    const mainPromise = geometry(pointer),
      overlayPointer =
        targetOverlay.checked && state.value !== "target_skin" ? c.states.target_skin : null,
      overlayPromise = overlayPointer ? geometry(overlayPointer) : null;
    const main = await mainPromise,
      overlayData = overlayPromise ? await overlayPromise : null;
    if (ticket !== loadNumber) return;
    const nextMesh = makeMesh(main),
      nextOverlay = overlayData ? makeMesh(overlayData, true) : null;
    if (ticket !== loadNumber) {
      dispose(nextMesh);
      if (nextOverlay) dispose(nextOverlay);
      return;
    }
    if (mesh) {
      scene.remove(mesh);
      dispose(mesh);
    }
    if (overlay) {
      scene.remove(overlay);
      dispose(overlay);
    }
    mesh = nextMesh;
    overlay = nextOverlay;
    scene.add(mesh);
    if (overlay) scene.add(overlay);
    status("Orbit: drag · zoom: wheel · saved exterior triangles");
  } catch (error) {
    if (ticket === loadNumber) status("Geometry load failed: " + error.message);
  }
}
let fiberLoadNumber = 0;
async function updateFibers() {
  const ticket = ++fiberLoadNumber;
  if (fiberLines) {
    scene.remove(fiberLines);
    dispose(fiberLines);
    fiberLines = null;
  }
  if (!fiber.checked || !data.fibers) return;
  status("Loading estimated fibers…");
  try {
    const d = await json(data.fibers.file),
      p = d.regions[fiberRegion.value];
    if (ticket !== fiberLoadNumber || !fiber.checked || !p) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    const next = new THREE.LineSegments(
      g,
      new THREE.LineBasicMaterial({ color: 0x167a85, transparent: true, opacity: 0.72 }),
    );
    if (ticket !== fiberLoadNumber) {
      dispose(next);
      return;
    }
    fiberLines = next;
    scene.add(fiberLines);
    status(d.label + " · " + d.orientation);
  } catch (error) {
    if (ticket === fiberLoadNumber) status("Fiber load failed: " + error.message);
  }
}
function resize() {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);
select.onchange = () => {
  if (!(selectedCase().history || []).length) state.value = "endpoint";
  syncHistory();
  load();
};
state.onchange = () => {
  syncHistory();
  load();
};
step.oninput = () => {
  syncHistory();
  if (state.value === "history") load();
};
cutaway.onchange = () => load();
material.onchange = () => load();
targetOverlay.onchange = () => load();
fiber.onchange = () => updateFibers();
fiberRegion.onchange = () => updateFibers();
document
  .querySelectorAll("[data-camera]")
  .forEach((b) => (b.onclick = () => preset(b.dataset.camera)));
preset("front");
syncHistory();
load();
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
