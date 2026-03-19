async function loadManifest() {
  const response = await fetch('assets/generated/portfolio-manifest.json');
  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.status}`);
  }
  return response.json();
}

function text(el, value) {
  const node = document.getElementById(el);
  if (node) node.textContent = value;
}

function renderQuickStats(metrics) {
  const list = document.getElementById('quick-stats');
  list.innerHTML = [
    `<li><strong>${metrics.best_miou.toFixed(2)}</strong><br />Best mIoU</li>`,
    `<li><strong>${metrics.best_experiment}</strong><br />Best completed run</li>`,
    `<li><strong>${metrics.hero_duration_seconds}s</strong><br />Hero clip length</li>`,
    `<li><strong>${metrics.hero_size_mb} MB</strong><br />Hero clip size</li>`,
  ].join('');
}

function assetCard(asset) {
  const media = asset.kind === 'video'
    ? `<video controls preload="metadata" poster="${asset.src.replace('.mp4', '-poster.jpg')}"><source src="${asset.src}" type="video/mp4" /></video>`
    : `<img src="${asset.src}" alt="${asset.alt}" loading="lazy" />`;
  return `
    <article class="asset-card">
      ${media}
      <h3>${asset.title}</h3>
      <p>${asset.caption}</p>
      <span class="asset-meta">Source: ${asset.source}</span>
    </article>
  `;
}

function renderHero(assets) {
  const heroVideo = assets.find((asset) => asset.id === 'hero-video');
  const heroPoster = assets.find((asset) => asset.id === 'hero-poster');
  document.getElementById('hero-media').innerHTML = `
    <figure>
      <video autoplay muted loop playsinline controls poster="${heroPoster.src}">
        <source src="${heroVideo.src}" type="video/mp4" />
      </video>
      <figcaption>${heroVideo.caption}</figcaption>
    </figure>
  `;
}

function renderSectionAssets(containerId, assets, ids) {
  const container = document.getElementById(containerId);
  const selected = ids
    .map((id) => assets.find((asset) => asset.id === id))
    .filter(Boolean)
    .map(assetCard)
    .join('');
  container.innerHTML = selected;
}

function renderList(containerId, items, formatter) {
  const node = document.getElementById(containerId);
  node.innerHTML = items.map(formatter).join('');
}

loadManifest()
  .then((manifest) => {
    const { project, sections, assets, highlights, engineering_lessons: lessons, repro_steps: reproSteps } = manifest;
    text('project-title', project.title);
    text('project-subtitle', project.subtitle);
    text('project-abstract', project.abstract);
    document.getElementById('repo-link').href = project.repo_url;
    document.getElementById('report-link').href = project.report_url;
    document.getElementById('readme-link').href = `${project.repo_url}blob/main/README.md`;
    document.getElementById('report-card-link').href = project.report_url;

    renderQuickStats(project.metrics);
    renderHero(assets);
    renderList('highlights', highlights, (item) => `<li>${item}</li>`);
    renderList('lessons-grid', lessons, (item) => `<article class="lesson-card"><h3>Constraint</h3><p>${item}</p></article>`);
    renderList('repro-steps', reproSteps, (item) => `<li>${item}</li>`);

    const overview = sections.find((section) => section.id === 'overview');
    renderSectionAssets('overview-media', assets, overview.asset_ids);
    const method = sections.find((section) => section.id === 'method');
    renderSectionAssets('method-assets', assets, method.asset_ids);
    const results = sections.find((section) => section.id === 'results');
    renderSectionAssets('results-assets', assets, results.asset_ids);
  })
  .catch((error) => {
    console.error(error);
    document.getElementById('project-title').textContent = 'Failed to load portfolio manifest';
  });
