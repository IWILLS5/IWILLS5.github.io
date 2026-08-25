/**
 * app.js — 主页逻辑
 * 负责加载个人信息、标签页切换、卡片列表渲染
 */

(function () {
  'use strict';

  // ===== DOM References =====
  const profileName = document.getElementById('profileName');
  const profileTitle = document.getElementById('profileTitle');
  const profileLocation = document.getElementById('profileLocation');
  const profileBio = document.getElementById('profileBio');
  const avatarPlaceholder = document.getElementById('avatarPlaceholder');
  const socialLinks = document.getElementById('socialLinks');
  const tabNav = document.getElementById('tabNav');
  const cardGrid = document.getElementById('cardGrid');
  const mobileToggle = document.getElementById('mobileToggle');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebarOverlay');

  // ===== Data Cache =====
  const dataCache = {};
  let currentTab = 'projects';

  // ===== SVG Icons =====
  const icons = {
    github: '<svg viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>',
    email: '<svg viewBox="0 0 24 24"><path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>',
    scholar: '<svg viewBox="0 0 24 24"><path d="M12 3L1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9zM17 15.99l-5 2.73-5-2.73v-3.72L12 15l5-2.73v3.72z"/></svg>',
    external: '<svg viewBox="0 0 24 24"><path d="M19 19H5V5h7V3H5a2 2 0 00-2 2v14a2 2 0 002 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>'
  };

  // ===== Load Profile =====
  async function loadProfile() {
    try {
      const res = await fetch(`data/profile.json?t=${Date.now()}`);
      const profile = await res.json();

      profileName.textContent = profile.name;
      profileTitle.textContent = profile.title;
      profileBio.textContent = profile.bio;

      // Avatar — use first character as placeholder
      if (profile.avatar) {
        avatarPlaceholder.innerHTML = `<img src="${profile.avatar}" alt="${profile.name}" />`;
        avatarPlaceholder.classList.remove('avatar-placeholder');
      } else {
        avatarPlaceholder.textContent = profile.name.charAt(0);
      }

      // Location
      if (profile.location) {
        profileLocation.querySelector('span').textContent = profile.location;
      } else {
        profileLocation.style.display = 'none';
      }

      // Social links
      renderSocialLinks(profile.links);

      // Update page title
      document.title = `${profile.name} — 个人主页`;
    } catch (err) {
      console.error('Failed to load profile:', err);
      profileName.textContent = 'Your Name';
    }
  }

  function renderSocialLinks(links) {
    if (!links) return;
    let html = '';

    if (links.github) {
      html += `<a class="social-link" href="${links.github}" target="_blank" rel="noopener" title="GitHub">${icons.github}</a>`;
    }
    if (links.email) {
      html += `<a class="social-link" href="mailto:${links.email}" title="Email">${icons.email}</a>`;
    }
    if (links.scholar) {
      html += `<a class="social-link" href="${links.scholar}" target="_blank" rel="noopener" title="Google Scholar">${icons.scholar}</a>`;
    }

    socialLinks.innerHTML = html;
  }

  // ===== Load Tab Data =====
  async function loadTabData(tab) {
    if (dataCache[tab]) return dataCache[tab];

    const fileMap = {
      projects: 'data/projects.json',
      papers: 'data/papers.json',
      notes: 'data/notes.json'
    };

    try {
      const res = await fetch(`${fileMap[tab]}?t=${Date.now()}`);
      const data = await res.json();
      dataCache[tab] = data;
      return data;
    } catch (err) {
      console.error(`Failed to load ${tab}:`, err);
      return [];
    }
  }

  // ===== Render Cards =====
  function renderCards(data, type) {
    if (!data || data.length === 0) {
      cardGrid.innerHTML = '<div class="empty-state"><p>暂无内容</p></div>';
      return;
    }

    cardGrid.innerHTML = data.map((item, index) => {
      const delay = `style="animation-delay: ${index * 0.06}s"`;


      if (type === 'projects') {

        return `
          <article class="card no-hover" ${delay}>
            <h3 class="card-title" style="font-size: 1.25rem; color: var(--accent-start);">${item.title}</h3>
            <div class="card-meta" style="margin-bottom: 12px; color: var(--text-primary); font-weight: 500;">
              <svg viewBox="0 0 24 24" style="width:16px;height:16px;margin-right:4px;fill:currentColor;vertical-align:text-bottom"><path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z"/></svg>
              ${item.date}
            </div>
            <div style="margin-bottom: 6px; font-weight: 600; color: var(--text-heading); font-size: 0.9rem;">项目简介：</div>
            <p class="card-summary" style="margin-bottom: 14px; -webkit-line-clamp: unset; display: block;">${item.summary}</p>
            <div style="margin-bottom: 6px; font-weight: 600; color: var(--text-heading); font-size: 0.9rem;">项目内容：</div>
            <p class="card-summary" style="white-space: pre-line; -webkit-line-clamp: unset; display: block;">${item.content}</p>
          </article>
        `;
      } else if (type === 'papers') {
        return `
          <article class="card no-hover" ${delay} style="padding: 24px;">
            <p style="font-size: 0.95rem; line-height: 1.8; color: var(--text-primary); margin: 0;">
              ${item.citation}
            </p>
          </article>
        `;
      } else if (type === 'notes') {
        let meta = `<span>${item.date}</span>`;
        if (item.category) {
          meta += `<span class="separator"></span><span>${item.category}</span>`;
        }
        const tags = (item.tags || [])
          .map(t => `<span class="tag">${t}</span>`)
          .join('');

        return `
          <article class="card" data-type="${type}" data-id="${item.id}" ${delay} onclick="navigateToDetail('${type}', '${item.id}')">
            <h3 class="card-title">${item.title}</h3>
            <div class="card-meta">${meta}</div>
            ${tags ? `<div class="card-tags">${tags}</div>` : ''}
            <p class="card-summary">${item.summary}</p>
          </article>
        `;
      }
    }).join('');
  }

  // ===== Navigation =====
  window.navigateToDetail = function (type, id) {
    window.location.href = `detail.html#type=${type}&id=${id}`;
  };

  // ===== Tab Switching =====
  async function switchTab(tab) {
    if (tab === currentTab) return;
    currentTab = tab;
    // Update active state
    tabNav.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Show loading
    cardGrid.innerHTML = '<div class="loading-state"><div class="loading-spinner"></div><p>加载中...</p></div>';

    // Load data
    const data = await loadTabData(tab);
    renderCards(data, tab);
  }

  // ===== Mobile Sidebar =====
  function toggleSidebar() {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('open');
  }

  // ===== Event Listeners =====
  tabNav.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (btn) switchTab(btn.dataset.tab);
  });

  mobileToggle.addEventListener('click', toggleSidebar);
  sidebarOverlay.addEventListener('click', toggleSidebar);

  // ===== Initialize =====
  async function init() {
    await loadProfile();

    // Restore tab from URL hash (e.g. index.html#papers)
    const hash = window.location.hash.replace('#', '');
    const validTabs = ['projects', 'papers', 'notes'];
    const initialTab = validTabs.includes(hash) ? hash : 'projects';

    currentTab = initialTab;
    tabNav.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === initialTab);
    });

    const data = await loadTabData(initialTab);
    renderCards(data, initialTab);
  }

  init();
})();
