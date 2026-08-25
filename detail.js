/**
 * detail.js — 详情页逻辑
 * 解析 URL 参数，加载对应 JSON 数据，渲染详情内容
 * 支持 text / image / list / code / formula / link 六种段落类型
 */

(function () {
  'use strict';

  const detailHeader = document.getElementById('detailHeader');
  const detailContent = document.getElementById('detailContent');
  const backBtn = document.getElementById('backBtn');

  // ===== URL Params =====
  const hashStr = window.location.hash.replace(/^#\/?/, '');
  const hashParams = new URLSearchParams(hashStr);
  const searchParams = new URLSearchParams(window.location.search);
  const type = hashParams.get('type') || searchParams.get('type');   // projects | papers | notes
  const id = hashParams.get('id') || searchParams.get('id');
  
  const fileMap = {
    projects: 'data/projects.json',
    papers: 'data/papers.json',
    notes: 'data/notes.json'
  };

  const typeLabels = {
    projects: '项目经历',
    papers: '学术论文',
    notes: '学习笔记'
  };

  // Removed old processText and renderers.

  // ===== Render Header =====
  function renderHeader(item) {
    let html = `<h1 class="detail-title">${item.title}</h1>`;
    html += '<div class="detail-meta">';

    // Type-specific meta
    if (type === 'projects' && item.date) {
      html += `<span class="detail-meta-item">
        <svg viewBox="0 0 24 24"><path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z" fill="currentColor"/></svg>
        ${item.date}
      </span>`;
    }

    if (type === 'papers') {
      if (item.venue) {
        html += `<span class="detail-meta-item">
          <svg viewBox="0 0 24 24"><path d="M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z" fill="currentColor"/></svg>
          ${item.venue}
        </span>`;
      }
      if (item.year) {
        html += `<span class="detail-meta-item">${item.year}</span>`;
      }
    }

    if (type === 'notes') {
      if (item.date) {
        html += `<span class="detail-meta-item">
          <svg viewBox="0 0 24 24"><path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z" fill="currentColor"/></svg>
          ${item.date}
        </span>`;
      }
      if (item.category) {
        html += `<span class="detail-meta-item">
          <svg viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" fill="currentColor"/></svg>
          ${item.category}
        </span>`;
      }
    }

    html += '</div>';

    // Authors (papers)
    if (type === 'papers' && item.authors && item.authors.length) {
      html += '<p class="detail-authors">作者：';
      html += item.authors.map((a, i) =>
        i === 0 ? `<strong>${a}</strong>` : a
      ).join('、');
      html += '</p>';
    }

    // Tags
    if (item.tags && item.tags.length) {
      html += '<div class="detail-tags">';
      item.tags.forEach(t => {
        html += `<span class="tag">${t}</span>`;
      });
      html += '</div>';
    }

    detailHeader.innerHTML = html;
    document.title = `${item.title} — ${typeLabels[type] || '详情'}`;
  }

  // ===== Render Markdown =====
  async function renderMarkdown(contentPath) {
    if (!contentPath) {
      detailContent.innerHTML = '<div class="empty-state"><p>未提供内容路径</p></div>';
      return;
    }

    try {
      const res = await fetch(`${contentPath}?t=${Date.now()}`);
      if (!res.ok) throw new Error('Failed to load markdown');
      const md = await res.text();
      
      // Fix relative image paths in markdown
      const basePath = contentPath.substring(0, contentPath.lastIndexOf('/') + 1);
      const fixedMd = md.replace(/!\[([^\]]*)\]\((?!http|https|\/)([^)]+)\)/g, `![$1](${basePath}$2)`);
      
      detailContent.innerHTML = marked.parse(fixedMd);

      // Post-render: highlight code blocks
      requestAnimationFrame(() => {
        if (typeof hljs !== 'undefined') {
          detailContent.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
          });
        }

        // Post-render: render KaTeX formulas
        if (typeof renderMathInElement !== 'undefined') {
          renderMathInElement(detailContent, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false}
            ],
            throwOnError: false
          });
        }
      });
    } catch (err) {
      console.error(err);
      detailContent.innerHTML = '<div class="empty-state"><p>加载 Markdown 内容失败</p></div>';
    }
  }

  // ===== Main =====
  async function init() {
    if (!type || !id || !fileMap[type]) {
      detailContent.innerHTML = '<div class="empty-state"><p>无效的页面参数</p></div>';
      return;
    }

    // Set back link to preserve tab state
    backBtn.href = `index.html#${type}`;

    try {
      const res = await fetch(`${fileMap[type]}?t=${Date.now()}`);
      const data = await res.json();
      const item = data.find(d => d.id === id);

      if (!item) {
        detailContent.innerHTML = '<div class="empty-state"><p>未找到对应内容</p></div>';
        return;
      }

      renderHeader(item);
      if (item.content_path) {
        await renderMarkdown(item.content_path);
      } else {
        detailContent.innerHTML = '<div class="empty-state"><p>未提供 Markdown 文件路径</p></div>';
      }
    } catch (err) {
      console.error('Failed to load detail:', err);
      detailContent.innerHTML = '<div class="empty-state"><p>加载失败，请稍后重试</p></div>';
    }
  }

  // Wait for KaTeX and Highlight.js to load
  function waitForLibs(callback, maxWait = 3000) {
    const start = Date.now();
    const check = () => {
      if ((typeof marked !== 'undefined' && typeof renderMathInElement !== 'undefined' && typeof hljs !== 'undefined') || Date.now() - start > maxWait) {
        callback();
      } else {
        requestAnimationFrame(check);
      }
    };
    check();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => waitForLibs(init));
  } else {
    waitForLibs(init);
  }
})();
