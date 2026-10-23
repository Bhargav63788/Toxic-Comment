/**
 * YouTube Toxic Comment Analyzer - Client-Side Controller
 * Connects the clean single-page UI with Flask backend /api/analyze_youtube
 */

let analysisData = null;
let currentFilter = "all";
let displayedCommentCount = 0;
const COMMENT_BATCH_SIZE = 100;

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("video-url");
  const form = document.getElementById("video-form");

  if (input) {
    input.focus();

    // Auto-preview video on paste or input
    const checkAndPreview = () => {
      const vid = getVideoId(input.value);
      if (vid) {
        renderVideoPlayer(vid);
      }
    };

    input.addEventListener("input", checkAndPreview);
    input.addEventListener("paste", () => setTimeout(checkAndPreview, 60));
  }

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      startAnalysis();
    });
  }
});

function getVideoId(value) {
  if (!value || typeof value !== "string") return null;
  let str = value.trim();

  // If protocol omitted but youtube domain present, prepend https://
  if (!/^https?:\/\//i.test(str) && (str.includes("youtube.com") || str.includes("youtu.be"))) {
    str = "https://" + str;
  }

  try {
    const url = new URL(str);
    const host = url.hostname.toLowerCase().replace(/^www\./, "");
    let videoId = null;

    if (host === "youtu.be") {
      videoId = url.pathname.split("/").filter(Boolean)[0];
    } else if (host === "youtube.com" || host === "m.youtube.com" || host === "youtube-nocookie.com") {
      const parts = url.pathname.split("/").filter(Boolean);
      if (url.pathname === "/watch") {
        videoId = url.searchParams.get("v");
      } else if (["shorts", "embed", "live", "v"].includes(parts[0])) {
        videoId = parts[1];
      }
    }

    if (videoId && /^[\w-]{11}$/.test(videoId)) {
      return videoId;
    }
  } catch {
    // If not a URL, check if raw 11-char ID
    if (/^[\w-]{11}$/.test(str)) {
      return str;
    }
  }
  return null;
}

function setMessage(text, type = "") {
  const msg = document.getElementById("form-message");
  if (!msg) return;
  msg.textContent = text;
  msg.className = "form-message";
  if (type === "error") {
    msg.classList.add("error");
  } else if (type === "success") {
    msg.classList.add("success");
  }
}

async function startAnalysis() {
  const input = document.getElementById("video-url");
  const urlValue = input ? input.value.trim() : "";
  const videoId = getVideoId(urlValue);

  if (!videoId) {
    setMessage("Enter a valid YouTube watch, youtu.be, or Shorts link.", "error");
    resetVideoPreview();
    return;
  }

  // 1. Embed Video Player in Left Column
  renderVideoPlayer(videoId);

  // 2. Prepare Right Column Loading State
  showLoadingState(true);

  setMessage("Retrieving and analyzing up to 5,000 comments with machine learning ...", "success");

  try {
    const response = await fetch("/api/analyze_youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: urlValue,
        max_comments: 5000,
        include_replies: true
      })
    });

    const data = await response.json();

    if (!response.ok || data.status === "error") {
      if (data && data.video_meta) {
        updateVideoMeta(data.video_meta, data.video_id);
      }
      throw new Error(data.message || `Server returned error (${response.status})`);
    }

    analysisData = data;
    renderAnalysisResults(data);
    if (data.total_analyzed === 0) {
      setMessage(data.disclaimer || "Video loaded successfully. No public comments found on this video.", "success");
    } else {
      setMessage(`Successfully analyzed ${data.total_analyzed.toLocaleString()} comments in ${(data.latency_ms / 1000).toFixed(2)}s!`, "success");
    }

  } catch (err) {
    console.error("Analysis failure:", err);
    setMessage(err.message || "Failed to analyze comments. Please try again.", "error");
    showEmptyState(err.message);
  } finally {
    showLoadingState(false);
  }
}

function formatNumber(num) {
  if (num === null || num === undefined || num === "") return "-";
  const n = typeof num === "string" ? parseInt(num, 10) : num;
  if (isNaN(n)) return "-";
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1).replace(/\.0$/, "") + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "K";
  return n.toLocaleString();
}

function formatDate(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    return "Published " + d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return "";
  }
}

function toggleDescription() {
  const desc = document.getElementById("video-description");
  const btn = document.getElementById("desc-toggle-btn");
  if (!desc) return;
  const isCollapsed = desc.classList.contains("collapsed");
  if (isCollapsed) {
    desc.classList.remove("collapsed");
    if (btn) btn.textContent = "Show less";
  } else {
    desc.classList.add("collapsed");
    if (btn) btn.textContent = "Show more";
  }
}

function updateVideoMeta(meta, videoId) {
  if (!meta) return;
  const infoBlock = document.getElementById("video-info");
  if (!infoBlock) return;
  infoBlock.style.display = "block";

  // Channel & Avatar
  const channel = meta.channel || "YouTube Channel";
  const channelEl = document.getElementById("video-channel");
  const avatarEl = document.getElementById("channel-avatar");
  if (channelEl) channelEl.textContent = channel;
  if (avatarEl) avatarEl.textContent = (channel.trim()[0] || "Y").toUpperCase();

  // Date
  const dateEl = document.getElementById("video-published");
  if (dateEl) dateEl.textContent = formatDate(meta.published_at);

  // Title & Watch Link
  const titleEl = document.getElementById("video-title");
  if (titleEl) titleEl.textContent = meta.title || "YouTube Video";

  const linkEl = document.getElementById("video-external-link");
  if (linkEl && videoId) linkEl.href = `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;

  // Stat Chips
  const viewsEl = document.getElementById("val-views");
  const likesEl = document.getElementById("val-likes");
  const commentsEl = document.getElementById("val-comments");

  if (viewsEl) viewsEl.textContent = formatNumber(meta.view_count);
  if (likesEl) likesEl.textContent = formatNumber(meta.like_count);
  if (commentsEl) commentsEl.textContent = formatNumber(meta.total_comment_count);

  // Description
  const descEl = document.getElementById("video-description");
  const descBtn = document.getElementById("desc-toggle-btn");
  const desc = (meta.description || "").trim();
  if (descEl) {
    descEl.textContent = desc || "No description provided for this video.";
    descEl.classList.add("collapsed");
    if (descBtn) {
      descBtn.textContent = "Show more";
      descBtn.style.display = desc && desc.length > 110 ? "inline-block" : "none";
    }
  }

  // Tags
  const tagsRow = document.getElementById("video-tags-row");
  if (tagsRow) {
    const tags = Array.isArray(meta.tags) ? meta.tags : [];
    if (tags.length > 0) {
      tagsRow.style.display = "flex";
      tagsRow.innerHTML = tags.map(t => `<span class="video-tag-pill">#${escapeHtml(t)}</span>`).join("");
    } else {
      tagsRow.style.display = "none";
      tagsRow.innerHTML = "";
    }
  }
}

function renderVideoPlayer(videoId) {
  const frame = document.getElementById("video-frame");
  if (!frame) return;

  const embedUrl = `https://www.youtube.com/embed/${encodeURIComponent(videoId)}`;
  const currentIframe = frame.querySelector("iframe");
  if (currentIframe && currentIframe.src === embedUrl) {
    return;
  }

  const player = document.createElement("iframe");
  player.title = "YouTube video preview";
  player.src = embedUrl;
  player.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
  player.allowFullscreen = true;
  frame.innerHTML = "";
  frame.appendChild(player);

  const badge = document.getElementById("video-badge");
  if (badge) badge.style.display = "inline-block";
}

function resetVideoPreview() {
  const frame = document.getElementById("video-frame");
  if (frame) {
    frame.innerHTML = `
      <div class="video-placeholder-content">
        <span class="video-placeholder-icon">&#9654;</span>
        <p id="video-placeholder-text">Paste any YouTube video link above to preview the player, explore its full description, and analyze comments.</p>
      </div>`;
  }
  const badge = document.getElementById("video-badge");
  if (badge) badge.style.display = "none";
  const infoBlock = document.getElementById("video-info");
  if (infoBlock) infoBlock.style.display = "none";
}

function showLoadingState(isLoading) {
  const btn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsDiv = document.getElementById("analysis-results");

  if (isLoading) {
    if (btn) btn.disabled = true;
    if (btnText) btnText.textContent = "Analyzing ...";
    if (emptyState) emptyState.style.display = "none";
    if (resultsDiv) resultsDiv.style.display = "none";
    if (loadingState) loadingState.style.display = "flex";
  } else {
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = "Analyze comments";
    if (loadingState) loadingState.style.display = "none";
  }
}

function showEmptyState(customMsg = null) {
  const emptyState = document.getElementById("empty-state");
  const resultsDiv = document.getElementById("analysis-results");
  if (emptyState) {
    emptyState.style.display = "grid";
    if (customMsg) {
      const heading = emptyState.querySelector("h2");
      const desc = emptyState.querySelector("p");
      if (heading) heading.textContent = "Could Not Retrieve Comments";
      if (desc) desc.textContent = customMsg;
    }
  }
  if (resultsDiv) resultsDiv.style.display = "none";
}

function renderAnalysisResults(data) {
  const emptyState = document.getElementById("empty-state");
  const resultsDiv = document.getElementById("analysis-results");
  if (emptyState) emptyState.style.display = "none";
  if (resultsDiv) resultsDiv.style.display = "block";

  // 1. Update Video Info and Description below player
  updateVideoMeta(data.video_meta, data.video_id);

  // 2. Metrics Numbers
  const total = data.total_analyzed || 0;
  const toxicCount = data.toxic_count || 0;
  const cleanCount = data.non_toxic_count || 0;
  const toxicPct = data.toxic_percentage || 0.0;
  const cleanPct = data.non_toxic_percentage || 0.0;

  document.getElementById("stat-total").textContent = total.toLocaleString();
  document.getElementById("stat-toxic").textContent = toxicCount.toLocaleString();
  document.getElementById("stat-clean").textContent = cleanCount.toLocaleString();

  document.getElementById("stat-toxic-pct").textContent = `${toxicPct.toFixed(1)}%`;
  document.getElementById("stat-clean-pct").textContent = `${cleanPct.toFixed(1)}%`;

  // 3. Distribution Split Bar
  const barClean = document.getElementById("bar-clean");
  const barToxic = document.getElementById("bar-toxic");
  const textClean = document.getElementById("ratio-clean-text");
  const textToxic = document.getElementById("ratio-toxic-text");

  if (barClean) barClean.style.width = total === 0 ? "50%" : `${cleanPct}%`;
  if (barToxic) barToxic.style.width = total === 0 ? "50%" : `${toxicPct}%`;
  if (textClean) textClean.textContent = `${cleanPct.toFixed(1)}%`;
  if (textToxic) textToxic.textContent = `${toxicPct.toFixed(1)}%`;

  // 4. Filter Button Counts
  document.getElementById("count-all").textContent = total;
  document.getElementById("count-toxic").textContent = toxicCount;
  document.getElementById("count-clean").textContent = cleanCount;

  // 5. Render Comment Cards
  currentFilter = "all";
  updateFilterButtons();
  renderCommentsFeed(true);
}

function getFilteredComments() {
  if (!analysisData || !analysisData.comments) return [];
  if (currentFilter === "all") return analysisData.comments;
  if (currentFilter === "toxic") return analysisData.comments.filter(c => c.is_toxic);
  if (currentFilter === "clean") return analysisData.comments.filter(c => !c.is_toxic);
  return analysisData.comments;
}

function renderCommentsFeed(reset = true) {
  const container = document.getElementById("comments-container");
  const emptyFilter = document.getElementById("empty-filter");
  const loadMoreBar = document.getElementById("load-more-bar");
  const loadMoreInfo = document.getElementById("load-more-info");
  const loadMoreBtn = document.getElementById("load-more-btn");
  if (!container) return;

  const filtered = getFilteredComments();

  if (reset) {
    container.innerHTML = "";
    displayedCommentCount = 0;
  }

  if (!filtered || filtered.length === 0) {
    if (emptyFilter) {
      const p = emptyFilter.querySelector("p");
      if (p) {
        if (!analysisData || !analysisData.comments || analysisData.comments.length === 0) {
          p.textContent = (analysisData && analysisData.disclaimer)
            ? analysisData.disclaimer
            : "This video currently has 0 public comments to analyze.";
        } else {
          p.textContent = "No comments found under the selected filter.";
        }
      }
      emptyFilter.style.display = "block";
    }
    if (loadMoreBar) loadMoreBar.style.display = "none";
    return;
  }

  if (emptyFilter) emptyFilter.style.display = "none";

  const nextBatch = filtered.slice(displayedCommentCount, displayedCommentCount + COMMENT_BATCH_SIZE);
  const fragment = document.createDocumentFragment();

  nextBatch.forEach(c => {
    fragment.appendChild(createCommentCardElement(c));
  });

  container.appendChild(fragment);
  displayedCommentCount += nextBatch.length;

  if (loadMoreBar) {
    if (displayedCommentCount < filtered.length) {
      loadMoreBar.style.display = "flex";
      if (loadMoreInfo) {
        loadMoreInfo.textContent = `Showing ${displayedCommentCount.toLocaleString()} of ${filtered.length.toLocaleString()} comments`;
      }
      if (loadMoreBtn) {
        const remaining = filtered.length - displayedCommentCount;
        const nextCount = Math.min(COMMENT_BATCH_SIZE, remaining);
        loadMoreBtn.textContent = `Load next ${nextCount.toLocaleString()} comments`;
      }
    } else {
      loadMoreBar.style.display = "none";
    }
  }
}

function createCommentCardElement(c) {
  const isToxic = c.is_toxic;
  const card = document.createElement("div");
  card.className = `comment-card ${isToxic ? "is-toxic" : "is-clean"}`;

  const authorInitials = (c.author || "U").substring(0, 2).toUpperCase();
  const categories = c.flagged_categories || [];

  let categoryTagsHtml = "";
  if (isToxic && categories.length > 0) {
    categoryTagsHtml = `<div class="category-tags">` +
      categories.map(cat => `<span class="tag-item">${cat.replace("_", " ")}</span>`).join("") +
      `</div>`;
  }

  const replyBadge = c.is_reply ? `<span class="badge-reply-indicator">↳ Reply</span>` : "";

  card.innerHTML = `
    <div class="comment-card-top">
      <div class="comment-author-info">
        <div class="author-circle">${authorInitials}</div>
        <span class="author-text">${escapeHtml(c.author)}</span>
        ${replyBadge}
      </div>
      <span class="badge-pill ${isToxic ? "toxic" : "clean"}">
        ${isToxic ? "Toxic" : "Non-Toxic"}
      </span>
    </div>

    <div class="comment-body">${escapeHtml(c.text)}</div>

    <div class="comment-meta-bar">
      ${categoryTagsHtml}
      ${c.like_count ? `<span>👍 ${Number(c.like_count).toLocaleString()}</span>` : "<span></span>"}
    </div>
  `;

  return card;
}

function loadMoreComments() {
  renderCommentsFeed(false);
}

function loadAllComments() {
  const container = document.getElementById("comments-container");
  const loadMoreBar = document.getElementById("load-more-bar");
  if (!container) return;

  const filtered = getFilteredComments();
  const remaining = filtered.slice(displayedCommentCount);
  const fragment = document.createDocumentFragment();

  remaining.forEach(c => {
    fragment.appendChild(createCommentCardElement(c));
  });

  container.appendChild(fragment);
  displayedCommentCount = filtered.length;

  if (loadMoreBar) loadMoreBar.style.display = "none";
}

function setFilter(type) {
  currentFilter = type;
  updateFilterButtons();
  renderCommentsFeed(true);
}

function updateFilterButtons() {
  document.querySelectorAll(".filter-tab").forEach(tab => {
    if (tab.dataset.filter === currentFilter) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
