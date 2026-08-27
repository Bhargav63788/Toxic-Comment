let currentFilter = "all";
let currentVideoId = null;

function setVideoFrame(videoId) {
  const wrapper = document.querySelector(".video-wrapper");
  wrapper.innerHTML = `
    <iframe id="ytFrame"
      src="https://www.youtube.com/embed/${videoId}"
      frameborder="0" allowfullscreen
      onerror="showEmbedFallback('${videoId}')">
    </iframe>
  `;

  // YouTube doesn't always trigger onerror for blocked embeds - check via thumbnail load instead
  setTimeout(() => checkEmbedBlocked(videoId), 2500);
}

function checkEmbedBlocked(videoId) {
  const frame = document.getElementById("ytFrame");
  if (!frame) return;
  try {
    // If we can't read anything and it's been a couple seconds, YouTube often
    // still renders an "embedding disabled" message INSIDE the iframe, which
    // we can't detect directly (cross-origin). So we offer the fallback link
    // as a permanent safety net below the player instead.
  } catch (e) {}
}

function showEmbedFallback(videoId) {
  const wrapper = document.querySelector(".video-wrapper");
  wrapper.innerHTML = `
    <div class="embed-fallback">
      <img src="https://img.youtube.com/vi/${videoId}/hqdefault.jpg">
      <p>This video's owner disabled embedding.</p>
      <a href="https://www.youtube.com/watch?v=${videoId}" target="_blank">▶ Watch on YouTube</a>
    </div>
  `;
}

function renderIdentityChart(rates) {
  const container = document.getElementById("identityChart");
  container.innerHTML = "";

  if (rates.length === 0) {
    container.innerHTML = '<p style="color:#7c7f91; font-size:0.85rem;">No identity-tagged comments yet.</p>';
    return;
  }

  rates.forEach(item => {
    const row = document.createElement("div");
    row.className = "identity-row";
    row.innerHTML = `
      <span class="identity-name">${item.identity}</span>
      <div class="identity-bar-bg">
        <div class="identity-bar-fill" style="width:${item.rate}%"></div>
      </div>
      <span class="identity-pct">${item.rate}%</span>
    `;
    container.appendChild(row);
  });
}

function renderComments(comments) {
  const feed = document.getElementById("commentFeed");
  feed.innerHTML = "";

  if (comments.length === 0) {
    feed.innerHTML = '<p style="color:#7c7f91; font-size:0.85rem;">No comments yet — waiting for data...</p>';
    return;
  }

  comments.forEach(c => {
    const card = document.createElement("div");
    card.className = "comment-card" + (c.is_toxic ? " toxic" : "");

    const label = c.is_toxic ? "🔴 TOXIC" : "🟢 CLEAN";
    const labelClass = c.is_toxic ? "toxic" : "clean";
    const identityBadge = c.identity_flags ? ` · 🏷 ${c.identity_flags}` : "";

    card.innerHTML = `
      <div class="comment-top">
        <span class="comment-author">${escapeHtml(c.author || "Unknown")}</span>
        <span class="comment-label ${labelClass}">${label} (${c.toxicity_prob.toFixed(2)})</span>
      </div>
      <div class="comment-text">${escapeHtml(c.text || "")}</div>
      <div class="comment-meta">${c.scored_at}${identityBadge}</div>
    `;
    feed.appendChild(card);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function loadVideoMeta() {
  try {
    const res = await fetch("/api/video_meta");
    const meta = await res.json();
    const card = document.getElementById("videoInfoCard");
    if (meta.title) {
      document.getElementById("videoTitle").textContent = meta.title;
      document.getElementById("videoSub").textContent =
        `${meta.channel || ""} · ${meta.views || "?"} views · ${meta.likes || "?"} likes`;
      card.style.display = "block";
    }
  } catch (err) {
    console.error(err);
  }
}

async function refreshData() {
  try {
    const res = await fetch(`/api/data?filter=${currentFilter}`);
    const data = await res.json();

    document.getElementById("totalCount").textContent = data.total;
    document.getElementById("toxicityRate").textContent = data.toxicity_rate + "%";
    document.getElementById("identityCount").textContent = data.identity_count;
    document.getElementById("commentCount").textContent = `${data.comments.length} comments`;

    const statusEl = document.getElementById("statusText");
    if (data.status === "error") {
      statusEl.textContent = "⚠ Network hiccup, retrying...";
      statusEl.className = "status-text error";
    } else if (data.status === "polling") {
      statusEl.textContent = "● Live";
      statusEl.className = "status-text ok";
    }

    if (data.video_id && data.video_id !== currentVideoId) {
      currentVideoId = data.video_id;
      setVideoFrame(data.video_id);
      loadVideoMeta();

      const watchLink = document.getElementById("watchLink");
      const watchHref = document.getElementById("watchLinkHref");
      watchHref.href = `https://www.youtube.com/watch?v=${data.video_id}`;
      watchLink.style.display = "block";
    }

    renderIdentityChart(data.identity_rates);
    renderComments(data.comments);
  } catch (err) {
    console.error("Failed to fetch data:", err);
  }
}

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    refreshData();
  });
});

document.getElementById("goBtn").addEventListener("click", async () => {
  const url = document.getElementById("videoUrlInput").value.trim();
  const statusEl = document.getElementById("statusText");

  if (!url) {
    statusEl.textContent = "Please enter a video URL.";
    statusEl.className = "status-text error";
    return;
  }

  statusEl.textContent = "Starting...";
  statusEl.className = "status-text";

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const result = await res.json();

    if (result.ok) {
      currentVideoId = null;
      statusEl.textContent = "● Live";
      statusEl.className = "status-text ok";
    } else {
      statusEl.textContent = "⚠ " + result.error;
      statusEl.className = "status-text error";
    }
  } catch (err) {
    statusEl.textContent = "⚠ Could not reach server.";
    statusEl.className = "status-text error";
  }
});

setInterval(refreshData, 4000);
refreshData();