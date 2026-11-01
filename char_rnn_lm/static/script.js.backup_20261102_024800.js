/**
 * static/script.js
 * Frontend controller for MiniGPT Online Assistant.
 * Implements ChatGPT/Claude/Gemini-grade minimal UI, word-by-word streaming reveal,
 * session history in localStorage, offline inline SVGs, and responsive interactions.
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // DOM Element References (Preserved IDs)
    // =========================================================================
    const chatViewport = document.getElementById("chat-viewport");
    const messagesList = document.getElementById("messages-list");
    const welcomeCard = document.getElementById("welcome-card");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("btn-send");
    const toolBanner = document.getElementById("tool-activity-banner");
    const activityStatus = document.getElementById("activity-status-text");
    const activityQuery = document.getElementById("activity-query-text");
    const meterFill = document.getElementById("meter-fill");
    const tokenCounter = document.getElementById("token-counter");
    const tokenCountLabel = document.getElementById("token-count-label");
    const liveClockText = document.getElementById("live-clock-text");
    const modelParamsBadge = document.getElementById("model-params-badge");
    const modelStepBadge = document.getElementById("model-step-badge");

    // Sidebar & Navigation Elements
    const sidebar = document.getElementById("sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
    const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
    const btnNewChat = document.getElementById("btn-new-chat");
    const historyListToday = document.getElementById("history-list-today");
    const historyListEarlier = document.getElementById("history-list-earlier");

    // Action & Modal Buttons (Preserved IDs)
    const btnSpecs = document.getElementById("btn-specs");
    const btnLogs = document.getElementById("btn-logs");
    const btnClear = document.getElementById("btn-clear");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const btnScrollBottom = document.getElementById("btn-scroll-bottom");

    // Modals
    const specsModalOverlay = document.getElementById("specs-modal-overlay");
    const specsModalBody = document.getElementById("specs-modal-body");
    const specsCloseBtn = document.getElementById("specs-close-btn");
    const logsModalOverlay = document.getElementById("logs-modal-overlay");
    const logsModalBody = document.getElementById("logs-modal-body");
    const logsCloseBtn = document.getElementById("logs-close-btn");
    const btnRefreshLogs = document.getElementById("btn-refresh-logs");

    // State Variables
    let isGenerating = false;
    let stopGeneration = false;
    let modelSpecsCache = null;
    let currentSessionId = null;
    let chatSessions = [];
    const MAX_TOKENS = 128;
    const RING_CIRCUMFERENCE = 62.83; // 2 * PI * 10

    // =========================================================================
    // 1. Theme Management (Dark / Light with localStorage persistence)
    // =========================================================================
    function initTheme() {
        const savedTheme = localStorage.getItem("minigpt_theme");
        if (savedTheme) {
            document.documentElement.setAttribute("data-theme", savedTheme);
        } else {
            const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
            document.documentElement.setAttribute("data-theme", prefersLight ? "light" : "dark");
        }
    }

    btnThemeToggle?.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("minigpt_theme", newTheme);
    });

    initTheme();

    // =========================================================================
    // 2. Live System Clock (Preserved)
    // =========================================================================
    function updateClock() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, "0");
        const day = String(now.getDate()).padStart(2, "0");
        const hours = String(now.getHours()).padStart(2, "0");
        const minutes = String(now.getMinutes()).padStart(2, "0");
        const seconds = String(now.getSeconds()).padStart(2, "0");
        if (liveClockText) {
            liveClockText.textContent = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // =========================================================================
    // 3. Model Status Fetch (/api/status - Preserved Endpoint)
    // =========================================================================
    async function fetchModelStatus() {
        try {
            // [HOOK: Existing Flask Endpoint /api/status]
            const resp = await fetch("/api/status");
            if (resp.ok) {
                modelSpecsCache = await resp.json();
                if (modelSpecsCache.params && modelParamsBadge) {
                    const mParams = (modelSpecsCache.params / 1000000).toFixed(2);
                    modelParamsBadge.textContent = `${mParams}M params`;
                }
                if (modelSpecsCache.step && modelStepBadge) {
                    modelStepBadge.textContent = `Step ${modelSpecsCache.step} • Tool-Augmented`;
                }
            }
        } catch (e) {
            console.warn("Could not fetch model status:", e);
        }
    }
    fetchModelStatus();

    // =========================================================================
    // 4. Auto-Resize Textarea & Circular Token Counter Ring (Preserved Calculation)
    // =========================================================================
    function updateTokenMeter() {
        const text = userInput.value;
        // Strict 128-token context window safety budget (~3.6 chars/token)
        const estTokens = text.trim() ? Math.ceil(text.trim().length / 3.6) : 0;
        const pct = Math.min((estTokens / MAX_TOKENS) * 100, 100);

        if (meterFill) {
            const offset = RING_CIRCUMFERENCE - (pct / 100) * RING_CIRCUMFERENCE;
            meterFill.style.strokeDashoffset = offset;

            if (estTokens >= 120) {
                meterFill.style.stroke = "var(--danger)";
            } else if (estTokens >= 102) {
                meterFill.style.stroke = "var(--warning)";
            } else {
                meterFill.style.stroke = "var(--accent)";
            }
        }

        if (tokenCounter) {
            tokenCounter.setAttribute("title", `${estTokens} / ${MAX_TOKENS} tokens`);
        }
        if (tokenCountLabel) {
            tokenCountLabel.textContent = `${estTokens} / ${MAX_TOKENS} tok`;
        }

        // Enable or disable send button
        if (!isGenerating) {
            sendBtn.disabled = text.trim().length === 0;
        }
    }

    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
        updateTokenMeter();
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating && userInput.value.trim().length > 0) {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener("click", () => {
        if (isGenerating) {
            // Stop generating action
            stopGeneration = true;
        } else {
            sendMessage();
        }
    });

    // =========================================================================
    // 5. Chat Sessions in localStorage (Grouped by Today / Earlier)
    // =========================================================================
    function loadSessionsFromStorage() {
        try {
            const data = localStorage.getItem("minigpt_chat_sessions");
            chatSessions = data ? JSON.parse(data) : [];
        } catch {
            chatSessions = [];
        }
    }

    function saveSessionsToStorage() {
        try {
            localStorage.setItem("minigpt_chat_sessions", JSON.stringify(chatSessions));
        } catch (e) {
            console.warn("Storage quota exceeded", e);
        }
    }

    function createNewSession() {
        const newId = "session_" + Date.now();
        currentSessionId = newId;
        const newSession = {
            id: newId,
            title: "New conversation",
            timestamp: Date.now(),
            messages: []
        };
        chatSessions.unshift(newSession);
        saveSessionsToStorage();
        renderHistorySidebar();
        renderCurrentSessionMessages();
    }

    function isToday(timestamp) {
        const date = new Date(timestamp);
        const today = new Date();
        return (
            date.getDate() === today.getDate() &&
            date.getMonth() === today.getMonth() &&
            date.getFullYear() === today.getFullYear()
        );
    }

    function renderHistorySidebar() {
        if (!historyListToday || !historyListEarlier) return;
        historyListToday.innerHTML = "";
        historyListEarlier.innerHTML = "";

        let todayCount = 0;
        let earlierCount = 0;

        chatSessions.forEach(session => {
            const item = document.createElement("div");
            item.className = `history-item ${session.id === currentSessionId ? "active" : ""}`;

            const title = document.createElement("span");
            title.className = "history-item-title";
            title.textContent = session.title || "Conversation";
            item.appendChild(title);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "history-item-delete";
            deleteBtn.title = "Delete conversation";
            deleteBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            `;
            deleteBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteSession(session.id);
            });
            item.appendChild(deleteBtn);

            item.addEventListener("click", () => {
                if (currentSessionId !== session.id) {
                    currentSessionId = session.id;
                    renderHistorySidebar();
                    renderCurrentSessionMessages();
                    closeSidebarOnMobile();
                }
            });

            if (isToday(session.timestamp)) {
                historyListToday.appendChild(item);
                todayCount++;
            } else {
                historyListEarlier.appendChild(item);
                earlierCount++;
            }
        });

        const groupToday = document.getElementById("history-group-today");
        const groupEarlier = document.getElementById("history-group-earlier");
        if (groupToday) groupToday.style.display = todayCount > 0 ? "flex" : "none";
        if (groupEarlier) groupEarlier.style.display = earlierCount > 0 ? "flex" : "none";
    }

    function deleteSession(sessionId) {
        chatSessions = chatSessions.filter(s => s.id !== sessionId);
        saveSessionsToStorage();
        if (currentSessionId === sessionId) {
            if (chatSessions.length > 0) {
                currentSessionId = chatSessions[0].id;
            } else {
                createNewSession();
                return;
            }
        }
        renderHistorySidebar();
        renderCurrentSessionMessages();
    }

    function renderCurrentSessionMessages() {
        messagesList.innerHTML = "";
        const session = chatSessions.find(s => s.id === currentSessionId);
        if (!session || session.messages.length === 0) {
            welcomeCard.style.display = "flex";
            return;
        }

        welcomeCard.style.display = "none";
        session.messages.forEach(msg => {
            appendMessage(msg, false);
        });
        scrollChatToBottom();
    }

    btnNewChat?.addEventListener("click", () => {
        createNewSession();
        closeSidebarOnMobile();
        userInput.focus();
    });

    // =========================================================================
    // 6. Mobile Sidebar Drawer Logic
    // =========================================================================
    function openSidebar() {
        sidebar?.classList.add("open");
        sidebarOverlay?.classList.add("open");
    }

    function closeSidebarOnMobile() {
        sidebar?.classList.remove("open");
        sidebarOverlay?.classList.remove("open");
    }

    sidebarToggleBtn?.addEventListener("click", openSidebar);
    sidebarCloseBtn?.addEventListener("click", closeSidebarOnMobile);
    sidebarOverlay?.addEventListener("click", closeSidebarOnMobile);

    // =========================================================================
    // 7. Suggestion Chips (Preserved event binding)
    // =========================================================================
    document.querySelectorAll(".prompt-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query && !isGenerating) {
                userInput.value = query;
                userInput.dispatchEvent(new Event("input"));
                sendMessage();
            }
        });
    });

    // =========================================================================
    // 8. Send Message Workflow (/api/chat - Preserved Endpoint)
    // =========================================================================
    async function sendMessage(overrideText = null) {
        const text = (overrideText !== null ? overrideText : userInput.value).trim();
        if (!text || isGenerating) return;

        isGenerating = true;
        stopGeneration = false;
        sendBtn.classList.add("generating");
        sendBtn.disabled = false;
        userInput.value = "";
        userInput.style.height = "auto";
        updateTokenMeter();

        // Ensure session exists
        let currentSession = chatSessions.find(s => s.id === currentSessionId);
        if (!currentSession) {
            createNewSession();
            currentSession = chatSessions.find(s => s.id === currentSessionId);
        }

        // Set session title from first prompt
        if (currentSession && currentSession.messages.length === 0) {
            currentSession.title = text.length > 30 ? text.slice(0, 30) + "..." : text;
            renderHistorySidebar();
        }

        // Hide Welcome Card
        if (welcomeCard) welcomeCard.style.display = "none";

        // Append User Message
        const userMsg = {
            role: "user",
            content: text,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        };
        appendMessage(userMsg);
        currentSession.messages.push(userMsg);
        saveSessionsToStorage();

        // Show Predictive Tool Activity Bar
        showToolIndicator(text);

        // Append Temporary Assistant Typing Indicator
        const typingRow = appendTypingIndicator();
        scrollChatToBottom();

        try {
            // [HOOK: Existing Flask Endpoint /api/chat - POST]
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });

            const data = await resp.json();
            hideToolIndicator();
            removeTypingIndicator(typingRow);

            if (!resp.ok) {
                const errorMsg = {
                    role: "assistant",
                    content: `⚠️ Error: ${data.error || "Failed to generate response"}`,
                    route: "error",
                    latency: 0
                };
                appendMessage(errorMsg);
                currentSession.messages.push(errorMsg);
            } else {
                const botMsg = {
                    role: "assistant",
                    content: data.response,
                    route: data.route,
                    sources: data.sources || [],
                    toolCall: data.tool_call,
                    timestamp: data.timestamp,
                    latency: data.latency_ms
                };

                // Stream reply word-by-word into the chat UI
                await streamAssistantMessage(botMsg);
                currentSession.messages.push(botMsg);
            }
            saveSessionsToStorage();
        } catch (err) {
            hideToolIndicator();
            removeTypingIndicator(typingRow);
            const connErrorMsg = {
                role: "assistant",
                content: "⚠️ Connection Error: Could not reach agent server.",
                route: "error",
                latency: 0
            };
            appendMessage(connErrorMsg);
            currentSession.messages.push(connErrorMsg);
            saveSessionsToStorage();
        } finally {
            isGenerating = false;
            stopGeneration = false;
            sendBtn.classList.remove("generating");
            updateTokenMeter();
            userInput.focus();
        }
    }

    // =========================================================================
    // 9. Tool Indicator Logic (Preserved predictive patterns)
    // =========================================================================
    function showToolIndicator(query) {
        if (!toolBanner) return;
        const qLower = query.toLowerCase();
        const searchKeywords = ["latest", "today", "aaj", "news", "price", "weather", "score", "match", "winner"];
        const timeKeywords = ["time", "date", "samay", "tarikh", "ghadi"];

        if (timeKeywords.some(kw => qLower.includes(kw))) {
            activityStatus.textContent = "Synchronizing System Clock...";
            activityQuery.textContent = "";
        } else if (searchKeywords.some(kw => qLower.includes(kw))) {
            activityStatus.textContent = "Searching Web (DuckDuckGo)...";
            activityQuery.textContent = `"${query.slice(0, 36)}"`;
        } else {
            activityStatus.textContent = "MiniGPT Transformer Thinking...";
            activityQuery.textContent = "(6 Layers • 192 Dim)";
        }
        toolBanner.classList.add("active");
    }

    function hideToolIndicator() {
        toolBanner?.classList.remove("active");
    }

    // =========================================================================
    // 10. Message Rendering & Word-by-Word Reveal
    // =========================================================================
    function appendTypingIndicator() {
        const row = document.createElement("div");
        row.className = "chat-message-row assistant-row typing-row";
        row.innerHTML = `
            <div class="bot-avatar">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/>
                    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/>
                </svg>
            </div>
            <div class="bot-content-wrap">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        messagesList.appendChild(row);
        return row;
    }

    function removeTypingIndicator(row) {
        if (row && row.parentNode) {
            row.parentNode.removeChild(row);
        }
    }

    async function streamAssistantMessage(msg) {
        const row = document.createElement("div");
        row.className = "chat-message-row assistant-row";

        // Bot Avatar
        const avatar = document.createElement("div");
        avatar.className = "bot-avatar";
        avatar.innerHTML = `
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/>
                <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/>
            </svg>
        `;
        row.appendChild(avatar);

        const contentWrap = document.createElement("div");
        contentWrap.className = "bot-content-wrap";

        const textBody = document.createElement("div");
        textBody.className = "bot-text-body";
        contentWrap.appendChild(textBody);
        row.appendChild(contentWrap);
        messagesList.appendChild(row);

        // Word-by-word streaming reveal animation
        const rawContent = msg.content || "";
        const words = rawContent.split(" ");
        let currentText = "";

        for (let i = 0; i < words.length; i++) {
            if (stopGeneration) {
                // Instantly dump full markdown if stopped
                break;
            }
            currentText += (i === 0 ? "" : " ") + words[i];
            textBody.innerHTML = parseMarkdown(currentText);
            scrollChatToBottom(false);
            await new Promise(r => setTimeout(r, 22));
        }

        // Final full parse
        textBody.innerHTML = parseMarkdown(rawContent);

        // Metadata & Actions row
        const footerRow = createBotFooterRow(msg);
        contentWrap.appendChild(footerRow);

        // Sources if available
        if (msg.sources && msg.sources.length > 0) {
            contentWrap.appendChild(createSourcesElement(msg.sources));
        }

        scrollChatToBottom();
    }

    function appendMessage(msg, shouldScroll = true) {
        const row = document.createElement("div");
        row.className = `chat-message-row ${msg.role === "user" ? "user-row" : "assistant-row"}`;

        if (msg.role === "user") {
            const bubble = document.createElement("div");
            bubble.className = "user-bubble";
            bubble.textContent = msg.content;
            row.appendChild(bubble);
        } else {
            const avatar = document.createElement("div");
            avatar.className = "bot-avatar";
            avatar.innerHTML = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/>
                    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/>
                </svg>
            `;
            row.appendChild(avatar);

            const contentWrap = document.createElement("div");
            contentWrap.className = "bot-content-wrap";

            const textBody = document.createElement("div");
            textBody.className = "bot-text-body";
            textBody.innerHTML = parseMarkdown(msg.content);
            contentWrap.appendChild(textBody);

            const footerRow = createBotFooterRow(msg);
            contentWrap.appendChild(footerRow);

            if (msg.sources && msg.sources.length > 0) {
                contentWrap.appendChild(createSourcesElement(msg.sources));
            }

            row.appendChild(contentWrap);
        }

        messagesList.appendChild(row);
        if (shouldScroll) {
            scrollChatToBottom();
        }
    }

    function createBotFooterRow(msg) {
        const footer = document.createElement("div");
        footer.className = "bot-footer-row";

        // Muted Mono Label
        const metaTag = document.createElement("span");
        metaTag.className = "bot-meta-tag";

        let routeText = "Neural weights";
        let routeSvg = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
        `;

        if (msg.route === "clock") {
            routeText = "Live clock";
            routeSvg = `
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
            `;
        } else if (msg.route === "search_rule" || msg.route === "model_tool" || msg.route === "fallback_search" || msg.route === "search") {
            routeText = "Web search";
            routeSvg = `
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="2" y1="12" x2="22" y2="12"></line>
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                </svg>
            `;
        } else if (msg.route === "memory") {
            routeText = "Memory";
            routeSvg = `
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            `;
        }

        const latencyDisplay = msg.latency ? `${msg.latency}ms` : "";
        metaTag.innerHTML = `${routeSvg}<span>${routeText}${latencyDisplay ? " · " + latencyDisplay : ""}</span>`;
        footer.appendChild(metaTag);

        // Action Buttons Group (Copy, Regenerate, Thumbs)
        const actionGroup = document.createElement("div");
        actionGroup.className = "bot-action-group";

        // Copy button
        const copyBtn = document.createElement("button");
        copyBtn.className = "action-icon-btn";
        copyBtn.title = "Copy message";
        copyBtn.setAttribute("aria-label", "Copy message");
        copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        `;
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(msg.content);
            copyBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="var(--success)" stroke-width="2.5">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
            setTimeout(() => {
                copyBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                `;
            }, 1500);
        });
        actionGroup.appendChild(copyBtn);

        // Regenerate button
        const regenBtn = document.createElement("button");
        regenBtn.className = "action-icon-btn";
        regenBtn.title = "Regenerate response";
        regenBtn.setAttribute("aria-label", "Regenerate response");
        regenBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
        `;
        regenBtn.addEventListener("click", () => {
            // Find preceding user query
            const session = chatSessions.find(s => s.id === currentSessionId);
            if (session) {
                const userTurns = session.messages.filter(m => m.role === "user");
                if (userTurns.length > 0) {
                    const lastUserPrompt = userTurns[userTurns.length - 1].content;
                    sendMessage(lastUserPrompt);
                }
            }
        });
        actionGroup.appendChild(regenBtn);

        // Thumbs Up
        const thumbsUp = document.createElement("button");
        thumbsUp.className = "action-icon-btn";
        thumbsUp.title = "Good response";
        thumbsUp.setAttribute("aria-label", "Good response");
        thumbsUp.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
            </svg>
        `;
        thumbsUp.addEventListener("click", () => {
            thumbsUp.classList.toggle("active");
            thumbsDown.classList.remove("active");
        });
        actionGroup.appendChild(thumbsUp);

        // Thumbs Down
        const thumbsDown = document.createElement("button");
        thumbsDown.className = "action-icon-btn";
        thumbsDown.title = "Bad response";
        thumbsDown.setAttribute("aria-label", "Bad response");
        thumbsDown.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
            </svg>
        `;
        thumbsDown.addEventListener("click", () => {
            thumbsDown.classList.toggle("active");
            thumbsUp.classList.remove("active");
        });
        actionGroup.appendChild(thumbsDown);

        footer.appendChild(actionGroup);
        return footer;
    }

    function createSourcesElement(sources) {
        const wrap = document.createElement("div");
        wrap.className = "bot-sources-list";

        sources.forEach(url => {
            try {
                const parsed = new URL(url);
                const chip = document.createElement("a");
                chip.className = "source-chip";
                chip.href = url;
                chip.target = "_blank";
                chip.rel = "noopener noreferrer";
                chip.innerHTML = `
                    <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                    <span>${parsed.hostname}</span>
                `;
                wrap.appendChild(chip);
            } catch {
                // Ignore invalid URLs
            }
        });
        return wrap;
    }

    // =========================================================================
    // 11. Minimal Markdown Parser (XSS Safe)
    // =========================================================================
    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function parseMarkdown(text) {
        if (!text) return "";

        // Escape raw HTML first to eliminate XSS risks
        let escaped = escapeHtml(text);

        // Fenced Code Blocks (```lang ... ```)
        const codeBlocks = [];
        escaped = escaped.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const token = `__CODE_BLOCK_${codeBlocks.length}__`;
            codeBlocks.push({ lang: lang.trim(), code: code });
            return token;
        });

        // Inline Code (`code`)
        escaped = escaped.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Bold (**text**)
        escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

        // Italic (*text*)
        escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");

        // Line-by-line list and paragraph handling
        const lines = escaped.split("\n");
        const parsedLines = [];
        let inList = false;
        let listTag = "";

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const bulletMatch = line.match(/^(\s*)[-*•]\s+(.*)$/);
            const numberMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);

            if (bulletMatch) {
                if (!inList || listTag !== "ul") {
                    if (inList) parsedLines.push(`</${listTag}>`);
                    parsedLines.push("<ul>");
                    inList = true;
                    listTag = "ul";
                }
                parsedLines.push(`<li>${bulletMatch[2]}</li>`);
            } else if (numberMatch) {
                if (!inList || listTag !== "ol") {
                    if (inList) parsedLines.push(`</${listTag}>`);
                    parsedLines.push("<ol>");
                    inList = true;
                    listTag = "ol";
                }
                parsedLines.push(`<li>${numberMatch[2]}</li>`);
            } else {
                if (inList) {
                    parsedLines.push(`</${listTag}>`);
                    inList = false;
                    listTag = "";
                }
                parsedLines.push(line);
            }
        }
        if (inList) {
            parsedLines.push(`</${listTag}>`);
        }

        let html = parsedLines.join("\n");

        // Format paragraphs on double newline
        const blocks = html.split(/\n\s*\n/);
        html = blocks
            .map(block => {
                const b = block.trim();
                if (!b) return "";
                if (b.startsWith("<ul>") || b.startsWith("<ol>") || b.startsWith("__CODE_BLOCK_")) {
                    return b;
                }
                return `<p>${b.replace(/\n/g, "<br>")}</p>`;
            })
            .filter(Boolean)
            .join("\n");

        // Reinsert Code Blocks with copy buttons
        codeBlocks.forEach((cb, idx) => {
            const token = `__CODE_BLOCK_${idx}__`;
            const replacement = `
                <div class="code-block">
                    <div class="code-header">
                        <span>${cb.lang || "code"}</span>
                        <button class="code-copy-btn" onclick="copyCodeSnippet(this)">
                            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        </button>
                    </div>
                    <pre><code>${cb.code}</code></pre>
                </div>
            `;
            html = html.replace(token, replacement);
        });

        return html;
    }

    // Global helper for code snippet copy
    window.copyCodeSnippet = function(button) {
        const codeElement = button.closest(".code-block").querySelector("pre code");
        if (codeElement) {
            navigator.clipboard.writeText(codeElement.textContent);
            const label = button.querySelector("span");
            if (label) {
                const prev = label.textContent;
                label.textContent = "Copied!";
                setTimeout(() => { label.textContent = prev; }, 1500);
            }
        }
    };

    // =========================================================================
    // 12. Smart Scroll Management
    // =========================================================================
    function isNearBottom() {
        if (!chatViewport) return true;
        const threshold = 90;
        return chatViewport.scrollHeight - chatViewport.scrollTop - chatViewport.clientHeight <= threshold;
    }

    function scrollChatToBottom(force = true) {
        if (!chatViewport) return;
        if (force || isNearBottom()) {
            chatViewport.scrollTop = chatViewport.scrollHeight;
        }
    }

    chatViewport?.addEventListener("scroll", () => {
        if (!btnScrollBottom) return;
        if (!isNearBottom()) {
            btnScrollBottom.classList.add("visible");
        } else {
            btnScrollBottom.classList.remove("visible");
        }
    });

    btnScrollBottom?.addEventListener("click", () => {
        chatViewport.scrollTo({ top: chatViewport.scrollHeight, behavior: "smooth" });
    });

    // =========================================================================
    // 13. Architecture Specs Modal (/api/status - Preserved)
    // =========================================================================
    function openSpecsModal() {
        if (!specsModalOverlay) return;
        specsModalOverlay.classList.add("active");
        renderSpecsTable();
    }

    function closeSpecsModal() {
        specsModalOverlay?.classList.remove("active");
    }

    async function renderSpecsTable() {
        if (!modelSpecsCache) {
            await fetchModelStatus();
        }
        const s = modelSpecsCache || {};
        specsModalBody.innerHTML = `
            <table class="specs-table">
                <tr><td class="spec-key">Model Type</td><td class="spec-val">Decoder-Only Causal Transformer</td></tr>
                <tr><td class="spec-key">Parameters</td><td class="spec-val">${(s.params || 3076992).toLocaleString()} (~3.08M)</td></tr>
                <tr><td class="spec-key">Transformer Layers</td><td class="spec-val">${s.n_layer || 6}</td></tr>
                <tr><td class="spec-key">Attention Heads</td><td class="spec-val">${s.n_head || 6}</td></tr>
                <tr><td class="spec-key">Embedding Dim (d_model)</td><td class="spec-val">${s.n_embd || 192}</td></tr>
                <tr><td class="spec-key">Context Window (block_size)</td><td class="spec-val">${s.block_size || 128} tokens</td></tr>
                <tr><td class="spec-key">Vocabulary Size</td><td class="spec-val">2,048 (Byte-Level BPE)</td></tr>
                <tr><td class="spec-key">Checkpoint File</td><td class="spec-val">${s.checkpoint || 'tool_gpt.pt'}</td></tr>
                <tr><td class="spec-key">Fine-Tuned Step</td><td class="spec-val">Step ${s.step || 180}</td></tr>
                <tr><td class="spec-key">Validation Loss</td><td class="spec-val">${s.val_loss || 1.1534}</td></tr>
                <tr><td class="spec-key">Compute Hardware</td><td class="spec-val">${(s.device || 'CPU').toUpperCase()}</td></tr>
                <tr><td class="spec-key">Connected Tools</td><td class="spec-val">DuckDuckGo Search • Live Clock • Memory</td></tr>
            </table>
            <div style="font-size: 12px; color: var(--text-faint); line-height: 1.6;">
                Built from scratch in PyTorch with strict 128-token context budgeting. No external API keys required.
            </div>
        `;
    }

    btnSpecs?.addEventListener("click", () => {
        openSpecsModal();
        closeSidebarOnMobile();
    });
    specsCloseBtn?.addEventListener("click", closeSpecsModal);
    specsModalOverlay?.addEventListener("click", (e) => {
        if (e.target === specsModalOverlay) closeSpecsModal();
    });

    // =========================================================================
    // 14. Audit Logs Modal (/api/logs - Preserved Endpoint)
    // =========================================================================
    function openLogsModal() {
        if (!logsModalOverlay) return;
        logsModalOverlay.classList.add("active");
        fetchAndRenderLogs();
    }

    function closeLogsModal() {
        logsModalOverlay?.classList.remove("active");
    }

    async function fetchAndRenderLogs() {
        logsModalBody.innerHTML = `
            <div style="color: var(--text-muted); text-align: center; padding: 28px;">
                Loading audit logs...
            </div>
        `;

        try {
            // [HOOK: Existing Flask Endpoint /api/logs - GET]
            const resp = await fetch("/api/logs");
            const data = await resp.json();
            const logs = data.logs || [];

            if (logs.length === 0) {
                logsModalBody.innerHTML = `
                    <div style="color: var(--text-muted); text-align: center; padding: 28px;">
                        No logs recorded yet.
                    </div>
                `;
                return;
            }

            let html = "";
            logs.forEach(logItem => {
                html += `<div class="log-entry-item">${escapeHtml(logItem)}</div>`;
            });
            logsModalBody.innerHTML = html;
        } catch (e) {
            logsModalBody.innerHTML = `
                <div style="color: var(--danger); padding: 12px;">
                    Failed to fetch logs: ${escapeHtml(e.message)}
                </div>
            `;
        }
    }

    btnLogs?.addEventListener("click", () => {
        openLogsModal();
        closeSidebarOnMobile();
    });
    btnRefreshLogs?.addEventListener("click", fetchAndRenderLogs);
    logsCloseBtn?.addEventListener("click", closeLogsModal);
    logsModalOverlay?.addEventListener("click", (e) => {
        if (e.target === logsModalOverlay) closeLogsModal();
    });

    // Global Escape Key listener for closing all modals
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeSpecsModal();
            closeLogsModal();
            closeSidebarOnMobile();
        }
    });

    // =========================================================================
    // 15. Clear Chat Action (Preserved ID btn-clear)
    // =========================================================================
    btnClear?.addEventListener("click", () => {
        if (confirm("Clear current conversation messages?")) {
            const session = chatSessions.find(s => s.id === currentSessionId);
            if (session) {
                session.messages = [];
                session.title = "New conversation";
                saveSessionsToStorage();
                renderHistorySidebar();
            }
            messagesList.innerHTML = "";
            if (welcomeCard) welcomeCard.style.display = "flex";
            userInput.focus();
        }
    });

    // =========================================================================
    // 16. Initialize Chat Session
    // =========================================================================
    loadSessionsFromStorage();
    if (chatSessions.length === 0) {
        createNewSession();
    } else {
        currentSessionId = chatSessions[0].id;
        renderHistorySidebar();
        renderCurrentSessionMessages();
    }
});
