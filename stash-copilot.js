(function() {
    'use strict';

    // Plugin configuration
    const PLUGIN_ID = 'stash-copilot';
    const PLUGIN_NAME = 'My Plugin';
    const SUMMARY_FILE = '/plugin/stash-copilot/assets/last_summary.json';
    const CHAT_FILE = '/plugin/stash-copilot/assets/chat_history.json';
    const SCENE_VISION_PATH = '/plugin/stash-copilot/assets/scene_vision';

    // State management
    const state = {
        initialized: false,
        lastSummary: null,
        isGenerating: false,
        generationStartTime: null,
        // Chat state
        activeTab: 'summary',
        chatHistory: null,
        conversationId: null,
        isChatting: false,
        chatStartTime: null,
        chatPollInterval: null,
        lastRenderedMessageCount: 0,
        // Recommendations state
        recommendationsMode: localStorage.getItem('stash-copilot-rec-mode') || 'discover_new',
        recommendationsTimeDecayDays: (() => {
            const stored = localStorage.getItem('stash-copilot-time-decay-days');
            if (stored === 'session') return 'session';
            return parseInt(stored) || 28;
        })(),
        recommendationsResults: null,
        recommendationsProfile: null,
        recommendationsGeneratedAt: null,
        recommendationsCache: {}, // Per-mode cache: { discover_new: {results, profile, generatedAt}, rewatch: {...} }
        isGeneratingRecommendations: false,
        recommendationsRequestId: null,
        recommendationsPollInterval: null,
        recommendationsPage: 1,
        recommendationsPerPage: 12,
        // Unified recommendations view (modal only)
        recommendationsViewFilter: 'all',         // 'all' | 'new' | 'rewatch'
        recommendationsDiscoverResults: [],
        recommendationsRewatchResults: [],
        recommendationsMergedResults: [],
        recommendationsDiscoverRequestId: null,
        recommendationsRewatchRequestId: null,
        // Peak Moments state
        peakResults: null,
        peakPage: 1,
        peakTotalPages: 1,
        isGeneratingPeak: false,
        peakRequestId: null,
        peakPollInterval: null,
        // Taste Map state
        tasteMapData: null,
        tasteMapRequestId: null,
        tasteMapLoading: false,
        tasteMapChart: null,
        // Tag Gaps state
        tagGapsLoading: false,
        tagGapsData: null,
        tagGapsRequestId: null,
        // AI Insights Modal state
        insightsModalOpen: false,
        // Tag dedup state
        tagDedupCandidates: [],
        tagDedupCurrentIndex: 0,
        tagDedupMergeCount: 0,
        tagDedupSkipCount: 0,
        tagDedupScenesUpdated: 0,
        tagDedupPollInterval: null,
        tagDedupRequestId: null,
        tagDedupProcessing: false
    };

    // Scene vision state (separate from main state)
    const visionState = {
        sceneId: null,
        conversationId: null,
        isAnalyzing: false,
        analysisStartTime: null,
        hasSeenNewAnalysisStart: false,  // Track if we've seen the new analysis start (for re-analysis)
        resultRendered: false,  // Track if result has been rendered to prevent progress updates after completion
        description: null,
        suggestedTags: [],
        tagTimestamps: {},  // Map of tag name -> timestamp in seconds
        messages: [],
        modalOpen: false,
        pendingMessage: null  // Track pending follow-up message to prevent stale renders
    };

    // Semantic search state
    const searchState = {
        query: '',
        allResults: [],         // All pre-fetched results (10 pages worth)
        isSearching: false,
        currentPage: 1,
        perPage: 24,
        pagesPerBatch: 10,      // Pre-fetch 10 pages at a time
        totalFetched: 0,        // Total results fetched so far
        hasMoreOnServer: true,  // Server has more results to fetch
        requestId: null,
        pollInterval: null,
        lastQuery: localStorage.getItem('stash-copilot-last-search') || '',
        // Model selection for embedding comparison
        availableModels: [],    // List of {model_key, count, dimensions} from backend
        selectedModel: localStorage.getItem('stash-copilot-selected-model') || '',  // Selected model_key
        modelsLoaded: false,    // Flag to prevent duplicate loading
        // Frame-level search toggle
        frameSearch: localStorage.getItem('stash-copilot-frame-search') === 'true'
    };

    // Markdown library state (loaded from CDN)
    const markdownLibs = {
        marked: null,
        DOMPurify: null,
        loading: false,
        loaded: false,
        callbacks: []
    };

    // Custom prompts localStorage key
    const CUSTOM_PROMPTS_KEY = 'stash-copilot-custom-prompts';

    /**
     * Load custom prompts from localStorage
     */
    function loadCustomPrompts() {
        try {
            const stored = localStorage.getItem(CUSTOM_PROMPTS_KEY);
            return stored ? JSON.parse(stored) : {};
        } catch (e) {
            console.error('Failed to load custom prompts:', e);
            return {};
        }
    }

    /**
     * Save custom prompts to localStorage
     */
    function saveCustomPrompts(prompts) {
        try {
            localStorage.setItem(CUSTOM_PROMPTS_KEY, JSON.stringify(prompts));
        } catch (e) {
            console.error('Failed to save custom prompts:', e);
        }
    }

    /**
     * Reset custom prompts to defaults
     */
    function resetCustomPrompts() {
        localStorage.removeItem(CUSTOM_PROMPTS_KEY);
    }

    /**
     * Get current analysis options from UI
     */
    function getAnalysisOptions(container) {
        const quickMode = container.querySelector('.stash-copilot-option-quick-mode')?.checked || false;
        const skipVerification = container.querySelector('.stash-copilot-option-skip-verification')?.checked || false;
        const frameCountSelect = container.querySelector('.stash-copilot-option-frame-count');
        const frameCount = frameCountSelect?.value === 'auto' ? null : parseInt(frameCountSelect?.value || '0', 10) || null;

        return {
            quick_mode: quickMode,
            skip_verification: skipVerification,
            frame_count: frameCount,
            custom_prompts: loadCustomPrompts()
        };
    }

    // Load Plotly.js GL3D bundle for 3D taste map visualization
    function loadPlotly() {
        return new Promise((resolve, reject) => {
            if (window.Plotly) {
                resolve(window.Plotly);
                return;
            }
            const script = document.createElement('script');
            script.src = '/plugin/stash-copilot/assets/plotly-gl3d.min.js';
            script.onload = () => {
                log('Plotly.js GL3D loaded');
                resolve(window.Plotly);
            };
            script.onerror = () => reject(new Error('Failed to load Plotly'));
            document.head.appendChild(script);
        });
    }

    // Load external markdown libraries from CDN
    function loadMarkdownLibraries() {
        if (markdownLibs.loaded) return Promise.resolve();
        if (markdownLibs.loading) {
            return new Promise(resolve => markdownLibs.callbacks.push(resolve));
        }

        markdownLibs.loading = true;

        const loadScript = (src) => new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed to load ${src}`));
            document.head.appendChild(script);
        });

        // Load from local assets (CSP blocks external CDNs)
        return Promise.all([
            loadScript('/plugin/stash-copilot/assets/marked.min.js'),
            loadScript('/plugin/stash-copilot/assets/purify.min.js')
        ]).then(() => {
            markdownLibs.marked = window.marked;
            markdownLibs.DOMPurify = window.DOMPurify;
            markdownLibs.loaded = true;
            markdownLibs.loading = false;

            // Configure marked options
            markdownLibs.marked.setOptions({
                gfm: true,           // GitHub Flavored Markdown
                breaks: true,        // Convert \n to <br>
                headerIds: false,    // No auto-generated IDs (cleaner output)
                mangle: false        // Don't mangle email addresses
            });

            // Add DOMPurify hook to make links safe
            markdownLibs.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
                if (node.tagName === 'A') {
                    node.setAttribute('target', '_blank');
                    node.setAttribute('rel', 'noopener noreferrer');
                    node.classList.add('stash-copilot-link');
                }
            });

            log('Markdown libraries loaded successfully');
            markdownLibs.callbacks.forEach(cb => cb());
            markdownLibs.callbacks = [];
        }).catch(err => {
            log(`Failed to load markdown libraries: ${err.message}`, 'error');
            markdownLibs.loading = false;
            throw err;
        });
    }

    // Available tools with descriptions and example prompts
    const AVAILABLE_TOOLS = [
        {
            name: "query_performer_tags",
            description: "Query tags associated with a performer's scenes, ranked by view frequency",
            examples: ["What tags are associated with cutegeekie?", "Show me themes in MissAlice's content"]
        },
        {
            name: "query_tag_performers",
            description: "Find performers who have scenes with a specific tag",
            examples: ["Who has scenes tagged 'POV'?", "Which performers have 'blonde' content?"]
        },
        {
            name: "query_viewing_stats",
            description: "Get overall viewing statistics for your library",
            examples: ["What are my viewing statistics?", "How much content have I watched?"]
        },
        {
            name: "query_top_performers",
            description: "Get top performers ranked by view count, scene count, or o-count",
            examples: ["Who are my most watched performers?", "Top 10 performers by play count"]
        },
        {
            name: "query_top_tags",
            description: "Get top tags ranked by various metrics",
            examples: ["What are my most watched tags?", "Which tags do I watch most often?"]
        },
        {
            name: "query_top_studios",
            description: "Get top studios ranked by view count or scene count",
            examples: ["What studios do I watch the most?", "Show my top studios"]
        },
        {
            name: "query_library_stats",
            description: "Get an overview of your entire library (counts, sizes, durations)",
            examples: ["How big is my library?", "Give me an overview of my collection"]
        },
        {
            name: "query_watching_patterns",
            description: "Analyze viewing patterns by hour, day of week, or month",
            examples: ["When do I usually watch?", "What time of day do I watch most?"]
        },
        {
            name: "query_tag_correlations",
            description: "Find tags that commonly appear together with a given tag",
            examples: ["What tags often appear with 'blonde'?", "Tags commonly paired with 'POV'"]
        },
        {
            name: "query_top_performer_common_tags",
            description: "Find tags that are common across your top performers",
            examples: ["What tags do my top performers have in common?", "Shared themes among favorites"]
        },
        {
            name: "query_performer_pairs",
            description: "Find performers who frequently appear together in scenes",
            examples: ["Which performers appear together most?", "Who does cutegeekie work with?"]
        },
        {
            name: "query_interactive_content",
            description: "Find scenes with funscript/haptic support",
            examples: ["Do I have any interactive content?", "Show me scenes with funscripts"]
        },
        {
            name: "query_unwatched_content",
            description: "Find scenes you haven't watched yet, with optional filters",
            examples: ["What haven't I watched yet?", "Unwatched scenes from cutegeekie"]
        },
        {
            name: "rank_scenes_by_engagement",
            description: "Rank scenes by engagement score with multiple scoring modes",
            examples: ["Which scenes are my favorites?", "Rank my most replayed scenes"]
        },
        // Phase 1 tools
        {
            name: "query_performers_by_attribute",
            description: "Find performers by physical/demographic attributes",
            examples: ["Find blonde performers", "Show me performers from Japan", "Tall brunettes?"]
        },
        {
            name: "query_scenes_by_date",
            description: "Find scenes by release date, date added, or viewing date",
            examples: ["What did I watch last week?", "Scenes released in 2024", "Recently added content"]
        },
        {
            name: "query_favorites",
            description: "Get all favorited performers, studios, or tags",
            examples: ["Who are my favorite performers?", "Show my favorite studios"]
        },
        {
            name: "query_resume_points",
            description: "Find scenes you started but didn't finish (continue watching)",
            examples: ["What did I leave unfinished?", "Show my continue watching list"]
        },
        {
            name: "query_scenes_by_rating",
            description: "Find scenes by star rating",
            examples: ["Show my 5-star scenes", "What's my best-rated content?"]
        },
        {
            name: "query_all_tags",
            description: "List all tags with optional search and filtering",
            examples: ["List all tags", "What tags contain 'anal'?", "Show favorite tags"]
        },
        {
            name: "query_all_performers",
            description: "List all performers with optional search and filtering",
            examples: ["List all performers", "Performers with 'anna' in name", "Show favorite performers"]
        },
        // Phase 2 tools
        {
            name: "query_performer_profile",
            description: "Get detailed profile and statistics for a performer",
            examples: ["Tell me about Mia Malkova", "What are cutegeekie's stats?"]
        },
        {
            name: "query_studio_profile",
            description: "Get detailed statistics for a studio",
            examples: ["Tell me about Brazzers", "What are the stats for Reality Kings?"]
        },
        {
            name: "query_group_progress",
            description: "Track completion progress for a group/series",
            examples: ["How much of Best Of have I watched?", "What's next in this series?"]
        },
        {
            name: "query_viewing_history",
            description: "Get detailed viewing history with timestamps",
            examples: ["What did I watch yesterday?", "Viewing history for last week"]
        },
        {
            name: "query_storage_stats",
            description: "Analyze storage usage by studio, performer, tag, or resolution",
            examples: ["Storage by studio", "Which performers use the most space?"]
        },
        // Phase 3 tools
        {
            name: "query_tag_hierarchy",
            description: "Explore tag parent/child relationships",
            examples: ["What sub-tags does 'oral' have?", "Show tag hierarchy for 'position'"]
        },
        {
            name: "query_studio_hierarchy",
            description: "Explore studio networks and sub-studios",
            examples: ["Show Mindgeek's sub-studios", "What studios are under Brazzers?"]
        },
        {
            name: "query_scene_markers",
            description: "Find and analyze scene markers (timestamps)",
            examples: ["What markers exist in this scene?", "Find scenes with 'blowjob' markers"]
        },
        {
            name: "query_tag_usage_over_time",
            description: "Analyze tag trends over time",
            examples: ["How has my taste changed?", "What tags am I watching more lately?"]
        },
        {
            name: "query_performer_comparison",
            description: "Compare 2-5 performers side-by-side",
            examples: ["Compare Mia Malkova and Riley Reid", "Who has more scenes, X or Y?"]
        },
        // Phase 4 tools
        {
            name: "query_duplicates",
            description: "Find potential duplicate files using fingerprints, file size, or duration",
            examples: ["Find duplicate files", "Are there any redundant scenes?", "Show duplicates by size"]
        },
        {
            name: "query_o_history",
            description: "Analyze O event patterns by time period",
            examples: ["When do I typically O?", "O events this month", "What scenes have I O'd to?"]
        },
        {
            name: "query_performer_career_timeline",
            description: "Analyze a performer's content over time",
            examples: ["Show cutegeekie's career timeline", "When was Mia Malkova most active?"]
        }
    ];

    // Helper: Log messages with plugin prefix
    function log(message, level = 'info') {
        const prefix = `[${PLUGIN_NAME}]`;
        if (level === 'error') {
            console.error(prefix, message);
        } else if (level === 'warn') {
            console.warn(prefix, message);
        } else {
            console.log(prefix, message);
        }
    }

    // Helper: Parse markdown to sanitized HTML using marked.js + DOMPurify
    function renderMarkdown(text) {
        if (!text) return '';

        // If libraries aren't loaded yet, use safe fallback
        if (!markdownLibs.loaded || !markdownLibs.marked || !markdownLibs.DOMPurify) {
            // Fallback: escape HTML and convert line breaks only
            return escapeHtml(text).replace(/\n/g, '<br>');
        }

        try {
            // Parse markdown to HTML
            const rawHtml = markdownLibs.marked.parse(text);

            // Sanitize HTML to prevent XSS
            const cleanHtml = markdownLibs.DOMPurify.sanitize(rawHtml, {
                ALLOWED_TAGS: [
                    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'p', 'br', 'hr',
                    'strong', 'b', 'em', 'i', 'u', 's', 'del',
                    'code', 'pre',
                    'ul', 'ol', 'li',
                    'a',
                    'blockquote',
                    'table', 'thead', 'tbody', 'tr', 'th', 'td'
                ],
                ALLOWED_ATTR: ['href', 'target', 'rel', 'class'],
                ALLOW_DATA_ATTR: false
            });

            return cleanHtml;
        } catch (err) {
            log(`Markdown render error: ${err.message}`, 'error');
            // Safe fallback on error
            return escapeHtml(text).replace(/\n/g, '<br>');
        }
    }

    // Helper: Wait for element to appear in DOM
    function waitForElement(selector, timeout = 10000) {
        return new Promise((resolve, reject) => {
            const element = document.querySelector(selector);
            if (element) {
                return resolve(element);
            }

            const observer = new MutationObserver(() => {
                const element = document.querySelector(selector);
                if (element) {
                    observer.disconnect();
                    resolve(element);
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });

            setTimeout(() => {
                observer.disconnect();
                reject(new Error(`Element ${selector} not found within ${timeout}ms`));
            }, timeout);
        });
    }

    // Helper: GraphQL query executor
    async function callGQL(query, variables = {}) {
        try {
            const response = await fetch('/graphql', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    variables: variables
                })
            });

            const result = await response.json();

            if (result.errors) {
                log(`GraphQL error: ${JSON.stringify(result.errors)}`, 'error');
                return null;
            }

            return result.data;
        } catch (error) {
            log(`GraphQL request failed: ${error.message}`, 'error');
            return null;
        }
    }

    // Get library statistics
    async function getLibraryStats() {
        const query = `
            query Stats {
                stats {
                    scene_count
                    scenes_duration
                    performer_count
                    studio_count
                    tag_count
                    total_o_count
                    total_play_count
                    total_play_duration
                }
            }
        `;
        return await callGQL(query);
    }

    // Get scene by ID
    async function getScene(sceneId) {
        const query = `
            query FindScene($id: ID!) {
                findScene(id: $id) {
                    id
                    title
                    date
                    rating100
                    play_count
                    o_counter
                    organized
                    interactive
                    files {
                        path
                        size
                        duration
                        height
                        width
                        fingerprints {
                            type
                            value
                        }
                    }
                    performers {
                        id
                        name
                    }
                    tags {
                        id
                        name
                    }
                    studio {
                        id
                        name
                    }
                }
            }
        `;

        const result = await callGQL(query, { id: sceneId });
        return result?.findScene || null;
    }

    // Run a plugin task
    async function runPluginTask(taskName, args = {}) {
        const query = `
            mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
            }
        `;

        const argsArray = Object.entries(args).map(([key, value]) => ({
            key: key,
            value: { str: String(value) }
        }));

        return await callGQL(query, {
            plugin_id: PLUGIN_ID,
            task_name: taskName,
            args: argsArray
        });
    }

    // Fetch the last generated summary
    async function fetchLastSummary() {
        try {
            // Add cache-busting query param to prevent browser caching
            const cacheBuster = `?t=${Date.now()}`;
            const response = await fetch(SUMMARY_FILE + cacheBuster, {
                cache: 'no-store'
            });
            if (response.ok) {
                const data = await response.json();
                state.lastSummary = data;
                return data;
            } else {
                log(`Fetch failed with status: ${response.status}`, 'warn');
            }
        } catch (error) {
            log(`Fetch error: ${error.message}`, 'warn');
        }
        return null;
    }

    // Fetch chat history
    async function fetchChatHistory() {
        try {
            const cacheBuster = `?t=${Date.now()}`;
            const response = await fetch(CHAT_FILE + cacheBuster, {
                cache: 'no-store'
            });
            if (response.ok) {
                const data = await response.json();
                state.chatHistory = data;
                state.conversationId = data.conversation_id;
                return data;
            }
        } catch (error) {
            // Chat history may not exist yet
        }
        return null;
    }

    // Format duration in hours
    function formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
    }

    // Format timestamp as MM:SS or HH:MM:SS
    function formatTimestamp(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Seek video player to specific timestamp
    function seekToTimestamp(timestamp) {
        // Find the video element on the page
        const video = document.querySelector('video');
        if (video) {
            log(`Seeking to timestamp: ${timestamp}s (${formatTimestamp(timestamp)})`);
            video.currentTime = timestamp;
            // Optionally start playback if paused
            if (video.paused) {
                video.play().catch(e => log(`Could not auto-play: ${e.message}`, 'warn'));
            }
        } else {
            log('No video element found on page', 'warn');
        }
    }

    // Format tool name for display
    function formatToolName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/query /i, '')
            .split(' ')
            .map(w => w.charAt(0).toUpperCase() + w.slice(1))
            .join(' ');
    }

    // Check if dropdown is open (from localStorage)
    function isDropdownOpen() {
        return localStorage.getItem('stash-copilot-dropdown-open') === 'true';
    }

    // Save dropdown state
    function setDropdownOpen(open) {
        localStorage.setItem('stash-copilot-dropdown-open', open ? 'true' : 'false');
    }

    // Get active tab from localStorage
    function getActiveTab() {
        return localStorage.getItem('stash-copilot-active-tab') || 'summary';
    }

    // Save active tab
    function setActiveTab(tab) {
        localStorage.setItem('stash-copilot-active-tab', tab);
        state.activeTab = tab;
    }

    // Switch tabs
    function switchTab(dropdown, tabName) {
        setActiveTab(tabName);

        // Update tab buttons
        dropdown.querySelectorAll('.stash-copilot-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update tab content
        dropdown.querySelectorAll('.stash-copilot-tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.tab === tabName);
        });

        // Load chat history when switching to chat tab
        if (tabName === 'chat') {
            loadChatHistory(dropdown);
        }

        // Render tools list when switching to tools tab
        if (tabName === 'tools') {
            renderToolsList(dropdown);
        }

        // Load recommendations when switching to recommendations tab
        if (tabName === 'recommendations') {
            loadRecommendationsResults(dropdown);
        }

        // Load peak stats when switching to peak tab
        if (tabName === 'peak') {
            loadPeakStats(dropdown);
        }
    }

    // Render the tools list with examples
    function renderToolsList(dropdown) {
        const container = dropdown.querySelector('.stash-copilot-tools-list');
        if (!container) return;

        container.innerHTML = AVAILABLE_TOOLS.map(tool => `
            <div class="stash-copilot-tool-card">
                <div class="stash-copilot-tool-name">${tool.name}</div>
                <div class="stash-copilot-tool-description">${tool.description}</div>
                <div class="stash-copilot-tool-examples">
                    <span class="stash-copilot-tool-examples-label">Try:</span>
                    ${tool.examples.map(ex => `<span class="stash-copilot-tool-example" data-prompt="${ex.replace(/"/g, '&quot;')}">"${ex}"</span>`).join('')}
                </div>
            </div>
        `).join('');

        // Add click handlers to copy example prompts to chat input
        container.querySelectorAll('.stash-copilot-tool-example').forEach(example => {
            example.addEventListener('click', (e) => {
                e.stopPropagation();
                const prompt = example.dataset.prompt;
                const chatInput = dropdown.querySelector('.stash-copilot-chat-input');
                if (chatInput) {
                    chatInput.value = prompt;
                    switchTab(dropdown, 'chat');
                    chatInput.focus();
                }
            });
        });
    }

    // ========== Recommendations Functions ==========

    // Generate recommendations
    async function generateRecommendations(dropdown) {
        if (state.isGeneratingRecommendations) return;

        const generateBtn = dropdown.querySelector('.stash-copilot-rec-generate-btn');
        const contentContainer = dropdown.querySelector('.stash-copilot-recommendations-content');

        // Clear old cached data
        state.recommendationsResults = null;
        state.recommendationsProfile = null;
        state.recommendationsGeneratedAt = null;
        state.recommendationsDiscoverResults = [];
        state.recommendationsRewatchResults = [];
        state.recommendationsMergedResults = [];
        localStorage.removeItem('stash-copilot-recommendations-discover_new');
        localStorage.removeItem('stash-copilot-recommendations-rewatch');

        // Generate unique request IDs for both tasks
        const timestamp = Date.now();
        const discoverRequestId = `rec_discover_${timestamp}`;
        const rewatchRequestId = `rec_rewatch_${timestamp}`;
        state.recommendationsDiscoverRequestId = discoverRequestId;
        state.recommendationsRewatchRequestId = rewatchRequestId;
        state.isGeneratingRecommendations = true;
        state.recommendationsPage = 1; // Reset to first page for new results

        // Update UI
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="stash-copilot-spinner"></span>';
        contentContainer.innerHTML = `
            <div class="stash-copilot-rec-loading">
                <span class="stash-copilot-spinner"></span>
                <p>Generating recommendations...</p>
            </div>
        `;

        try {
            const isSessionMode = state.recommendationsTimeDecayDays === 'session';
            const sessionScenes = isSessionMode ? getSessionScenes() : [];

            // Check if session mode has scenes
            if (isSessionMode && sessionScenes.length === 0) {
                state.isGeneratingRecommendations = false;
                generateBtn.disabled = false;
                generateBtn.textContent = 'Generate';
                contentContainer.innerHTML = `
                    <p class="stash-copilot-rec-empty">No scenes viewed this session yet. Browse some scenes first, then come back for recommendations!</p>
                `;
                return;
            }

            // Build common task parameters
            const commonParams = {
                limit: '120'  // 10 pages x 12 per page
            };

            if (isSessionMode) {
                commonParams.scoring_method = 'base_weighted';
                commonParams.session_scene_ids = sessionScenes.join(',');
            } else {
                const useTimeDecay = state.recommendationsTimeDecayDays > 0;
                commonParams.scoring_method = useTimeDecay ? 'time_decayed' : 'base_weighted';
                commonParams.half_life_days = String(state.recommendationsTimeDecayDays || 30);
            }

            // Build separate params for each task
            const discoverParams = {
                ...commonParams,
                request_id: discoverRequestId
            };
            const rewatchParams = {
                ...commonParams,
                request_id: rewatchRequestId
            };

            const decayLabel = isSessionMode
                ? `this session (${sessionScenes.length} scenes)`
                : (state.recommendationsTimeDecayDays > 0 ? `${state.recommendationsTimeDecayDays} days` : 'all time');
            log(`Firing both recommendation tasks concurrently (recency: ${decayLabel})`);

            // Fire BOTH tasks concurrently
            await Promise.all([
                runPluginTask('Get Recommendations (Discover)', discoverParams),
                runPluginTask('Get Recommendations (Re-watch)', rewatchParams)
            ]);

            // Start unified polling for both result files
            pollUnifiedRecommendationsResults(dropdown, discoverRequestId, rewatchRequestId);

        } catch (error) {
            log(`Error starting recommendations: ${error.message}`, 'error');
            state.isGeneratingRecommendations = false;
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
            contentContainer.innerHTML = `
                <p class="stash-copilot-rec-error">Failed to generate recommendations. Please try again.</p>
            `;
        }
    }

    /**
     * Poll for both discover and rewatch result files in the modal, merge when both complete.
     */
    function pollUnifiedRecommendationsResults(dropdown, discoverRequestId, rewatchRequestId) {
        if (state.recommendationsPollInterval) {
            clearInterval(state.recommendationsPollInterval);
        }

        const discoverFile = `/plugin/stash-copilot/assets/recommendations_${discoverRequestId}.json`;
        const rewatchFile = `/plugin/stash-copilot/assets/recommendations_${rewatchRequestId}.json`;

        let discoverDone = false;
        let rewatchDone = false;
        let discoverError = null;
        let rewatchError = null;

        state.recommendationsPollInterval = setInterval(async () => {
            // Bail if request IDs have changed (new search started)
            if (state.recommendationsDiscoverRequestId !== discoverRequestId ||
                state.recommendationsRewatchRequestId !== rewatchRequestId) {
                clearInterval(state.recommendationsPollInterval);
                state.recommendationsPollInterval = null;
                return;
            }

            try {
                // Poll discover results
                if (!discoverDone) {
                    try {
                        const resp = await fetch(discoverFile + `?t=${Date.now()}`, { cache: 'no-store' });
                        if (resp.ok) {
                            const data = await resp.json();
                            if (data.status === 'complete') {
                                discoverDone = true;
                                state.recommendationsDiscoverResults = (data.results || []).map(r => ({
                                    ...r,
                                    _source: 'discover'
                                }));
                                state.recommendationsProfile = data.profile;
                                log(`Modal discover results received: ${state.recommendationsDiscoverResults.length} results`);
                            } else if (data.status === 'error') {
                                discoverDone = true;
                                discoverError = data.error || 'Discover failed';
                                log(`Modal discover task error: ${discoverError}`, 'error');
                            }
                        }
                    } catch (e) {
                        // File not ready yet
                    }
                }

                // Poll rewatch results
                if (!rewatchDone) {
                    try {
                        const resp = await fetch(rewatchFile + `?t=${Date.now()}`, { cache: 'no-store' });
                        if (resp.ok) {
                            const data = await resp.json();
                            if (data.status === 'complete') {
                                rewatchDone = true;
                                state.recommendationsRewatchResults = (data.results || []).map(r => ({
                                    ...r,
                                    _source: 'rewatch'
                                }));
                                if (!state.recommendationsProfile) {
                                    state.recommendationsProfile = data.profile;
                                }
                                log(`Modal rewatch results received: ${state.recommendationsRewatchResults.length} results`);
                            } else if (data.status === 'error') {
                                rewatchDone = true;
                                rewatchError = data.error || 'Re-watch failed';
                                log(`Modal rewatch task error: ${rewatchError}`, 'error');
                            }
                        }
                    } catch (e) {
                        // File not ready yet
                    }
                }

                // When both are done, merge and render
                if (discoverDone && rewatchDone) {
                    clearInterval(state.recommendationsPollInterval);
                    state.recommendationsPollInterval = null;
                    state.isGeneratingRecommendations = false;

                    // Update generate button
                    const generateBtn = dropdown.querySelector('.stash-copilot-rec-generate-btn');
                    if (generateBtn) {
                        generateBtn.disabled = false;
                        generateBtn.textContent = 'Generate';
                    }

                    // If both errored, show error
                    if (discoverError && rewatchError) {
                        const contentContainer = dropdown.querySelector('.stash-copilot-recommendations-content');
                        contentContainer.innerHTML = `
                            <p class="stash-copilot-rec-error">Failed to generate recommendations. Discover: ${discoverError}; Re-watch: ${rewatchError}</p>
                        `;
                        return;
                    }

                    mergeAndRenderModalRecsResults(dropdown);
                }
            } catch (e) {
                log(`Unified recs poll error: ${e.message}`);
            }
        }, 200);

        // Timeout after 120s
        setTimeout(() => {
            // Don't interfere if a newer search has started
            if (state.recommendationsDiscoverRequestId !== discoverRequestId ||
                state.recommendationsRewatchRequestId !== rewatchRequestId) {
                return;
            }
            if (state.recommendationsPollInterval) {
                clearInterval(state.recommendationsPollInterval);
                state.recommendationsPollInterval = null;
                state.isGeneratingRecommendations = false;

                const generateBtn = dropdown.querySelector('.stash-copilot-rec-generate-btn');
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate';
                }

                // If timed out but have partial results, render what we have
                if (state.recommendationsDiscoverResults.length > 0 || state.recommendationsRewatchResults.length > 0) {
                    log('Modal recs poll timed out but partial results available, rendering');
                    mergeAndRenderModalRecsResults(dropdown);
                } else {
                    const contentContainer = dropdown.querySelector('.stash-copilot-recommendations-content');
                    contentContainer.innerHTML = `
                        <p class="stash-copilot-rec-error">Request timed out. Please try again.</p>
                    `;
                }
            }
        }, 120000);
    }

    /**
     * Merge discover + rewatch results for the modal, deduplicate, sort, and render.
     */
    function mergeAndRenderModalRecsResults(dropdown) {
        const discoverResults = state.recommendationsDiscoverResults || [];
        const rewatchResults = state.recommendationsRewatchResults || [];

        // Merge: discover results first (take priority), then rewatch, dedup by scene_id
        const seenIds = new Set();
        const merged = [];

        for (const result of discoverResults) {
            const sceneId = result.scene?.id || result.scene_id;
            if (sceneId && !seenIds.has(sceneId)) {
                seenIds.add(sceneId);
                merged.push(result);
            }
        }

        for (const result of rewatchResults) {
            const sceneId = result.scene?.id || result.scene_id;
            if (sceneId && !seenIds.has(sceneId)) {
                seenIds.add(sceneId);
                merged.push(result);
            }
        }

        // Sort by combined/similarity score descending
        merged.sort((a, b) => {
            const scoreA = a.combined_score || a.similarity_score || 0;
            const scoreB = b.combined_score || b.similarity_score || 0;
            return scoreB - scoreA;
        });

        state.recommendationsMergedResults = merged;
        state.recommendationsGeneratedAt = Date.now();

        // Apply current view filter
        applyModalRecsViewFilter();

        // Save both modes to localStorage for persistence
        if (discoverResults.length > 0) {
            saveRecommendationsToStorage({ results: discoverResults, profile: state.recommendationsProfile }, 'discover_new');
            state.recommendationsCache['discover_new'] = {
                results: discoverResults,
                profile: state.recommendationsProfile,
                generatedAt: state.recommendationsGeneratedAt
            };
        }
        if (rewatchResults.length > 0) {
            saveRecommendationsToStorage({ results: rewatchResults, profile: state.recommendationsProfile }, 'rewatch');
            state.recommendationsCache['rewatch'] = {
                results: rewatchResults,
                profile: state.recommendationsProfile,
                generatedAt: state.recommendationsGeneratedAt
            };
        }

        // Render results
        renderRecommendationsResults(dropdown, {
            results: state.recommendationsResults,
            profile: state.recommendationsProfile,
            generatedAt: state.recommendationsGeneratedAt
        });
    }

    /**
     * Apply the current view filter (all/new/rewatch) to merged modal results.
     * Stores filtered results in state.recommendationsResults.
     */
    function applyModalRecsViewFilter() {
        const filter = state.recommendationsViewFilter || 'all';
        const merged = state.recommendationsMergedResults || [];

        let filtered;
        switch (filter) {
            case 'new':
                filtered = merged.filter(r => r._source === 'discover');
                break;
            case 'rewatch':
                filtered = merged.filter(r => r._source === 'rewatch');
                break;
            case 'all':
            default:
                filtered = [...merged];
                break;
        }

        state.recommendationsResults = filtered;
    }

    // ============================================
    // Peak Moments Functions
    // ============================================

    /**
     * Load O-moment embedding statistics for Peak tab
     */
    async function loadPeakStats(dropdown) {
        // Try new selector first, fall back to old one for backwards compatibility
        const statsContainer = dropdown.querySelector('.stash-copilot-peak-stats-display') ||
                               dropdown.querySelector('.stash-copilot-peak-stats');
        if (!statsContainer) return;

        // Also get the tip container to update based on embed status
        const tipContainer = dropdown.querySelector('.stash-copilot-peak-tip');
        const modal = dropdown.closest('.stash-copilot-insights-modal');

        // Check if we have results already - if so, don't show empty state
        if (state.peakResults && state.peakResults.length > 0) {
            if (modal) modal.removeAttribute('data-empty');
            return; // Keep existing results displayed
        }

        // Set empty state for dynamic sizing
        if (modal) modal.setAttribute('data-empty', 'true');

        try {
            // Fetch stats from backend
            const response = await fetch(`/plugin/stash-copilot/assets/o_moment_stats.json?t=${Date.now()}`);

            if (!response.ok) {
                // No stats file yet - show zeros
                statsContainer.innerHTML = `
                    <span><strong>0</strong> markers</span>
                    <span><strong>0</strong> embedded</span>
                    <span><strong>0</strong> scenes</span>
                `;
                // Update tip to show warning
                if (tipContainer) {
                    tipContainer.innerHTML = `
                        <p class="warning">⚠️ No O markers found. Add markers to scenes and run <strong>Embed Markers</strong> to enable this feature.</p>
                    `;
                }
                return;
            }

            const stats = await response.json();
            const totalMarkers = stats.total_markers || 0;
            const embeddedCount = stats.embedded_count || 0;
            const sceneCount = stats.scene_count || 0;

            statsContainer.innerHTML = `
                <span><strong>${totalMarkers}</strong> markers</span>
                <span><strong>${embeddedCount}</strong> embedded</span>
                <span><strong>${sceneCount}</strong> scenes</span>
            `;

            // Update tip based on embed status
            if (tipContainer) {
                if (embeddedCount === 0 && totalMarkers > 0) {
                    tipContainer.innerHTML = `
                        <p class="warning">⚠️ Markers found but not yet embedded. Click <strong>Embed Markers</strong> first.</p>
                    `;
                } else if (embeddedCount > 0) {
                    tipContainer.innerHTML = `
                        <p>Click <strong>Generate</strong> to discover scenes similar to your peak moments.</p>
                    `;
                } else {
                    tipContainer.innerHTML = `
                        <p class="warning">⚠️ No O markers found. Add markers to scenes first.</p>
                    `;
                }
            }

        } catch (e) {
            log(`Error loading peak stats: ${e.message}`, 'warn');
            statsContainer.innerHTML = `
                <span>Could not load stats</span>
            `;
        }
    }

    /**
     * Generate Peak Moments recommendations
     */
    async function generatePeakMoments(dropdown) {
        const generateBtn = dropdown.querySelector('.stash-copilot-peak-generate-btn');
        const contentContainer = dropdown.querySelector('.stash-copilot-peak-content');

        if (!generateBtn || !contentContainer) return;

        // Generate unique request ID
        const requestId = `peak_${Date.now()}`;
        state.peakRequestId = requestId;
        state.isGeneratingPeak = true;

        // Update UI
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="stash-copilot-spinner"></span>';
        contentContainer.innerHTML = `
            <div class="stash-copilot-peak-loading">
                <span class="stash-copilot-spinner"></span>
                <p>Finding scenes similar to your peak moments...</p>
            </div>
        `;

        try {
            // Build task parameters
            const taskParams = {
                request_id: requestId,
                limit: '60',  // 5 pages x 12 per page
                scoring_method: 'base_weighted'
            };

            await runPluginTask('Get Recommendations (Peak Moments)', taskParams);
            log(`Peak Moments task started (request_id: ${requestId})`);

            // Start polling for results
            pollPeakResults(dropdown, requestId);

        } catch (error) {
            log(`Error starting Peak Moments: ${error.message}`, 'error');
            state.isGeneratingPeak = false;
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
            contentContainer.innerHTML = `
                <p class="stash-copilot-peak-error">Failed to generate recommendations. Please try again.</p>
            `;
        }
    }

    /**
     * Poll for Peak Moments results
     */
    function pollPeakResults(dropdown, requestId) {
        if (state.peakPollInterval) {
            clearInterval(state.peakPollInterval);
        }

        let attempts = 0;
        const maxAttempts = 300; // 300 * 200ms = 60 seconds max

        state.peakPollInterval = setInterval(async () => {
            attempts++;

            try {
                const response = await fetch(`/plugin/stash-copilot/assets/recommendations_${requestId}.json?t=${Date.now()}`);

                if (response.ok) {
                    const data = await response.json();

                    if (data.status === 'complete' || data.status === 'error') {
                        clearInterval(state.peakPollInterval);
                        state.peakPollInterval = null;
                        state.isGeneratingPeak = false;

                        // Update generate button
                        const generateBtn = dropdown.querySelector('.stash-copilot-peak-generate-btn');
                        if (generateBtn) {
                            generateBtn.disabled = false;
                            generateBtn.textContent = 'Generate';
                        }

                        if (data.status === 'error') {
                            const contentContainer = dropdown.querySelector('.stash-copilot-peak-content');
                            contentContainer.innerHTML = `
                                <p class="stash-copilot-peak-error">${data.error || 'An error occurred'}</p>
                            `;
                            return;
                        }

                        // Render results
                        renderPeakResults(dropdown, data);
                    }
                }
            } catch (e) {
                // Network error - continue polling
                log(`Peak poll error: ${e.message}`, 'debug');
            }

            // Timeout check
            if (attempts >= maxAttempts) {
                clearInterval(state.peakPollInterval);
                state.peakPollInterval = null;
                state.isGeneratingPeak = false;

                const generateBtn = dropdown.querySelector('.stash-copilot-peak-generate-btn');
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate';
                }

                const contentContainer = dropdown.querySelector('.stash-copilot-peak-content');
                contentContainer.innerHTML = `
                    <p class="stash-copilot-peak-error">Request timed out. Please try again.</p>
                `;
            }
        }, 200);
    }

    /**
     * Render Peak Moments results
     */
    function renderPeakResults(dropdown, data) {
        const contentContainer = dropdown.querySelector('.stash-copilot-peak-content');
        if (!contentContainer) return;

        const modal = dropdown.closest('.stash-copilot-insights-modal');
        const results = data.results || [];

        // Remove empty state when rendering results
        if (modal && results.length > 0) {
            modal.removeAttribute('data-empty');
        }

        if (results.length === 0) {
            contentContainer.innerHTML = `
                <div class="stash-copilot-peak-empty-results">
                    <div class="stash-copilot-peak-icon">🔥</div>
                    <h4>No Peak Moments Found</h4>
                    <p>Make sure you have:</p>
                    <ul>
                        <li>O markers on scenes you've watched</li>
                        <li>Run the "Embed O-Moments" task</li>
                    </ul>
                    <p>Click "🎬 Embed Markers" to generate embeddings from your markers.</p>
                </div>
            `;
            return;
        }

        // Paginate results
        const perPage = 12;
        const currentPage = state.peakPage || 1;
        const totalPages = Math.ceil(results.length / perPage);
        const start = (currentPage - 1) * perPage;
        const end = start + perPage;
        const pageResults = results.slice(start, end);

        // Render cards using unified card system
        const cardsHtml = pageResults.map((result, idx) => {
            const scene = result.scene;
            const score = result.combined_score || result.similarity_score || 0;
            return buildSceneCard({
                scene: scene,
                score: score,
                cardIndex: idx,
                theme: 'peak',
                scoreLabel: 'match'
            });
        }).join('');

        // Add pagination if needed
        const paginationHtml = totalPages > 1 ? `
            <div class="stash-copilot-peak-pagination">
                <button class="stash-copilot-peak-page-btn prev" ${currentPage <= 1 ? 'disabled' : ''}>&lt;</button>
                <span class="stash-copilot-peak-page-info">${currentPage} / ${totalPages}</span>
                <button class="stash-copilot-peak-page-btn next" ${currentPage >= totalPages ? 'disabled' : ''}>&gt;</button>
            </div>
        ` : '';

        contentContainer.innerHTML = `
            <div class="stash-copilot-peak-results">
                ${cardsHtml}
            </div>
            ${paginationHtml}
        `;

        // Setup event handlers using unified system
        const resultsContainer = contentContainer.querySelector('.stash-copilot-peak-results');
        if (resultsContainer) {
            setupSceneCardEvents(resultsContainer, { theme: 'peak', tooltipMode: 'cursor' });
        }

        // Store results for pagination
        state.peakResults = results;
        state.peakTotalPages = totalPages;

        // Setup pagination handlers
        const prevBtn = contentContainer.querySelector('.stash-copilot-peak-page-btn.prev');
        const nextBtn = contentContainer.querySelector('.stash-copilot-peak-page-btn.next');

        if (prevBtn) {
            prevBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (state.peakPage > 1) {
                    state.peakPage--;
                    renderPeakResults(dropdown, { results: state.peakResults });
                }
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (state.peakPage < state.peakTotalPages) {
                    state.peakPage++;
                    renderPeakResults(dropdown, { results: state.peakResults });
                }
            });
        }
    }

    /**
     * Trigger the Embed O-Moments task
     */
    async function triggerEmbedOMoments(dropdown) {
        const embedBtn = dropdown.querySelector('.stash-copilot-peak-embed-btn');
        if (!embedBtn) return;

        embedBtn.disabled = true;
        const originalText = embedBtn.innerHTML;
        embedBtn.innerHTML = '<span class="stash-copilot-spinner"></span> Starting...';

        try {
            await runPluginTask('Embed O-Moments', {});
            embedBtn.innerHTML = '✓ Task Started';

            // Show hint to check Stash tasks
            setTimeout(() => {
                embedBtn.innerHTML = 'Check Stash Tasks';
                embedBtn.disabled = false;
                // After a delay, reset and reload stats
                setTimeout(() => {
                    embedBtn.innerHTML = originalText;
                    loadPeakStats(dropdown);
                }, 3000);
            }, 2000);

        } catch (e) {
            log(`Failed to start Embed O-Moments: ${e.message}`, 'error');
            embedBtn.disabled = false;
            embedBtn.innerHTML = originalText;
            embedBtn.classList.add('error');

            // Reset error state
            setTimeout(() => {
                embedBtn.classList.remove('error');
            }, 3000);
        }
    }

    // ============================================
    // End Peak Moments Functions
    // ============================================

    // Format relative time ("5 minutes ago", "1 day ago", etc.)
    function formatRelativeTime(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;

        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        const weeks = Math.floor(days / 7);

        if (seconds < 60) return 'just now';
        if (minutes === 1) return '1 minute ago';
        if (minutes < 60) return `${minutes} minutes ago`;
        if (hours === 1) return '1 hour ago';
        if (hours < 24) return `${hours} hours ago`;
        if (days === 1) return '1 day ago';
        if (days < 7) return `${days} days ago`;
        if (weeks === 1) return '1 week ago';
        return `${weeks} weeks ago`;
    }

    // Save recommendations to localStorage
    function saveRecommendationsToStorage(data, mode) {
        try {
            const storageData = {
                results: data.results,
                profile: data.profile,
                generatedAt: Date.now()
            };
            // Save to mode-specific key
            const storageKey = `stash-copilot-recommendations-${mode || state.recommendationsMode}`;
            localStorage.setItem(storageKey, JSON.stringify(storageData));
        } catch (e) {
            log('Failed to save recommendations to localStorage', 'warn');
        }
    }

    // Load recommendations from localStorage for a specific mode
    function loadRecommendationsFromStorage(mode) {
        try {
            const storageKey = `stash-copilot-recommendations-${mode || state.recommendationsMode}`;
            const stored = localStorage.getItem(storageKey);
            if (stored) {
                return JSON.parse(stored);
            }
        } catch (e) {
            log('Failed to load recommendations from localStorage', 'warn');
        }
        return null;
    }

    // Load existing recommendations results
    async function loadRecommendationsResults(dropdown) {
        const modal = dropdown.closest('.stash-copilot-insights-modal');

        // Load both modes from localStorage into cache (if not already cached)
        ['discover_new', 'rewatch'].forEach(mode => {
            if (!state.recommendationsCache[mode]) {
                const stored = loadRecommendationsFromStorage(mode);
                if (stored && stored.results && stored.results.length > 0) {
                    state.recommendationsCache[mode] = {
                        results: stored.results,
                        profile: stored.profile,
                        generatedAt: stored.generatedAt
                    };
                }
            }
        });

        // Build unified view from both cached modes
        const discoverCache = state.recommendationsCache['discover_new'];
        const rewatchCache = state.recommendationsCache['rewatch'];

        if ((discoverCache && discoverCache.results?.length > 0) ||
            (rewatchCache && rewatchCache.results?.length > 0)) {

            // Populate discover/rewatch results with source tags
            state.recommendationsDiscoverResults = (discoverCache?.results || []).map(r => ({
                ...r,
                _source: r._source || 'discover'
            }));
            state.recommendationsRewatchResults = (rewatchCache?.results || []).map(r => ({
                ...r,
                _source: r._source || 'rewatch'
            }));
            state.recommendationsProfile = discoverCache?.profile || rewatchCache?.profile;
            state.recommendationsGeneratedAt = Math.max(
                discoverCache?.generatedAt || 0,
                rewatchCache?.generatedAt || 0
            );

            // Merge and deduplicate
            const seenIds = new Set();
            const merged = [];
            for (const r of state.recommendationsDiscoverResults) {
                const sid = r.scene?.id || r.scene_id;
                if (sid && !seenIds.has(sid)) { seenIds.add(sid); merged.push(r); }
            }
            for (const r of state.recommendationsRewatchResults) {
                const sid = r.scene?.id || r.scene_id;
                if (sid && !seenIds.has(sid)) { seenIds.add(sid); merged.push(r); }
            }
            merged.sort((a, b) => {
                const sa = a.combined_score || a.similarity_score || 0;
                const sb = b.combined_score || b.similarity_score || 0;
                return sb - sa;
            });

            state.recommendationsMergedResults = merged;
            applyModalRecsViewFilter();

            renderRecommendationsResults(dropdown, {
                results: state.recommendationsResults,
                profile: state.recommendationsProfile,
                generatedAt: state.recommendationsGeneratedAt
            });
            // Remove empty state attribute
            if (modal) modal.removeAttribute('data-empty');
            return;
        }

        // Show empty state - set attribute for dynamic sizing
        if (modal) modal.setAttribute('data-empty', 'true');

        // Load stats into the empty state
        loadRecsEmptyStateStats(dropdown);
    }

    // Load stats for Recs empty state
    async function loadRecsEmptyStateStats(dropdown) {
        const statsContainer = dropdown.querySelector('.stash-copilot-rec-stats-placeholder');
        if (!statsContainer) return;

        try {
            const statsData = await getLibraryStats();
            if (statsData && statsData.stats) {
                const stats = statsData.stats;
                const watchTime = formatDuration(stats.total_play_duration);
                const engagedScenes = stats.watched_scene_count || Math.round(stats.total_play_count * 0.7);

                statsContainer.innerHTML = `
                    <span>Based on <strong>${watchTime}</strong> watched</span>
                    <span><strong>${engagedScenes.toLocaleString()}</strong> engaged scenes</span>
                `;
            } else {
                statsContainer.innerHTML = `
                    <span>Ready to analyze your viewing patterns</span>
                `;
            }
        } catch (e) {
            log(`Error loading recs stats: ${e.message}`, 'warn');
            statsContainer.innerHTML = `
                <span>Ready to generate recommendations</span>
            `;
        }
    }

    // Format duration for recommendation cards
    function formatRecDuration(seconds) {
        if (!seconds) return '';
        const s = parseFloat(seconds);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = Math.floor(s % 60);
        if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
        return `${m}:${sec.toString().padStart(2, '0')}`;
    }

    // Get resolution label from height
    function getRecResolution(height) {
        if (!height) return '';
        if (height >= 2160) return '4K';
        if (height >= 1440) return '1440p';
        if (height >= 1080) return '1080p';
        if (height >= 720) return '720p';
        if (height >= 480) return '480p';
        return `${height}p`;
    }

    // Build single recommendation card HTML
    function buildRecCardHtml(r, cardIndex) {
        const scene = r.scene || {};
        const sceneId = r.scene_id;

        // Use title, or filename from path, or fallback to Scene ID (same as Similar Scenes modal)
        let title = scene.title;
        if (!title && scene.files?.[0]?.path) {
            const path = scene.files[0].path;
            // Extract filename (handle both / and \ separators)
            title = path.split('/').pop().split('\\').pop();
        }
        title = title || `Scene ${sceneId}`;

        const performers = (scene.performers || []).map(p => p.name).filter(Boolean).join(', ');
        const tags = (scene.tags || []).map(t => t.name);
        const studio = scene.studio?.name || '';
        const date = scene.date || '';
        const rating = scene.rating100;
        const playCount = scene.play_count || 0;
        const oCount = scene.o_counter || 0;
        const isInteractive = scene.interactive || false;
        const engagementScore = calculateEngagementScore(playCount, oCount);

        // Get file info
        const file = (scene.files && scene.files[0]) || {};
        const duration = file.duration;
        const height = file.height;
        const fileSize = file.size;

        // Get oshash for VTT/sprite URLs (Stash uses oshash-based paths)
        const fingerprints = file.fingerprints || [];
        const oshash = fingerprints.find(f => f.type === 'oshash')?.value || '';

        // Calculate score
        const scorePct = (r.combined_score * 100).toFixed(0);
        const resolution = getRecResolution(height);

        // Build URLs (match Similar Scenes modal format)
        const screenshotUrl = `/scene/${sceneId}/screenshot`;
        const previewUrl = `/scene/${sceneId}/preview`;
        // Use oshash-based URLs for sprites/VTT (same as Similar Scenes modal)
        const spriteUrl = oshash ? `/scene/${oshash}_sprite.jpg` : `/scene/${sceneId}/vtt/sprite`;
        const vttUrl = oshash ? `/scene/${oshash}_thumbs.vtt` : '';

        return `
            <a href="/scenes/${sceneId}" class="stash-copilot-rec-card" data-scene-id="${sceneId}"
               data-title="${escapeHtml(title)}"
               data-studio="${escapeHtml(studio)}"
               data-performers="${escapeHtml(performers)}"
               data-tags="${escapeHtml(tags.join('|||'))}"
               data-date="${date}"
               data-rating="${rating || ''}"
               data-filesize="${fileSize || ''}"
               data-duration="${duration || ''}"
               data-playcount="${playCount}"
               data-ocount="${oCount}"
               data-engagement="${engagementScore.toFixed(1)}"
               data-score="${scorePct}"
               data-sprite-url="${spriteUrl}"
               data-vtt-url="${vttUrl}"
               style="--card-index: ${cardIndex}"
               ${playCount > 0 ? 'data-watched="true"' : ''}>
                <div class="stash-copilot-rec-card-thumb">
                    <img class="stash-copilot-rec-screenshot" src="${screenshotUrl}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.style.display='none'">
                    <video class="stash-copilot-rec-preview" src="" data-preview-src="${previewUrl}" muted loop playsinline></video>
                    <div class="stash-copilot-rec-sprite"></div>
                    <div class="stash-copilot-rec-card-score${parseInt(scorePct) >= 90 ? ' high-match' : ''}">${scorePct}%</div>
                    ${engagementScore > 0 ? `<div class="stash-copilot-card-engagement" title="Engagement: ${engagementScore.toFixed(1)}">🔥 ${Math.round(engagementScore)}</div>` : ''}
                    ${duration ? `<div class="stash-copilot-rec-duration">${formatRecDuration(duration)}</div>` : ''}
                    ${isInteractive ? '<div class="stash-copilot-rec-interactive" title="Interactive">🎮</div>' : ''}
                    ${resolution ? `<div class="stash-copilot-rec-resolution">${resolution}</div>` : ''}
                    <div class="stash-copilot-rec-scrubber" data-sprite-url="${spriteUrl}" data-vtt-url="${vttUrl}">
                        <div class="stash-copilot-rec-scrubber-progress"></div>
                        <div class="stash-copilot-rec-scrubber-time"></div>
                    </div>
                </div>
                <div class="stash-copilot-rec-card-info">
                    <div class="stash-copilot-rec-card-title">${escapeHtml(title)}</div>
                    ${performers ? `<div class="stash-copilot-rec-card-performers">${escapeHtml(performers.split(', ').slice(0, 2).join(', '))}</div>` : ''}
                    <div class="stash-copilot-rec-card-stats${playCount > 0 ? ' watched' : ''}">
                        <span class="${playCount > 0 ? 'has-plays' : ''}" title="Play count">▶ ${playCount}</span>
                        <span class="${oCount > 0 ? 'has-o' : ''}" title="O count">💦 ${oCount}</span>
                    </div>
                </div>
            </a>
        `;
    }

    // Setup recommendation card events (hover preview, tooltip, scrubber)
    function setupRecCardEvents(container) {
        const cards = container.querySelectorAll('.stash-copilot-rec-card');

        // Create floating tooltip element (shared across all cards)
        let tooltip = document.getElementById('stash-copilot-rec-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'stash-copilot-rec-tooltip';
            tooltip.className = 'stash-copilot-rec-tooltip';
            document.body.appendChild(tooltip);
        }

        // Helper functions
        const formatDuration = (seconds) => {
            if (!seconds) return '';
            const s = parseFloat(seconds);
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const sec = Math.floor(s % 60);
            if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
            return `${m}:${sec.toString().padStart(2, '0')}`;
        };

        const formatSize = (bytes) => {
            if (!bytes) return '';
            const b = parseFloat(bytes);
            const gb = b / (1024 * 1024 * 1024);
            if (gb >= 1) return `${gb.toFixed(2)} GB`;
            const mb = b / (1024 * 1024);
            return `${mb.toFixed(0)} MB`;
        };

        cards.forEach(card => {
            const screenshot = card.querySelector('.stash-copilot-rec-screenshot');
            const preview = card.querySelector('.stash-copilot-rec-preview');
            let previewLoaded = false;
            let hoverTimeout = null;
            let tooltipTimeout = null;

            card.addEventListener('mouseenter', (e) => {
                // Show tooltip after 250ms delay
                tooltipTimeout = setTimeout(() => {
                    const title = card.dataset.title || '';
                    const studio = card.dataset.studio || '';
                    const performers = card.dataset.performers || '';
                    const tagsStr = card.dataset.tags || '';
                    const date = card.dataset.date || '';
                    const rating = card.dataset.rating || '';
                    const filesize = card.dataset.filesize || '';
                    const duration = card.dataset.duration || '';
                    const playcount = card.dataset.playcount || '0';
                    const ocount = card.dataset.ocount || '0';
                    const score = card.dataset.score || '';

                    const tags = tagsStr ? tagsStr.split('|||').slice(0, 8) : [];
                    const tagsHtml = tags.map(t => `<span class="stash-copilot-rec-tooltip-tag">${escapeHtml(t)}</span>`).join('');

                    tooltip.innerHTML = `
                        <div class="stash-copilot-rec-tooltip-header">
                            <div class="stash-copilot-rec-tooltip-title">${escapeHtml(title)}</div>
                            ${studio ? `<div class="stash-copilot-rec-tooltip-studio">${escapeHtml(studio)}</div>` : ''}
                        </div>
                        ${performers ? `<div class="stash-copilot-rec-tooltip-performers">${escapeHtml(performers)}</div>` : ''}
                        <div class="stash-copilot-rec-tooltip-meta">
                            ${date ? `<span>📅 ${date}</span>` : ''}
                            ${duration ? `<span>⏱ ${formatDuration(duration)}</span>` : ''}
                            ${filesize ? `<span>💾 ${formatSize(filesize)}</span>` : ''}
                            ${rating ? `<span>⭐ ${(parseInt(rating) / 20).toFixed(1)}</span>` : ''}
                        </div>
                        <div class="stash-copilot-rec-tooltip-stats">
                            <span class="${parseInt(playcount) > 0 ? 'has-value' : ''}">▶ ${playcount} plays</span>
                            <span class="${parseInt(ocount) > 0 ? 'has-value' : ''}">💦 ${ocount}</span>
                            <span class="stash-copilot-rec-tooltip-score">${score}% match</span>
                        </div>
                        ${tagsHtml ? `<div class="stash-copilot-rec-tooltip-tags">${tagsHtml}</div>` : ''}
                    `;

                    // Position tooltip to the right of the card
                    const cardRect = card.getBoundingClientRect();
                    const tooltipWidth = 260;
                    const margin = 10;

                    let left = cardRect.right + margin;
                    let top = cardRect.top;

                    // If tooltip would go off-screen right, position to the left
                    if (left + tooltipWidth > window.innerWidth) {
                        left = cardRect.left - tooltipWidth - margin;
                    }

                    // Keep within vertical bounds
                    const tooltipHeight = 200;
                    if (top + tooltipHeight > window.innerHeight) {
                        top = window.innerHeight - tooltipHeight - margin;
                    }
                    if (top < margin) top = margin;

                    tooltip.style.left = `${left}px`;
                    tooltip.style.top = `${top}px`;
                    tooltip.classList.add('visible');
                }, 250);

                // Start preview after 400ms
                hoverTimeout = setTimeout(() => {
                    // Don't show video preview if scrubber is active
                    const scrubberEl = card.querySelector('.stash-copilot-rec-scrubber');
                    if (scrubberEl && scrubberEl.classList.contains('active')) {
                        return;
                    }

                    if (preview && !previewLoaded) {
                        const previewSrc = preview.dataset.previewSrc;
                        if (previewSrc) {
                            preview.src = previewSrc;
                            preview.onloadeddata = () => {
                                previewLoaded = true;
                                // Check again in case scrubber became active during load
                                if (scrubberEl && scrubberEl.classList.contains('active')) {
                                    return;
                                }
                                preview.style.display = 'block';
                                if (screenshot) screenshot.style.opacity = '0';
                                preview.play().catch(() => {});
                            };
                            preview.onerror = () => {
                                previewLoaded = false;
                            };
                        }
                    } else if (preview && previewLoaded) {
                        preview.style.display = 'block';
                        if (screenshot) screenshot.style.opacity = '0';
                        preview.play().catch(() => {});
                    }
                }, 400);
            });

            card.addEventListener('mouseleave', () => {
                clearTimeout(hoverTimeout);
                clearTimeout(tooltipTimeout);
                tooltip.classList.remove('visible');

                if (preview) {
                    preview.pause();
                    preview.style.display = 'none';
                }
                if (screenshot) {
                    screenshot.style.opacity = '1';
                }
            });

            // Setup scrubber (matching Similar Scenes modal implementation)
            const thumbnail = card.querySelector('.stash-copilot-rec-card-thumb');
            const scrubber = card.querySelector('.stash-copilot-rec-scrubber');
            const scrubberSprite = thumbnail?.querySelector('.stash-copilot-rec-sprite');
            const scrubberTime = card.querySelector('.stash-copilot-rec-scrubber-time');
            const scrubberProgress = card.querySelector('.stash-copilot-rec-scrubber-progress');
            let spriteLoaded = false;

            if (scrubber && scrubberSprite && thumbnail) {
                const duration = parseFloat(card.dataset.duration) || 0;
                const spriteUrl = scrubber.dataset.spriteUrl;
                const vttUrl = scrubber.dataset.vttUrl || '';
                const videoPreview = card.querySelector('.stash-copilot-rec-preview');
                const screenshotEl = card.querySelector('.stash-copilot-rec-screenshot');

                // Sprite data from VTT parsing
                let sprites = [];
                let spriteScale = 1;

                // Parse VTT time string to seconds
                function parseVTTTime(timeStr) {
                    const parts = timeStr.split(':');
                    const hours = parseInt(parts[0]);
                    const minutes = parseInt(parts[1]);
                    const seconds = parseFloat(parts[2]);
                    return hours * 3600 + minutes * 60 + seconds;
                }

                // Parse VTT file to get sprite coordinates (like StashApp)
                async function loadSpriteVTT() {
                    try {
                        if (!vttUrl) throw new Error('No VTT URL available');
                        const response = await fetch(vttUrl);
                        if (!response.ok) throw new Error('VTT fetch failed');
                        const vttText = await response.text();

                        // Parse VTT format - each cue has: timestamp --> timestamp \n url#xywh=x,y,w,h
                        const lines = vttText.split('\n');
                        let currentStart = 0;
                        let currentEnd = 0;

                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i].trim();

                            // Match timestamp line: 00:00:00.000 --> 00:00:05.000
                            const timeMatch = line.match(/^(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)/);
                            if (timeMatch) {
                                currentStart = parseVTTTime(timeMatch[1]);
                                currentEnd = parseVTTTime(timeMatch[2]);
                                continue;
                            }

                            // Match sprite line: url#xywh=x,y,w,h
                            const spriteMatch = line.match(/^([^#]+)#xywh=(\d+),(\d+),(\d+),(\d+)$/i);
                            if (spriteMatch) {
                                // Resolve relative URL against current origin
                                const baseUrl = window.location.origin;
                                const spritePathUrl = spriteMatch[1].startsWith('/')
                                    ? baseUrl + spriteMatch[1]
                                    : new URL(spriteMatch[1], baseUrl + vttUrl).href;
                                sprites.push({
                                    url: spritePathUrl,
                                    start: currentStart,
                                    end: currentEnd,
                                    x: parseInt(spriteMatch[2]),
                                    y: parseInt(spriteMatch[3]),
                                    w: parseInt(spriteMatch[4]),
                                    h: parseInt(spriteMatch[5])
                                });
                            }
                        }

                        if (sprites.length > 0) {
                            // Get container dimensions
                            const containerRect = thumbnail.getBoundingClientRect();
                            const containerW = containerRect.width;
                            const containerH = containerRect.height;

                            // Use first sprite's dimensions for scaling
                            const frameW = sprites[0].w;
                            const frameH = sprites[0].h;

                            // Calculate scale for "contain" - fit frame in container
                            const scaleX = containerW / frameW;
                            const scaleY = containerH / frameH;
                            spriteScale = Math.min(scaleX, scaleY);

                            // Load sprite image to get full sheet dimensions for background-size
                            const tempImg = new Image();
                            tempImg.onload = () => {
                                const scaledSheetW = tempImg.width * spriteScale;
                                const scaledSheetH = tempImg.height * spriteScale;

                                // Calculate scaled frame dimensions for centering
                                const scaledFrameW = frameW * spriteScale;
                                const scaledFrameH = frameH * spriteScale;
                                const offsetX = (containerW - scaledFrameW) / 2;
                                const offsetY = (containerH - scaledFrameH) / 2;

                                // Size sprite element to scaled frame dimensions (acts as clip mask)
                                scrubberSprite.style.width = `${scaledFrameW}px`;
                                scrubberSprite.style.height = `${scaledFrameH}px`;
                                scrubberSprite.style.left = `${offsetX}px`;
                                scrubberSprite.style.top = `${offsetY}px`;
                                scrubberSprite.style.backgroundImage = `url(${sprites[0].url})`;
                                scrubberSprite.style.backgroundSize = `${scaledSheetW}px ${scaledSheetH}px`;
                                scrubberSprite.style.transform = 'none';

                                // Store scale for position calculations in mousemove
                                scrubberSprite.dataset.spriteScale = spriteScale;

                                // Set initial position (first frame)
                                const scaledX = sprites[0].x * spriteScale;
                                const scaledY = sprites[0].y * spriteScale;
                                scrubberSprite.style.backgroundPosition = `-${scaledX}px -${scaledY}px`;

                                spriteLoaded = true;
                            };
                            tempImg.src = sprites[0].url;
                        }
                    } catch (err) {
                        // Fallback to grid-based approach
                        loadSpriteFallback();
                    }
                }

                // Fallback: grid-based approach if VTT unavailable
                function loadSpriteFallback() {
                    if (!spriteUrl) return;

                    const SPRITE_COLS = 9;
                    const SPRITE_ROWS = 9;

                    const spriteImage = new Image();
                    spriteImage.onload = () => {
                        const imgW = spriteImage.width;
                        const imgH = spriteImage.height;

                        // Calculate actual frame dimensions from sprite sheet (9x9 grid)
                        const actualFrameW = imgW / SPRITE_COLS;
                        const actualFrameH = imgH / SPRITE_ROWS;

                        const containerRect = thumbnail.getBoundingClientRect();
                        const containerW = containerRect.width;
                        const containerH = containerRect.height;

                        // Calculate scale for "contain" - fit frame in container
                        const scaleX = containerW / actualFrameW;
                        const scaleY = containerH / actualFrameH;
                        spriteScale = Math.min(scaleX, scaleY);

                        // Build sprites array from grid
                        for (let row = 0; row < SPRITE_ROWS; row++) {
                            for (let col = 0; col < SPRITE_COLS; col++) {
                                sprites.push({
                                    url: spriteUrl,
                                    x: col * actualFrameW,
                                    y: row * actualFrameH,
                                    w: actualFrameW,
                                    h: actualFrameH
                                });
                            }
                        }

                        // Apply CSS-based scaling
                        const scaledSheetW = imgW * spriteScale;
                        const scaledSheetH = imgH * spriteScale;
                        const scaledFrameW = actualFrameW * spriteScale;
                        const scaledFrameH = actualFrameH * spriteScale;
                        const offsetX = (containerW - scaledFrameW) / 2;
                        const offsetY = (containerH - scaledFrameH) / 2;

                        scrubberSprite.style.width = `${scaledFrameW}px`;
                        scrubberSprite.style.height = `${scaledFrameH}px`;
                        scrubberSprite.style.left = `${offsetX}px`;
                        scrubberSprite.style.top = `${offsetY}px`;
                        scrubberSprite.style.backgroundImage = `url(${spriteUrl})`;
                        scrubberSprite.style.backgroundSize = `${scaledSheetW}px ${scaledSheetH}px`;
                        scrubberSprite.style.backgroundPosition = '0 0';
                        scrubberSprite.style.transform = 'none';
                        scrubberSprite.dataset.spriteScale = spriteScale;

                        spriteLoaded = true;
                    };
                    spriteImage.src = spriteUrl;
                }

                scrubber.addEventListener('mouseenter', async () => {
                    scrubber.classList.add('active');
                    if (!spriteLoaded && sprites.length === 0) {
                        await loadSpriteVTT();
                    }
                });

                scrubber.addEventListener('mousemove', (e) => {
                    if (!spriteLoaded || sprites.length === 0) return;

                    const rect = scrubber.getBoundingClientRect();
                    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                    const time = pct * duration;

                    if (scrubberProgress) {
                        scrubberProgress.style.width = `${pct * 100}%`;
                    }

                    if (scrubberTime && duration > 0) {
                        scrubberTime.textContent = formatDuration(time);
                        scrubberTime.style.display = 'block';
                        const timeWidth = scrubberTime.offsetWidth;
                        let leftPos = pct * rect.width - timeWidth / 2;
                        leftPos = Math.max(0, Math.min(leftPos, rect.width - timeWidth));
                        scrubberTime.style.left = `${leftPos}px`;
                    }

                    // Find sprite for current time (VTT) or by percentage (grid fallback)
                    let sprite = null;
                    if (sprites[0]?.start !== undefined) {
                        // VTT-based: find by time
                        sprite = sprites.find(s => time >= s.start && time < s.end);
                    } else {
                        // Grid-based: find by percentage
                        const idx = Math.min(Math.floor(pct * sprites.length), sprites.length - 1);
                        sprite = sprites[idx];
                    }

                    if (sprite) {
                        const scale = parseFloat(scrubberSprite.dataset.spriteScale) || spriteScale;
                        const scaledX = sprite.x * scale;
                        const scaledY = sprite.y * scale;
                        scrubberSprite.style.backgroundPosition = `-${scaledX}px -${scaledY}px`;
                        scrubberSprite.style.opacity = '1';

                        if (videoPreview) videoPreview.style.display = 'none';
                        if (screenshotEl) screenshotEl.style.opacity = '0';
                    }
                });

                scrubber.addEventListener('mouseleave', () => {
                    scrubber.classList.remove('active');
                    if (scrubberProgress) scrubberProgress.style.width = '0';
                    if (scrubberTime) scrubberTime.style.display = 'none';
                    if (scrubberSprite) scrubberSprite.style.opacity = '0';
                    if (screenshotEl) screenshotEl.style.opacity = '1';
                });
            }
        });
    }

    // Render recommendations results
    function renderRecommendationsResults(dropdown, data, page = 1) {
        const contentContainer = dropdown.querySelector('.stash-copilot-recommendations-content');
        if (!contentContainer) return;

        const modal = dropdown.closest('.stash-copilot-insights-modal');
        const results = data.results || [];

        // Remove empty state when rendering results
        if (modal && results.length > 0) {
            modal.removeAttribute('data-empty');
        }

        if (results.length === 0) {
            contentContainer.innerHTML = `
                <p class="stash-copilot-rec-empty">No recommendations found. Make sure you have scene embeddings generated and viewing history.</p>
            `;
            return;
        }

        // Pagination logic
        const perPage = state.recommendationsPerPage;
        const totalPages = Math.ceil(results.length / perPage);
        const currentPage = Math.min(Math.max(1, page), totalPages);
        state.recommendationsPage = currentPage;

        const startIndex = (currentPage - 1) * perPage;
        const endIndex = startIndex + perPage;
        const pageResults = results.slice(startIndex, endIndex);

        // Build card grid using unified card system
        const cardsHtml = pageResults.map((r, index) => {
            let cardHtml = buildSceneCard({
                scene: r.scene || { id: r.scene_id },
                score: r.combined_score,
                cardIndex: index,
                theme: 'recs',
                scoreLabel: 'match'
            });
            // Add source badge (NEW or eye icon) for unified view
            const source = r._source;
            if (source) {
                const badgeClass = source === 'discover' ? 'stash-copilot-card-source-new' : 'stash-copilot-card-source-rewatch';
                const badgeText = source === 'discover' ? 'NEW' : '&#128065;';
                const badgeTitle = source === 'discover' ? 'Unwatched - from Discover' : 'Previously watched - from Re-watch';
                cardHtml = cardHtml.replace(
                    '<div class="stash-copilot-card-thumb">',
                    `<div class="stash-copilot-card-source-badge ${badgeClass}" title="${badgeTitle}">${badgeText}</div><div class="stash-copilot-card-thumb">`
                );
            }
            return cardHtml;
        }).join('');

        const profileInfo = data.profile || {};
        const headerText = `Recommendations (${results.length})`;
        const isSessionMode = state.recommendationsTimeDecayDays === 'session';
        const profileLabel = isSessionMode
            ? `Based on ${profileInfo.scene_count || 0} session scenes`
            : `Based on ${profileInfo.scene_count || 0} top scenes`;

        // Format timestamp
        const generatedAt = data.generatedAt || state.recommendationsGeneratedAt;
        const timeAgo = generatedAt ? formatRelativeTime(generatedAt) : '';

        // Build pagination controls
        const paginationHtml = totalPages > 1 ? `
            <div class="stash-copilot-rec-pagination">
                <button class="stash-copilot-rec-page-btn" data-page="prev" ${currentPage === 1 ? 'disabled' : ''}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="15 18 9 12 15 6"></polyline>
                    </svg>
                </button>
                <div class="stash-copilot-rec-page-numbers">
                    ${buildPageNumbers(currentPage, totalPages)}
                </div>
                <button class="stash-copilot-rec-page-btn" data-page="next" ${currentPage === totalPages ? 'disabled' : ''}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="9 18 15 12 9 6"></polyline>
                    </svg>
                </button>
            </div>
        ` : '';

        contentContainer.innerHTML = `
            <div class="stash-copilot-rec-header-info">
                <div class="stash-copilot-rec-header-left">
                    <span class="stash-copilot-rec-header-title">${headerText}</span>
                    ${timeAgo ? `<span class="stash-copilot-rec-timestamp">${timeAgo}</span>` : ''}
                </div>
                <span class="stash-copilot-rec-profile-info">${profileLabel}</span>
            </div>
            <div class="stash-copilot-rec-grid">
                ${cardsHtml}
            </div>
            ${paginationHtml}
        `;

        // Setup card interactions using unified event handler
        setupSceneCardEvents(contentContainer, { theme: 'recs', tooltipMode: 'fixed' });

        // Add click handlers to navigate to scene
        contentContainer.querySelectorAll('.stash-copilot-card').forEach(card => {
            card.addEventListener('click', (e) => {
                // Don't navigate if clicking scrubber
                if (e.target.closest('.stash-copilot-card-scrubber')) {
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                }
                e.preventDefault();
                const sceneId = card.dataset.sceneId;
                window.location.href = `/scenes/${sceneId}`;
            });
        });

        // Add pagination click handlers
        contentContainer.querySelectorAll('.stash-copilot-rec-page-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();  // Prevent dropdown from closing
                const action = btn.dataset.page;
                let newPage = currentPage;
                if (action === 'prev' && currentPage > 1) {
                    newPage = currentPage - 1;
                } else if (action === 'next' && currentPage < totalPages) {
                    newPage = currentPage + 1;
                }
                if (newPage !== currentPage) {
                    renderRecommendationsResults(dropdown, data, newPage);
                }
            });
        });

        // Add page number click handlers
        contentContainer.querySelectorAll('.stash-copilot-rec-page-num').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();  // Prevent dropdown from closing
                const pageNum = parseInt(btn.dataset.page, 10);
                if (pageNum !== currentPage && !isNaN(pageNum)) {
                    renderRecommendationsResults(dropdown, data, pageNum);
                }
            });
        });
    }

    // Build page number buttons with ellipsis for large page counts
    function buildPageNumbers(current, total) {
        const pages = [];
        const maxVisible = 5;

        if (total <= maxVisible) {
            // Show all pages
            for (let i = 1; i <= total; i++) {
                pages.push(i);
            }
        } else {
            // Always show first page
            pages.push(1);

            if (current > 3) {
                pages.push('...');
            }

            // Show pages around current
            const start = Math.max(2, current - 1);
            const end = Math.min(total - 1, current + 1);
            for (let i = start; i <= end; i++) {
                if (!pages.includes(i)) {
                    pages.push(i);
                }
            }

            if (current < total - 2) {
                pages.push('...');
            }

            // Always show last page
            if (!pages.includes(total)) {
                pages.push(total);
            }
        }

        return pages.map(p => {
            if (p === '...') {
                return '<span class="stash-copilot-rec-page-ellipsis">...</span>';
            }
            const isActive = p === current;
            return `<button class="stash-copilot-rec-page-num ${isActive ? 'active' : ''}" data-page="${p}">${p}</button>`;
        }).join('');
    }

    // ========== End Recommendations Functions ==========

    // Update clear button visibility based on whether there are messages
    function updateClearButtonVisibility(dropdown, hasMessages) {
        const clearBtn = dropdown.querySelector('.stash-copilot-clear-chat');
        if (clearBtn) {
            clearBtn.style.display = hasMessages ? 'flex' : 'none';
        }
    }

    // Load chat history and render messages
    async function loadChatHistory(dropdown) {
        const messagesContainer = dropdown.querySelector('.stash-copilot-chat-messages');
        if (!messagesContainer) return;

        const history = await fetchChatHistory();
        if (history && history.messages && history.messages.length > 0) {
            await renderChatMessages(messagesContainer, history.messages);
            // Track how many messages are now rendered
            state.lastRenderedMessageCount = history.messages.length;
            updateClearButtonVisibility(dropdown, true);

            // If still streaming, start polling
            if (history.status === 'streaming' || history.status === 'tool_executing') {
                state.isChatting = true;
                state.chatStartTime = new Date(history.updated_at).getTime() - 1000;
                pollChatResponse(dropdown);
            }
        } else {
            messagesContainer.innerHTML = '<p class="stash-copilot-chat-empty">Start a conversation by typing a message below.</p>';
            state.lastRenderedMessageCount = 0;
            updateClearButtonVisibility(dropdown, false);
        }
    }

    // Render chat messages
    async function renderChatMessages(container, messages) {
        container.innerHTML = '';

        for (const msg of messages) {
            if (msg.role === 'user') {
                container.appendChild(createUserMessage(msg));
            } else if (msg.role === 'assistant') {
                // Only render assistant message if it has content
                // (skip empty messages that only have tool_calls)
                if (msg.content && msg.content.trim()) {
                    const assistantEl = await createAssistantMessage(msg, messages);
                    container.appendChild(assistantEl);
                }

                // Render tool calls if present
                if (msg.tool_calls && msg.tool_calls.length > 0) {
                    msg.tool_calls.forEach(tc => {
                        container.appendChild(createToolCallDisplay(tc));
                    });
                }
            } else if (msg.role === 'tool_result') {
                container.appendChild(createToolResultDisplay(msg));
            }
        }

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    // Render only new chat messages (incremental update to avoid animation resets)
    async function renderNewChatMessages(container, messages, startIndex) {
        // CHANGE 1: Hide typing indicator during rendering to prevent flicker
        const typingIndicator = container.querySelector('.stash-copilot-typing');
        const wasShowingTyping = !!typingIndicator;

        // Temporarily hide it (don't remove, as polling logic expects it to exist)
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }

        for (let i = startIndex; i < messages.length; i++) {
            const msg = messages[i];
            let element = null;

            if (msg.role === 'user') {
                element = createUserMessage(msg);
            } else if (msg.role === 'assistant') {
                // Only render assistant message if it has content
                if (msg.content && msg.content.trim()) {
                    element = await createAssistantMessage(msg, messages);
                }
                // Tool calls are rendered incrementally by updateToolCallStatuses()
                // Do NOT render them here to avoid double-rendering
            } else if (msg.role === 'tool_result') {
                element = createToolResultDisplay(msg);
            }

            // CHANGE 2: Always append to end (simpler, no position juggling)
            if (element) {
                container.appendChild(element);
            }
        }

        // CHANGE 3: Restore typing indicator visibility if it was showing
        if (wasShowingTyping && typingIndicator) {
            typingIndicator.style.display = '';  // Restore to default (block)
        }

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    // Create user message element
    function createUserMessage(msg) {
        const el = document.createElement('div');
        el.className = 'stash-copilot-chat-message user';
        el.innerHTML = `<div class="stash-copilot-message-content">${escapeHtml(msg.content)}</div>`;
        return el;
    }

    /**
     * Extract per-scene metadata from tool call results across all messages.
     * Scans all assistant messages for tool_calls with results containing
     * scene data (similarity scores, frame timestamps, etc.).
     * @param {Object[]} messages - Full conversation messages array
     * @returns {Map<number, {score: number, matchTimestamp: number|null, overrideThumbnail: string|null, scoreLabel: string}>}
     */
    function extractSceneMetadataFromMessages(messages) {
        const metadata = new Map();
        if (!messages || messages.length === 0) return metadata;

        for (const msg of messages) {
            if (msg.role !== 'assistant' || !msg.tool_calls) continue;

            for (const tc of msg.tool_calls) {
                const data = tc.result?.data;
                if (!data) continue;

                // Look for results arrays from embedding tools
                const results = data.results || data.similar_scenes;
                if (!Array.isArray(results)) continue;

                const searchMode = data.search_mode || null;

                for (const item of results) {
                    const sceneId = item.scene_id;
                    if (!sceneId) continue;

                    const entry = {
                        score: item.similarity != null ? item.similarity : 1.0,
                        matchTimestamp: null,
                        overrideThumbnail: null,
                        scoreLabel: 'match',
                    };

                    // Frame-level data from filter_scenes_by_visual_content
                    if (searchMode === 'frame' && item.best_timestamp != null) {
                        entry.matchTimestamp = item.best_timestamp;
                        if (item.frame_path) {
                            entry.overrideThumbnail = `/plugin/stash-copilot/assets/${item.frame_path}`;
                        }
                        entry.scoreLabel = 'frame match';
                    }

                    metadata.set(sceneId, entry);
                }
            }
        }

        return metadata;
    }

    // Create assistant message element
    async function createAssistantMessage(msg, allMessages) {
        const el = document.createElement('div');
        el.className = 'stash-copilot-chat-message assistant';

        if (msg.content) {
            const renderedContent = renderMarkdown(msg.content);
            el.innerHTML = `<div class="stash-copilot-message-content">${renderedContent}</div>`;

            // Check for scene links and add carousel
            const sceneIds = extractSceneIdsFromHTML(renderedContent);
            if (sceneIds.length > 0) {
                const scenes = await fetchScenesById(sceneIds);
                if (scenes.length > 0) {
                    // Extract frame/similarity metadata from all tool results in conversation
                    const sceneMetadata = extractSceneMetadataFromMessages(allMessages);
                    const carouselHtml = buildSceneCarousel(scenes, sceneMetadata);
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = carouselHtml;
                    const carousel = tempDiv.firstElementChild;
                    el.appendChild(carousel);

                    // Setup event handlers for cards with cursor tooltip mode
                    setupSceneCardEvents(carousel, { theme: 'chat', tooltipMode: 'cursor' });
                }
            }
        }

        return el;
    }

    // Create tool call display element
    function createToolCallDisplay(toolCall) {
        const el = document.createElement('div');
        el.className = `stash-copilot-tool-call ${toolCall.status || 'pending'}`;
        if (toolCall.id) {
            el.setAttribute('data-tool-id', toolCall.id);
        }

        // Status icons with better visuals
        const statusIcon = {
            pending: '<span class="stash-copilot-status-dot pending"></span>',
            executing: '<span class="stash-copilot-spinner"></span>',
            completed: '<span class="stash-copilot-status-check">✓</span>',
            failed: '<span class="stash-copilot-status-x">✗</span>'
        }[toolCall.status] || '<span class="stash-copilot-status-dot"></span>';

        // Timing info
        let timingHtml = '';
        if (toolCall.started_at && toolCall.completed_at) {
            const duration = new Date(toolCall.completed_at) - new Date(toolCall.started_at);
            timingHtml = `<span class="stash-copilot-tool-timing">${duration}ms</span>`;
        }

        // Result formatting
        let resultHtml = '';
        if (toolCall.result) {
            if (toolCall.result.success) {
                resultHtml = `<div class="stash-copilot-tool-result success">${formatToolResult(toolCall.result)}</div>`;
            } else {
                resultHtml = `<div class="stash-copilot-tool-result error"><strong>Error:</strong> ${escapeHtml(toolCall.result.error || 'Unknown error')}</div>`;
            }
        }

        el.innerHTML = `
            <div class="stash-copilot-tool-header">
                ${statusIcon}
                <span class="stash-copilot-tool-name">${formatToolName(toolCall.name)}</span>
                ${timingHtml}
            </div>
            <div class="stash-copilot-tool-args">
                <code>${escapeHtml(JSON.stringify(toolCall.arguments, null, 2))}</code>
            </div>
            ${resultHtml}
        `;

        return el;
    }

    // Update an existing tool call element with new status/result
    function updateToolCallElement(el, toolCall) {
        // Update class for status
        el.className = `stash-copilot-tool-call ${toolCall.status || 'pending'}`;

        // Status icons with better visuals
        const statusIcon = {
            pending: '<span class="stash-copilot-status-dot pending"></span>',
            executing: '<span class="stash-copilot-spinner"></span>',
            completed: '<span class="stash-copilot-status-check">✓</span>',
            failed: '<span class="stash-copilot-status-x">✗</span>'
        }[toolCall.status] || '<span class="stash-copilot-status-dot"></span>';

        // Timing info
        let timingHtml = '';
        if (toolCall.started_at && toolCall.completed_at) {
            const duration = new Date(toolCall.completed_at) - new Date(toolCall.started_at);
            timingHtml = `<span class="stash-copilot-tool-timing">${duration}ms</span>`;
        }

        // Result formatting
        let resultHtml = '';
        if (toolCall.result) {
            if (toolCall.result.success) {
                resultHtml = `<div class="stash-copilot-tool-result success">${formatToolResult(toolCall.result)}</div>`;
            } else {
                resultHtml = `<div class="stash-copilot-tool-result error"><strong>Error:</strong> ${escapeHtml(toolCall.result.error || 'Unknown error')}</div>`;
            }
        }

        el.innerHTML = `
            <div class="stash-copilot-tool-header">
                ${statusIcon}
                <span class="stash-copilot-tool-name">${formatToolName(toolCall.name)}</span>
                ${timingHtml}
            </div>
            <div class="stash-copilot-tool-args">
                <code>${escapeHtml(JSON.stringify(toolCall.arguments, null, 2))}</code>
            </div>
            ${resultHtml}
        `;
    }

    // Update tool call statuses in the DOM without re-rendering messages
    // Also renders NEW tool calls incrementally as they're added by the backend
    function updateToolCallStatuses(container, messages) {
        // CHANGE 1: Hide typing indicator while adding tool calls
        const typingIndicator = container.querySelector('.stash-copilot-typing');
        const wasShowingTyping = !!typingIndicator;

        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }

        let newToolCallIndex = 0; // For staggered animation timing

        messages.forEach(msg => {
            if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
                msg.tool_calls.forEach(tc => {
                    if (tc.id) {
                        const existingEl = container.querySelector(`[data-tool-id="${tc.id}"]`);
                        if (existingEl) {
                            // Existing tool call - update status if changed
                            const currentStatus = existingEl.className.split(' ').find(c =>
                                ['pending', 'executing', 'completed', 'failed'].includes(c)
                            );
                            if (currentStatus !== tc.status) {
                                updateToolCallElement(existingEl, tc);
                            }
                        } else {
                            // NEW tool call - render it incrementally with animation
                            const tcElement = createToolCallDisplay(tc);
                            tcElement.dataset.rendered = 'true';

                            // Add entrance animation
                            tcElement.style.opacity = '0';
                            tcElement.style.transform = 'translateY(10px)';

                            // CHANGE 2: Always append (simpler, no juggling)
                            container.appendChild(tcElement);

                            // Trigger animation with small delay (stagger effect)
                            const animationDelay = newToolCallIndex * 100; // 100ms delay per tool call
                            setTimeout(() => {
                                tcElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                                tcElement.style.opacity = '1';
                                tcElement.style.transform = 'translateY(0)';
                            }, animationDelay);

                            container.scrollTop = container.scrollHeight;
                            newToolCallIndex++;
                        }
                    }
                });
            }
        });

        // CHANGE 3: Restore typing indicator
        if (wasShowingTyping && typingIndicator) {
            typingIndicator.style.display = '';
        }
    }

    // Create tool result display
    function createToolResultDisplay(msg) {
        const el = document.createElement('div');
        el.className = `stash-copilot-tool-result-msg ${msg.tool_success ? 'success' : 'error'}`;

        const icon = msg.tool_success ? '&#10003;' : '&#10007;';
        const toolName = formatToolName(msg.tool_name || 'Tool');

        el.innerHTML = `
            <span class="stash-copilot-tool-result-icon">${icon}</span>
            <span class="stash-copilot-tool-result-name">${toolName}</span>
        `;

        return el;
    }

    // Format tool result for display with expandable details
    function formatToolResult(result) {
        if (!result || !result.data) return '<span class="empty">No data</span>';

        const data = result.data;
        let summary = '';
        let details = null;

        // Format based on data type
        if (Array.isArray(data)) {
            if (data.length === 0) {
                summary = 'No results found';
            } else {
                summary = `Found <strong>${data.length} result${data.length === 1 ? '' : 's'}</strong>`;
                details = formatArrayDetails(data);
            }
        } else if (data.performers) {
            summary = `Found <strong>${data.performers.length} performer${data.performers.length === 1 ? '' : 's'}</strong>`;
            details = formatArrayDetails(data.performers);
        } else if (data.tags) {
            summary = `Found <strong>${data.tags.length} tag${data.tags.length === 1 ? '' : 's'}</strong>`;
            details = formatArrayDetails(data.tags);
        } else if (data.scenes) {
            summary = `Found <strong>${data.scenes.length} scene${data.scenes.length === 1 ? '' : 's'}</strong>`;
            details = formatArrayDetails(data.scenes);
        } else if (typeof data === 'object') {
            summary = formatObjectSummary(data);
            details = formatObjectDetails(data);
        } else {
            // Default: show primitive data
            summary = escapeHtml(String(data));
        }

        // Build HTML with expandable section
        if (details) {
            const detailsId = `details-${Math.random().toString(36).slice(2, 9)}`;
            return `
                <div class="stash-copilot-tool-result-summary">${summary}</div>
                <button class="stash-copilot-tool-expand" onclick="toggleToolDetails('${detailsId}')">
                    <span class="stash-copilot-expand-icon">▶</span> Show details
                </button>
                <div id="${detailsId}" class="stash-copilot-tool-details" style="display: none;">
                    ${details}
                </div>
            `;
        }

        return `<div class="stash-copilot-tool-result-summary">${summary}</div>`;
    }

    // Format array details with item previews
    function formatArrayDetails(arr) {
        if (!arr || arr.length === 0) return '';

        const maxItems = 10;
        const items = arr.slice(0, maxItems);
        let html = '<div class="stash-copilot-tool-items">';

        items.forEach((item, idx) => {
            if (typeof item === 'object') {
                // Show key-value pairs
                const preview = Object.entries(item)
                    .slice(0, 5)
                    .map(([k, v]) => {
                        const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
                        const truncated = val.length > 50 ? val.slice(0, 50) + '...' : val;
                        return `<span class="item-field"><strong>${k}:</strong> ${escapeHtml(truncated)}</span>`;
                    })
                    .join('');
                html += `<div class="stash-copilot-tool-item">${preview}</div>`;
            } else {
                html += `<div class="stash-copilot-tool-item">${escapeHtml(String(item))}</div>`;
            }
        });

        if (arr.length > maxItems) {
            html += `<div class="stash-copilot-tool-item-more">... and ${arr.length - maxItems} more</div>`;
        }

        html += '</div>';
        return html;
    }

    // Format object data with all key-value pairs
    function formatObjectDetails(obj) {
        if (!obj || typeof obj !== 'object') return escapeHtml(String(obj));

        let html = '<div class="stash-copilot-tool-object">';
        Object.entries(obj).forEach(([key, value]) => {
            const valStr = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
            html += `<div class="stash-copilot-object-field"><strong>${escapeHtml(key)}:</strong> ${escapeHtml(valStr)}</div>`;
        });
        html += '</div>';
        return html;
    }

    // Generate smart summary for objects
    function formatObjectSummary(obj) {
        const keys = Object.keys(obj);
        if (keys.length === 0) return 'Empty object';

        // Highlight important numeric fields
        const counts = keys.filter(k => k.includes('count') || k.includes('total'));
        if (counts.length > 0) {
            return counts.map(k => `<strong>${k}:</strong> ${obj[k]}`).join(', ');
        }

        return `Object with ${keys.length} field${keys.length !== 1 ? 's' : ''}`;
    }

    // Toggle expandable details
    window.toggleToolDetails = function(detailsId) {
        const details = document.getElementById(detailsId);
        const button = details?.previousElementSibling;
        if (!details || !button) return;

        const isHidden = details.style.display === 'none';
        details.style.display = isHidden ? 'block' : 'none';

        const icon = button.querySelector('.expand-icon');
        if (icon) {
            icon.textContent = isHidden ? '▼' : '▶';
        }

        const text = button.lastChild;
        if (text && text.nodeType === Node.TEXT_NODE) {
            text.textContent = isHidden ? ' Hide details' : ' Show details';
        }
    };

    // Escape HTML
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ===== SCENE CAROUSEL FOR CHAT MESSAGES =====

    /**
     * Extract scene IDs from rendered message HTML
     * @param {string} html - Rendered markdown HTML
     * @returns {number[]} Array of unique scene IDs
     */
    function extractSceneIdsFromHTML(html) {
        const temp = document.createElement('div');
        temp.innerHTML = html;
        const links = temp.querySelectorAll('a[href^="/scenes/"]');
        const sceneIds = new Set();

        links.forEach(link => {
            const href = link.getAttribute('href');
            const match = href.match(/^\/scenes\/(\d+)/);
            if (match && match[1]) {
                sceneIds.add(parseInt(match[1], 10));
            }
        });

        return Array.from(sceneIds);
    }

    /**
     * Fetch multiple scenes by ID using GraphQL
     * @param {number[]} sceneIds - Array of scene IDs to fetch
     * @returns {Promise<Object[]>} Array of scene objects (ordered by input IDs)
     */
    async function fetchScenesById(sceneIds) {
        if (!sceneIds || sceneIds.length === 0) return [];

        // Fetch scenes individually using findScene query
        const promises = sceneIds.map(async (sceneId) => {
            const query = `
                query FindScene($id: ID!) {
                    findScene(id: $id) {
                        id
                        title
                        date
                        rating100
                        play_count
                        o_counter
                        organized
                        interactive
                        files {
                            path
                            size
                            duration
                            height
                            width
                            fingerprints {
                                type
                                value
                            }
                        }
                        performers {
                            id
                            name
                        }
                        tags {
                            id
                            name
                        }
                        studio {
                            id
                            name
                        }
                    }
                }
            `;

            try {
                const result = await callGQL(query, { id: String(sceneId) });
                return result?.findScene || null;
            } catch (error) {
                log(`Failed to fetch scene ${sceneId}: ${error.message}`, 'error');
                return null;
            }
        });

        try {
            const scenes = await Promise.all(promises);
            return scenes.filter(Boolean); // Remove nulls
        } catch (error) {
            log(`Failed to fetch scenes for carousel: ${error.message}`, 'error');
            return [];
        }
    }

    /**
     * Build carousel HTML for scene cards
     * @param {Object[]} scenes - Array of scene objects
     * @param {Map<number, {score: number, matchTimestamp: number|null, overrideThumbnail: string|null, scoreLabel: string}>} [sceneMetadata] - Optional per-scene metadata from tool results
     * @returns {string} Carousel HTML string
     */
    function buildSceneCarousel(scenes, sceneMetadata) {
        if (!scenes || scenes.length === 0) return '';

        // Build cards using unified system with 'chat' theme
        const cardsHtml = scenes.map((scene, index) => {
            const sceneId = parseInt(scene.id || scene.scene_id, 10);
            const meta = sceneMetadata?.get(sceneId);

            return buildSceneCard({
                scene: scene,
                score: meta?.score ?? 1.0,
                cardIndex: index,
                theme: 'chat',
                scoreLabel: meta?.scoreLabel || 'linked',
                matchTimestamp: meta?.matchTimestamp ?? null,
                overrideThumbnail: meta?.overrideThumbnail ?? null,
            });
        }).join('');

        return `
            <div class="stash-copilot-message-carousel">
                <div class="stash-copilot-carousel-track">
                    ${cardsHtml}
                </div>
            </div>
        `;
    }

    // ===== UNIFIED SCENE CARD SYSTEM =====
    // Shared utilities for all scene cards (Similar, Recs, Sidebar)

    const SceneCardUtils = {
        formatDuration(seconds) {
            if (!seconds) return '';
            const s = parseFloat(seconds);
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const sec = Math.floor(s % 60);
            if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
            return `${m}:${sec.toString().padStart(2, '0')}`;
        },

        formatSize(bytes) {
            if (!bytes) return '';
            const b = parseFloat(bytes);
            const gb = b / (1024 * 1024 * 1024);
            if (gb >= 1) return `${gb.toFixed(2)} GB`;
            const mb = b / (1024 * 1024);
            return `${mb.toFixed(0)} MB`;
        },

        formatRating(rating100) {
            if (!rating100) return '';
            return `${(parseFloat(rating100) / 20).toFixed(1)}`;
        },

        getResolution(height) {
            if (!height) return '';
            if (height >= 2160) return '4K';
            if (height >= 1440) return '1440p';
            if (height >= 1080) return '1080p';
            if (height >= 720) return '720p';
            if (height >= 480) return '480p';
            return `${height}p`;
        }
    };

    /**
     * Calculate engagement score for a scene
     * Formula: (o_count * 20) + (replay_count * 2) where replay_count = max(views - 1, 0)
     * @param {number} playCount - Number of times scene was played
     * @param {number} oCount - O-counter value
     * @returns {number} Engagement score
     */
    function calculateEngagementScore(playCount, oCount) {
        const replayCount = Math.max((playCount || 0) - 1, 0);
        return (oCount || 0) * 20.0 + replayCount * 2.0;
    }

    /**
     * Build unified scene card HTML
     * @param {Object} options
     * @param {Object} options.scene - Scene data object
     * @param {number} options.score - Similarity/recommendation score (0-1)
     * @param {number} options.cardIndex - Index for staggered animation
     * @param {string} options.theme - 'similar' | 'recs' | 'scene-recs'
     * @param {string} options.scoreLabel - 'similarity' | 'match' (default: 'match')
     * @returns {string} HTML string
     */
    function buildSceneCard(options) {
        const {
            scene = {},
            score = 0,
            cardIndex = 0,
            theme = 'recs',
            scoreLabel = 'match',
            overrideThumbnail = null,
            matchTimestamp = null
        } = options;

        const sceneId = scene.id || scene.scene_id;
        if (!sceneId) return '';

        // Extract scene data with fallbacks
        let title = scene.title;
        if (!title && scene.files?.[0]?.path) {
            const path = scene.files[0].path;
            title = path.split('/').pop().split('\\').pop();
        }
        title = title || `Scene ${sceneId}`;

        const performers = (scene.performers || []).map(p => p.name).filter(Boolean).join(', ');
        const tags = (scene.tags || []).map(t => t.name);
        const studio = scene.studio?.name || '';
        const date = scene.date || '';
        const rating = scene.rating100;
        const playCount = scene.play_count || 0;
        const oCount = scene.o_counter || 0;
        const isInteractive = scene.interactive || false;
        const engagementScore = calculateEngagementScore(playCount, oCount);

        // File info
        const file = (scene.files && scene.files[0]) || {};
        const duration = file.duration;
        const height = file.height;
        const fileSize = file.size;
        const resolution = SceneCardUtils.getResolution(height);

        // Get oshash for sprite/VTT URLs
        const fingerprints = file.fingerprints || [];
        const oshash = fingerprints.find(f => f.type === 'oshash')?.value || '';

        // URLs - use overrideThumbnail for frame search results
        const screenshotUrl = overrideThumbnail || `/scene/${sceneId}/screenshot`;
        const previewUrl = `/scene/${sceneId}/preview`;
        const spriteUrl = oshash ? `/scene/${oshash}_sprite.jpg` : `/scene/${sceneId}/vtt/sprite`;
        const vttUrl = oshash ? `/scene/${oshash}_thumbs.vtt` : '';

        // Score calculation
        const scorePct = Math.round(score * 100);
        const isHighMatch = scorePct >= 90;

        return `
            <a href="/scenes/${sceneId}" class="stash-copilot-card" data-theme="${theme}"
               data-scene-id="${sceneId}"
               data-title="${escapeHtml(title)}"
               data-studio="${escapeHtml(studio)}"
               data-performers="${escapeHtml(performers)}"
               data-tags="${escapeHtml(tags.join('|||'))}"
               data-date="${date}"
               data-rating="${rating || ''}"
               data-filesize="${fileSize || ''}"
               data-duration="${duration || ''}"
               data-playcount="${playCount}"
               data-ocount="${oCount}"
               data-engagement="${engagementScore.toFixed(1)}"
               data-score="${scorePct}"
               data-score-label="${scoreLabel}"
               data-sprite-url="${spriteUrl}"
               data-vtt-url="${vttUrl}"
               style="--card-index: ${cardIndex}"
               ${playCount > 0 ? 'data-watched="true"' : ''}>
                <div class="stash-copilot-card-thumb">
                    <img class="stash-copilot-card-screenshot" src="${screenshotUrl}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.style.display='none'">
                    <video class="stash-copilot-card-preview" src="" data-preview-src="${previewUrl}" muted loop playsinline></video>
                    <div class="stash-copilot-card-sprite"></div>
                    <div class="stash-copilot-card-score${isHighMatch ? ' high-match' : ''}">${scorePct}%</div>
                    ${engagementScore > 0 ? `<div class="stash-copilot-card-engagement" title="Engagement: ${engagementScore.toFixed(1)}">🔥 ${Math.round(engagementScore)}</div>` : ''}
                    ${duration ? `<div class="stash-copilot-card-duration">${SceneCardUtils.formatDuration(duration)}</div>` : ''}
                    ${matchTimestamp !== null ? `<div class="stash-copilot-match-timestamp" title="Best matching frame">${formatTimestamp(matchTimestamp)}</div>` : ''}
                    ${isInteractive ? '<div class="stash-copilot-card-interactive" title="Interactive">🎮</div>' : ''}
                    ${resolution ? `<div class="stash-copilot-card-resolution">${resolution}</div>` : ''}
                    <div class="stash-copilot-card-scrubber" data-sprite-url="${spriteUrl}" data-vtt-url="${vttUrl}">
                        <div class="stash-copilot-card-scrubber-progress"></div>
                        <div class="stash-copilot-card-scrubber-time"></div>
                    </div>
                </div>
                <div class="stash-copilot-card-info">
                    <div class="stash-copilot-card-title">${escapeHtml(title)}</div>
                    ${performers ? `<div class="stash-copilot-card-performers">${escapeHtml(performers.split(', ').slice(0, 2).join(', '))}</div>` : ''}
                    <div class="stash-copilot-card-stats${playCount > 0 ? ' watched' : ''}">
                        <span class="${playCount > 0 ? 'has-plays' : ''}" title="Play count">▶ ${playCount}</span>
                        <span class="${oCount > 0 ? 'has-o' : ''}" title="O count">💦 ${oCount}</span>
                    </div>
                </div>
            </a>
        `;
    }

    /**
     * Setup event handlers for unified scene cards
     * @param {HTMLElement} container - Container with .stash-copilot-card elements
     * @param {Object} options
     * @param {string} options.theme - Color theme for tooltip accent
     * @param {string} options.tooltipMode - 'fixed' (beside card) | 'cursor' (follows mouse)
     */
    function setupSceneCardEvents(container, options = {}) {
        const { theme = 'recs', tooltipMode = 'fixed' } = options;
        const cards = container.querySelectorAll('.stash-copilot-card');

        // Create or get shared tooltip element
        let tooltip = document.getElementById('stash-copilot-card-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'stash-copilot-card-tooltip';
            tooltip.className = 'stash-copilot-card-tooltip';
            document.body.appendChild(tooltip);
        }

        cards.forEach(card => {
            const thumb = card.querySelector('.stash-copilot-card-thumb');
            const screenshot = card.querySelector('.stash-copilot-card-screenshot');
            const preview = card.querySelector('.stash-copilot-card-preview');
            const sprite = card.querySelector('.stash-copilot-card-sprite');
            const scrubber = card.querySelector('.stash-copilot-card-scrubber');
            const progress = scrubber?.querySelector('.stash-copilot-card-scrubber-progress');
            const timeDisplay = scrubber?.querySelector('.stash-copilot-card-scrubber-time');

            let previewLoaded = false;
            let hoverTimeout = null;
            let tooltipTimeout = null;
            let isScrubbing = false;

            // ===== FRAME THUMBNAIL FALLBACK =====
            // If frame thumbnail fails to load, fall back to scene screenshot
            if (screenshot && screenshot.src.includes('/embedded_frames/')) {
                screenshot.onerror = function() {
                    const sceneId = card.dataset.sceneId;
                    if (sceneId) {
                        this.src = `/scene/${sceneId}/screenshot`;
                        this.onerror = null; // Prevent infinite loop
                    }
                };
            }

            // ===== HOVER: Show tooltip and start preview =====
            card.addEventListener('mouseenter', () => {
                // Show tooltip after 250ms
                tooltipTimeout = setTimeout(() => {
                    const cardTheme = card.dataset.theme || theme;
                    const title = card.dataset.title || '';
                    const studio = card.dataset.studio || '';
                    const performers = card.dataset.performers || '';
                    const tagsStr = card.dataset.tags || '';
                    const date = card.dataset.date || '';
                    const rating = card.dataset.rating || '';
                    const filesize = card.dataset.filesize || '';
                    const duration = card.dataset.duration || '';
                    const playcount = card.dataset.playcount || '0';
                    const ocount = card.dataset.ocount || '0';
                    const score = card.dataset.score || '';
                    const scoreLabel = card.dataset.scoreLabel || 'match';

                    const tags = tagsStr ? tagsStr.split('|||').slice(0, 8) : [];
                    const tagsHtml = tags.map(t => `<span class="stash-copilot-card-tooltip-tag">${escapeHtml(t)}</span>`).join('');

                    tooltip.dataset.theme = cardTheme;
                    tooltip.innerHTML = `
                        <div class="stash-copilot-card-tooltip-header">
                            <div class="stash-copilot-card-tooltip-title">${escapeHtml(title)}</div>
                            ${studio ? `<div class="stash-copilot-card-tooltip-studio">${escapeHtml(studio)}</div>` : ''}
                        </div>
                        ${performers ? `<div class="stash-copilot-card-tooltip-performers">${escapeHtml(performers)}</div>` : ''}
                        <div class="stash-copilot-card-tooltip-meta">
                            ${date ? `<span>📅 ${date}</span>` : ''}
                            ${duration ? `<span>⏱ ${SceneCardUtils.formatDuration(duration)}</span>` : ''}
                            ${filesize ? `<span>💾 ${SceneCardUtils.formatSize(filesize)}</span>` : ''}
                            ${rating ? `<span>⭐ ${SceneCardUtils.formatRating(rating)}</span>` : ''}
                        </div>
                        <div class="stash-copilot-card-tooltip-stats">
                            <span class="${parseInt(playcount) > 0 ? 'has-value' : ''}">▶ ${playcount} plays</span>
                            <span class="${parseInt(ocount) > 0 ? 'has-value' : ''}">💦 ${ocount}</span>
                            <span class="stash-copilot-card-tooltip-score">${score}% ${scoreLabel}</span>
                        </div>
                        ${tagsHtml ? `<div class="stash-copilot-card-tooltip-tags">${tagsHtml}</div>` : ''}
                    `;

                    // Position tooltip
                    if (tooltipMode === 'fixed') {
                        const cardRect = card.getBoundingClientRect();
                        const tooltipWidth = 260;
                        const margin = 10;

                        let left = cardRect.right + margin;
                        let top = cardRect.top;

                        // If would go off-screen right, position to left
                        if (left + tooltipWidth > window.innerWidth) {
                            left = cardRect.left - tooltipWidth - margin;
                        }

                        // Keep within vertical bounds
                        const tooltipHeight = 200;
                        if (top + tooltipHeight > window.innerHeight) {
                            top = window.innerHeight - tooltipHeight - margin;
                        }
                        if (top < margin) top = margin;

                        tooltip.style.left = `${left}px`;
                        tooltip.style.top = `${top}px`;
                    }

                    tooltip.classList.add('visible');
                }, 250);

                // Start video preview after 300ms
                if (preview) {
                    hoverTimeout = setTimeout(() => {
                        if (!previewLoaded && preview.dataset.previewSrc) {
                            preview.src = preview.dataset.previewSrc;
                            previewLoaded = true;
                        }
                        preview.play().then(() => {
                            preview.style.display = 'block';
                            if (screenshot) screenshot.style.opacity = '0';
                        }).catch(() => {});
                    }, 300);
                }
            });

            // ===== CURSOR MOVE: Update tooltip position (cursor mode) =====
            if (tooltipMode === 'cursor') {
                card.addEventListener('mousemove', (e) => {
                    if (!tooltip.classList.contains('visible')) return;

                    const margin = 15;
                    let left = e.clientX + margin;
                    let top = e.clientY + margin;

                    // Keep within viewport bounds
                    const tooltipRect = tooltip.getBoundingClientRect();
                    if (left + tooltipRect.width > window.innerWidth) {
                        left = e.clientX - tooltipRect.width - margin;
                    }
                    if (top + tooltipRect.height > window.innerHeight) {
                        top = e.clientY - tooltipRect.height - margin;
                    }

                    tooltip.style.left = `${left}px`;
                    tooltip.style.top = `${top}px`;
                });
            }

            // ===== MOUSE LEAVE: Hide tooltip and stop preview =====
            card.addEventListener('mouseleave', () => {
                clearTimeout(hoverTimeout);
                clearTimeout(tooltipTimeout);

                tooltip.classList.remove('visible');

                if (preview) {
                    preview.pause();
                    preview.style.display = 'none';
                }
                if (screenshot) screenshot.style.opacity = '1';

                // Reset scrubber state
                if (thumb) thumb.classList.remove('scrubbing');
                if (sprite) sprite.style.opacity = '0';
                if (progress) progress.style.width = '0';
                if (timeDisplay) timeDisplay.style.display = 'none';
                isScrubbing = false;
            });

            // ===== SCRUBBER: Sprite scrubbing with VTT support =====
            if (scrubber && thumb) {
                const spriteUrl = scrubber.dataset.spriteUrl;
                const vttUrl = scrubber.dataset.vttUrl || card.dataset.vttUrl || '';
                const duration = parseFloat(card.dataset.duration) || 0;

                // Sprite data storage
                let sprites = [];
                let spriteScale = 1;
                let spriteLoaded = false;

                // Parse VTT time string to seconds
                function parseVTTTime(timeStr) {
                    const parts = timeStr.split(':');
                    const hours = parseInt(parts[0]);
                    const minutes = parseInt(parts[1]);
                    const seconds = parseFloat(parts[2]);
                    return hours * 3600 + minutes * 60 + seconds;
                }

                // Load and parse VTT file for sprite coordinates
                async function loadSpriteVTT() {
                    try {
                        if (!vttUrl) throw new Error('No VTT URL');
                        const response = await fetch(vttUrl);
                        if (!response.ok) throw new Error('VTT fetch failed');
                        const vttText = await response.text();

                        const lines = vttText.split('\n');
                        let currentStart = 0;
                        let currentEnd = 0;

                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i].trim();

                            // Match timestamp: 00:00:00.000 --> 00:00:05.000
                            const timeMatch = line.match(/^(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)/);
                            if (timeMatch) {
                                currentStart = parseVTTTime(timeMatch[1]);
                                currentEnd = parseVTTTime(timeMatch[2]);
                                continue;
                            }

                            // Match sprite: url#xywh=x,y,w,h
                            const spriteMatch = line.match(/^([^#]+)#xywh=(\d+),(\d+),(\d+),(\d+)$/i);
                            if (spriteMatch) {
                                const baseUrl = window.location.origin;
                                const spritePathUrl = spriteMatch[1].startsWith('/')
                                    ? baseUrl + spriteMatch[1]
                                    : new URL(spriteMatch[1], baseUrl + vttUrl).href;
                                sprites.push({
                                    url: spritePathUrl,
                                    start: currentStart,
                                    end: currentEnd,
                                    x: parseInt(spriteMatch[2]),
                                    y: parseInt(spriteMatch[3]),
                                    w: parseInt(spriteMatch[4]),
                                    h: parseInt(spriteMatch[5])
                                });
                            }
                        }

                        if (sprites.length > 0) {
                            const containerRect = thumb.getBoundingClientRect();
                            const containerW = containerRect.width;
                            const containerH = containerRect.height;
                            const frameW = sprites[0].w;
                            const frameH = sprites[0].h;

                            // Scale for "contain" fit
                            const scaleX = containerW / frameW;
                            const scaleY = containerH / frameH;
                            spriteScale = Math.min(scaleX, scaleY);

                            // Load sprite image for full dimensions
                            const tempImg = new Image();
                            tempImg.onload = () => {
                                const scaledSheetW = tempImg.width * spriteScale;
                                const scaledSheetH = tempImg.height * spriteScale;
                                const scaledFrameW = frameW * spriteScale;
                                const scaledFrameH = frameH * spriteScale;
                                const offsetX = (containerW - scaledFrameW) / 2;
                                const offsetY = (containerH - scaledFrameH) / 2;

                                sprite.style.width = `${scaledFrameW}px`;
                                sprite.style.height = `${scaledFrameH}px`;
                                sprite.style.left = `${offsetX}px`;
                                sprite.style.top = `${offsetY}px`;
                                sprite.style.backgroundImage = `url(${sprites[0].url})`;
                                sprite.style.backgroundSize = `${scaledSheetW}px ${scaledSheetH}px`;
                                sprite.style.transform = 'none';
                                sprite.dataset.spriteScale = spriteScale;

                                // Initial position
                                const scaledX = sprites[0].x * spriteScale;
                                const scaledY = sprites[0].y * spriteScale;
                                sprite.style.backgroundPosition = `-${scaledX}px -${scaledY}px`;

                                spriteLoaded = true;
                            };
                            tempImg.src = sprites[0].url;
                        }
                    } catch (err) {
                        // Fallback to grid-based approach
                        loadSpriteFallback();
                    }
                }

                // Fallback: 9x9 grid if VTT unavailable
                function loadSpriteFallback() {
                    if (!spriteUrl) return;

                    const SPRITE_COLS = 9;
                    const SPRITE_ROWS = 9;

                    const spriteImage = new Image();
                    spriteImage.onload = () => {
                        const imgW = spriteImage.width;
                        const imgH = spriteImage.height;
                        const actualFrameW = imgW / SPRITE_COLS;
                        const actualFrameH = imgH / SPRITE_ROWS;

                        const containerRect = thumb.getBoundingClientRect();
                        const containerW = containerRect.width;
                        const containerH = containerRect.height;

                        const scaleX = containerW / actualFrameW;
                        const scaleY = containerH / actualFrameH;
                        spriteScale = Math.min(scaleX, scaleY);

                        // Build sprites array from grid
                        for (let row = 0; row < SPRITE_ROWS; row++) {
                            for (let col = 0; col < SPRITE_COLS; col++) {
                                sprites.push({
                                    url: spriteUrl,
                                    x: col * actualFrameW,
                                    y: row * actualFrameH,
                                    w: actualFrameW,
                                    h: actualFrameH
                                });
                            }
                        }

                        const scaledSheetW = imgW * spriteScale;
                        const scaledSheetH = imgH * spriteScale;
                        const scaledFrameW = actualFrameW * spriteScale;
                        const scaledFrameH = actualFrameH * spriteScale;
                        const offsetX = (containerW - scaledFrameW) / 2;
                        const offsetY = (containerH - scaledFrameH) / 2;

                        sprite.style.width = `${scaledFrameW}px`;
                        sprite.style.height = `${scaledFrameH}px`;
                        sprite.style.left = `${offsetX}px`;
                        sprite.style.top = `${offsetY}px`;
                        sprite.style.backgroundImage = `url(${spriteUrl})`;
                        sprite.style.backgroundSize = `${scaledSheetW}px ${scaledSheetH}px`;
                        sprite.style.backgroundPosition = '0 0';
                        sprite.style.transform = 'none';
                        sprite.dataset.spriteScale = spriteScale;

                        spriteLoaded = true;
                    };
                    spriteImage.src = spriteUrl;
                }

                if (spriteUrl || vttUrl) {
                    scrubber.addEventListener('mouseenter', async () => {
                        isScrubbing = true;
                        scrubber.classList.add('active');
                        if (thumb) thumb.classList.add('scrubbing');

                        // Pause preview while scrubbing
                        if (preview) {
                            preview.pause();
                            preview.style.display = 'none';
                        }

                        // Load sprites on first hover
                        if (!spriteLoaded && sprites.length === 0) {
                            await loadSpriteVTT();
                        }
                    });

                    scrubber.addEventListener('mousemove', (e) => {
                        if (!spriteLoaded || sprites.length === 0) return;

                        const rect = scrubber.getBoundingClientRect();
                        const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                        const time = percent * duration;

                        // Update progress bar
                        if (progress) progress.style.width = `${percent * 100}%`;

                        // Update time display
                        if (timeDisplay && duration > 0) {
                            timeDisplay.textContent = SceneCardUtils.formatDuration(time);
                            timeDisplay.style.display = 'block';
                            const timeWidth = timeDisplay.offsetWidth || 40;
                            let leftPos = percent * rect.width - timeWidth / 2;
                            leftPos = Math.max(0, Math.min(leftPos, rect.width - timeWidth));
                            timeDisplay.style.left = `${leftPos}px`;
                        }

                        // Find sprite for current time (VTT) or by percentage (grid fallback)
                        let spriteData = null;
                        if (sprites[0]?.start !== undefined) {
                            // VTT-based: find by time
                            spriteData = sprites.find(s => time >= s.start && time < s.end);
                        } else {
                            // Grid-based: find by percentage
                            const idx = Math.min(Math.floor(percent * sprites.length), sprites.length - 1);
                            spriteData = sprites[idx];
                        }

                        if (spriteData && sprite) {
                            const scale = parseFloat(sprite.dataset.spriteScale) || spriteScale;
                            const scaledX = spriteData.x * scale;
                            const scaledY = spriteData.y * scale;
                            sprite.style.backgroundPosition = `-${scaledX}px -${scaledY}px`;
                            sprite.style.opacity = '1';

                            if (screenshot) screenshot.style.opacity = '0';
                        }
                    });

                    scrubber.addEventListener('mouseleave', () => {
                        isScrubbing = false;
                        scrubber.classList.remove('active');
                        if (thumb) thumb.classList.remove('scrubbing');
                        if (sprite) sprite.style.opacity = '0';
                        if (progress) progress.style.width = '0';
                        if (timeDisplay) timeDisplay.style.display = 'none';
                        if (screenshot) screenshot.style.opacity = '1';

                        // Resume preview if hovering
                        if (preview && previewLoaded && card.matches(':hover')) {
                            preview.play().catch(() => {});
                            preview.style.display = 'block';
                        }
                    });
                }
            }
        });
    }

    // ===== END UNIFIED SCENE CARD SYSTEM =====

    // Send a chat message
    async function sendChatMessage(dropdown, message) {
        if (!message.trim() || state.isChatting) return;

        const input = dropdown.querySelector('.stash-copilot-chat-input');
        const sendBtn = dropdown.querySelector('.stash-copilot-chat-send');
        const messagesContainer = dropdown.querySelector('.stash-copilot-chat-messages');

        state.isChatting = true;
        state.chatStartTime = Date.now();

        // Clear empty state message if present
        const emptyMsg = messagesContainer.querySelector('.stash-copilot-chat-empty');
        if (emptyMsg) emptyMsg.remove();

        // Show clear button now that we have messages
        updateClearButtonVisibility(dropdown, true);

        // Immediately render user message
        messagesContainer.appendChild(createUserMessage({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        }));
        // Track that we've rendered this message locally (it will also be in backend history)
        state.lastRenderedMessageCount++;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Clear and disable input
        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;

        // Show typing indicator
        showTypingIndicator(messagesContainer);

        try {
            await runPluginTask('Chat', {
                message: message,
                conversation_id: state.conversationId || ''
            });

            log('Chat task started');

            // Start polling for response
            pollChatResponse(dropdown);

        } catch (error) {
            log(`Chat send error: ${error.message}`, 'error');
            hideTypingIndicator(messagesContainer);
            state.isChatting = false;
            input.disabled = false;
            sendBtn.disabled = false;

            // Show error message
            const errorEl = document.createElement('div');
            errorEl.className = 'stash-copilot-chat-message error';
            errorEl.innerHTML = `<div class="stash-copilot-message-content">Failed to send message. Please try again.</div>`;
            messagesContainer.appendChild(errorEl);
        }
    }

    // Show typing indicator
    function showTypingIndicator(container) {
        // Remove existing typing indicator
        hideTypingIndicator(container);

        const typing = document.createElement('div');
        typing.className = 'stash-copilot-typing';
        typing.innerHTML = `
            <span>AI is thinking</span>
            <div class="stash-copilot-typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;
        container.appendChild(typing);
        container.scrollTop = container.scrollHeight;
    }

    // Hide typing indicator
    function hideTypingIndicator(container) {
        const typing = container.querySelector('.stash-copilot-typing');
        if (typing) typing.remove();
    }

    // Poll for chat response updates
    function pollChatResponse(dropdown) {
        // Clear any existing poll interval
        if (state.chatPollInterval) {
            clearInterval(state.chatPollInterval);
        }

        const input = dropdown.querySelector('.stash-copilot-chat-input');
        const sendBtn = dropdown.querySelector('.stash-copilot-chat-send');
        const messagesContainer = dropdown.querySelector('.stash-copilot-chat-messages');

        let attempts = 0;
        const maxAttempts = 600; // 2 minutes

        state.chatPollInterval = setInterval(async () => {
            attempts++;

            const history = await fetchChatHistory();
            if (!history) return;

            // Only process updates after we started chatting (unless status is terminal)
            const historyTime = new Date(history.updated_at).getTime();
            const isTerminalStatus = history.status === 'complete' || history.status === 'error' || history.status === 'idle';

            // Early exit only if timestamp is old AND status is not terminal
            if (historyTime < state.chatStartTime && !isTerminalStatus) return;

            // Only render if there are new messages (incremental update)
            const newMessageCount = history.messages.length;
            if (newMessageCount > state.lastRenderedMessageCount) {
                // Render only the new messages (preserves typing indicator animation)
                renderNewChatMessages(messagesContainer, history.messages, state.lastRenderedMessageCount);
                state.lastRenderedMessageCount = newMessageCount;
            }

            // Update tool call statuses for already-rendered messages
            updateToolCallStatuses(messagesContainer, history.messages);

            // Only toggle typing indicator if status actually changed
            const isProcessing = history.status === 'streaming' || history.status === 'tool_executing';
            const hasTypingIndicator = !!messagesContainer.querySelector('.stash-copilot-typing');

            if (isProcessing && !hasTypingIndicator) {
                showTypingIndicator(messagesContainer);
            } else if (!isProcessing && hasTypingIndicator) {
                hideTypingIndicator(messagesContainer);
            }

            // Check status - only stop if ALL conditions are met:
            // 1. Status is terminal (complete/error/idle)
            // 2. Timestamp is from current chat session (not stale)
            // 3. All messages have been rendered (no race condition)
            if (isTerminalStatus && historyTime >= state.chatStartTime && newMessageCount === state.lastRenderedMessageCount) {
                clearInterval(state.chatPollInterval);
                state.chatPollInterval = null;
                hideTypingIndicator(messagesContainer);
                state.isChatting = false;
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            }

            if (attempts >= maxAttempts) {
                clearInterval(state.chatPollInterval);
                state.chatPollInterval = null;
                hideTypingIndicator(messagesContainer);
                state.isChatting = false;
                input.disabled = false;
                sendBtn.disabled = false;
                log('Chat polling timed out', 'warn');
            }
        }, 200);
    }

    // Clear chat history
    async function clearChatHistory(dropdown) {
        state.chatHistory = null;
        state.conversationId = null;
        state.lastRenderedMessageCount = 0;

        const messagesContainer = dropdown.querySelector('.stash-copilot-chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '<p class="stash-copilot-chat-empty">Start a conversation by typing a message below.</p>';
        }

        // Hide clear button
        updateClearButtonVisibility(dropdown, false);

        // Clear server-side history
        try {
            await runPluginTask('Clear Chat', {});
            log('Chat history cleared');
        } catch (error) {
            log('Failed to clear chat history on server: ' + error.message, 'error');
        }
    }

    // ===== AI Insights Modal Functions =====

    function createInsightsModal() {
        // Remove existing modal if present
        const existing = document.getElementById('stash-copilot-insights-modal');
        if (existing) existing.remove();

        const savedTab = getActiveTab();
        state.activeTab = savedTab;

        const modal = document.createElement('div');
        modal.id = 'stash-copilot-insights-modal';
        modal.className = 'stash-copilot-insights-modal';
        modal.setAttribute('data-active-tab', savedTab);
        modal.innerHTML = `
            <div class="stash-copilot-insights-content">
                <div class="stash-copilot-insights-header">
                    <div class="stash-copilot-insights-title">
                        <h3>AI Library Insights</h3>
                    </div>
                    <div class="stash-copilot-insights-header-actions">
                        <button class="btn btn-secondary stash-copilot-refresh-btn" title="Refresh">↻</button>
                        <button class="stash-copilot-insights-close">&times;</button>
                    </div>
                </div>

                <div class="stash-copilot-insights-tabs">
                    <button class="stash-copilot-insights-tab ${savedTab === 'summary' ? 'active' : ''}" data-tab="summary">Summary</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'chat' ? 'active' : ''}" data-tab="chat">Chat</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'tools' ? 'active' : ''}" data-tab="tools">Tools</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'recommendations' ? 'active' : ''}" data-tab="recommendations">Recs</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'peak' ? 'active' : ''}" data-tab="peak">🔥 Peak</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'taste_map' ? 'active' : ''}" data-tab="taste_map">Taste Map</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'train' ? 'active' : ''}" data-tab="train">🧠 Train</button>
                    <button class="stash-copilot-insights-tab ${savedTab === 'tag_gaps' ? 'active' : ''}" data-tab="tag_gaps">Tag Gaps</button>
                </div>

                <div class="stash-copilot-insights-body">
                    <!-- Summary Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'summary' ? 'active' : ''}" data-tab="summary">
                        <div class="stash-copilot-stats">
                            <div class="stash-copilot-spinner"></div> Loading...
                        </div>
                        <div class="stash-copilot-summary-section">
                            <div class="stash-copilot-summary-header">
                                <h4>Latest Summary</h4>
                                <button class="btn btn-primary stash-copilot-generate-btn">Generate Summary</button>
                            </div>
                            <div class="stash-copilot-summary-content">
                                <p class="stash-copilot-info">No summary generated yet.</p>
                            </div>
                        </div>
                    </div>

                    <!-- Chat Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'chat' ? 'active' : ''}" data-tab="chat">
                        <div class="stash-copilot-chat-container">
                            <div class="stash-copilot-chat-messages">
                                <p class="stash-copilot-chat-empty">Start a conversation by typing a message below.</p>
                            </div>
                            <div class="stash-copilot-chat-input-container">
                                <input type="text" class="stash-copilot-chat-input"
                                       placeholder="Ask about your library..."
                                       maxlength="500">
                                <button class="btn btn-primary stash-copilot-chat-send">Send</button>
                                <button class="stash-copilot-clear-chat" title="Clear conversation" style="display: none;">
                                    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Tools Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'tools' ? 'active' : ''}" data-tab="tools">
                        <div class="stash-copilot-tools-list"></div>
                    </div>

                    <!-- Recommendations Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'recommendations' ? 'active' : ''}" data-tab="recommendations">
                        <div class="stash-copilot-rec-controls">
                            <!-- View filter pills -->
                            <div class="stash-copilot-rec-mode-tabs">
                                <button class="stash-copilot-rec-viewfilter ${state.recommendationsViewFilter === 'all' ? 'active' : ''}"
                                        data-filter="all"
                                        title="Show all recommendations">
                                    All
                                </button>
                                <button class="stash-copilot-rec-viewfilter ${state.recommendationsViewFilter === 'new' ? 'active' : ''}"
                                        data-filter="new"
                                        title="Show only unwatched scenes">
                                    New
                                </button>
                                <button class="stash-copilot-rec-viewfilter ${state.recommendationsViewFilter === 'rewatch' ? 'active' : ''}"
                                        data-filter="rewatch"
                                        title="Show only previously watched scenes">
                                    Re-watch
                                </button>
                            </div>
                            <div class="stash-copilot-rec-controls-row">
                                <div class="stash-copilot-rec-controls-left">
                                    <label>Recency:</label>
                                    <select class="stash-copilot-rec-decay-select">
                                        <option value="session" ${state.recommendationsTimeDecayDays === 'session' ? 'selected' : ''}>This Session (${getSessionScenes().length})</option>
                                        <option value="3" ${state.recommendationsTimeDecayDays === 3 ? 'selected' : ''}>3 days</option>
                                        <option value="7" ${state.recommendationsTimeDecayDays === 7 ? 'selected' : ''}>7 days</option>
                                        <option value="14" ${state.recommendationsTimeDecayDays === 14 ? 'selected' : ''}>14 days</option>
                                        <option value="28" ${state.recommendationsTimeDecayDays === 28 ? 'selected' : ''}>28 days</option>
                                        <option value="60" ${state.recommendationsTimeDecayDays === 60 ? 'selected' : ''}>60 days</option>
                                        <option value="90" ${state.recommendationsTimeDecayDays === 90 ? 'selected' : ''}>90 days</option>
                                        <option value="180" ${state.recommendationsTimeDecayDays === 180 ? 'selected' : ''}>180 days</option>
                                        <option value="365" ${state.recommendationsTimeDecayDays === 365 ? 'selected' : ''}>1 year</option>
                                        <option value="0" ${state.recommendationsTimeDecayDays === 0 ? 'selected' : ''}>All time</option>
                                    </select>
                                    <button class="stash-copilot-rec-clear-session-btn" title="Clear session" style="display: ${state.recommendationsTimeDecayDays === 'session' ? 'inline-flex' : 'none'}">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path>
                                        </svg>
                                    </button>
                                </div>
                                <button class="btn btn-primary stash-copilot-rec-generate-btn">Generate</button>
                            </div>
                        </div>
                        <div class="stash-copilot-recommendations-content">
                            <div class="stash-copilot-empty-state stash-copilot-rec-empty-state">
                                <div class="stash-copilot-empty-state-icon">✨</div>
                                <div class="stash-copilot-empty-state-stats stash-copilot-rec-stats-placeholder">
                                    <span class="stash-copilot-spinner"></span>
                                </div>
                                <div class="stash-copilot-empty-state-desc stash-copilot-rec-mode-desc">
                                    <h4>Personalized Recommendations</h4>
                                    <p>Discover new scenes and revisit favorites based on your viewing patterns and engagement history.</p>
                                </div>
                                <div class="stash-copilot-empty-state-tip">
                                    <p>Select a recency window above and click <strong>Generate</strong> to find your next favorites.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Peak Moments Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'peak' ? 'active' : ''}" data-tab="peak">
                        <div class="stash-copilot-peak-content">
                            <div class="stash-copilot-empty-state stash-copilot-peak-empty-state">
                                <div class="stash-copilot-empty-state-icon">🔥</div>
                                <div class="stash-copilot-empty-state-stats stash-copilot-peak-stats-display">
                                    <span class="stash-copilot-spinner"></span>
                                </div>
                                <div class="stash-copilot-empty-state-desc">
                                    <h4>Peak Moments Discovery</h4>
                                    <p>Find scenes with similar peak moments based on frames captured around your O markers.</p>
                                </div>
                                <div class="stash-copilot-peak-controls">
                                    <button class="btn btn-primary stash-copilot-peak-generate-btn">Generate</button>
                                    <button class="btn btn-secondary stash-copilot-peak-embed-btn" title="Embed O-Moments">
                                        🎬 Embed Markers
                                    </button>
                                </div>
                                <div class="stash-copilot-empty-state-tip stash-copilot-peak-tip">
                                    <p>Click <strong>Generate</strong> to discover similar scenes, or <strong>Embed Markers</strong> to process new O markers.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Taste Map Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'taste_map' ? 'active' : ''}" data-tab="taste_map">
                        <div class="stash-copilot-taste-map-container">
                            <div class="stash-copilot-taste-map-toolbar">
                                <button class="btn btn-primary stash-copilot-taste-map-build-btn">Build Taste Map</button>
                                <label class="stash-copilot-taste-map-k-label" title="Number of clusters (leave empty for auto-detect)">
                                    k: <input type="number" class="stash-copilot-taste-map-k-input" min="2" max="20" placeholder="auto" />
                                </label>
                                <span class="stash-copilot-taste-map-status"></span>
                            </div>
                            <div class="stash-copilot-taste-map-content" style="display:none">
                                <div class="stash-copilot-taste-map-main">
                                    <div class="stash-copilot-taste-map-chart" id="taste-map-chart"></div>
                                </div>
                                <div class="stash-copilot-taste-map-sidebar">
                                    <div class="stash-copilot-taste-map-sidebar-header" title="Toggle sidebar">
                                        <span>CLUSTERS</span>
                                        <span class="stash-copilot-taste-map-sidebar-toggle">◀</span>
                                    </div>
                                    <div class="stash-copilot-taste-map-clusters"></div>
                                </div>
                            </div>
                            <div class="stash-copilot-taste-map-empty">
                                <p>Build a Taste Map to visualize your preference clusters.</p>
                                <p class="stash-copilot-taste-map-empty-sub">Your most engaged scenes are grouped by visual similarity, auto-labeled, and plotted in 3D space.</p>
                            </div>
                        </div>
                    </div>

                    <!-- Train Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'train' ? 'active' : ''}" data-tab="train">
                        <div class="stash-copilot-train">
                            <div class="stash-copilot-train-intro">
                                <div class="stash-copilot-train-header">
                                    <span class="stash-copilot-train-icon">🧠</span>
                                    <h3>Train Your Taste</h3>
                                </div>
                                <p class="stash-copilot-train-desc">Swipe through scenes to teach the AI your preferences. Each swipe refines your personal recommendation model.</p>
                                <div class="stash-copilot-train-stats-row">
                                    <div class="stash-copilot-train-stat">
                                        <span class="stash-copilot-train-stat-value" data-stat="comparisons">0</span>
                                        <span class="stash-copilot-train-stat-label">Comparisons</span>
                                    </div>
                                    <div class="stash-copilot-train-stat">
                                        <span class="stash-copilot-train-stat-value" data-stat="confidence">0%</span>
                                        <span class="stash-copilot-train-stat-label">Confidence</span>
                                    </div>
                                    <div class="stash-copilot-train-stat">
                                        <span class="stash-copilot-train-stat-value" data-stat="phase">-</span>
                                        <span class="stash-copilot-train-stat-label">Phase</span>
                                    </div>
                                </div>
                                <div class="stash-copilot-taste-profile" style="display: none;">
                                    <div class="stash-copilot-taste-profile-section stash-copilot-taste-profile-likes">
                                        <span class="stash-copilot-taste-profile-label">Into</span>
                                        <div class="stash-copilot-taste-profile-pills"></div>
                                    </div>
                                    <div class="stash-copilot-taste-profile-section stash-copilot-taste-profile-dislikes">
                                        <span class="stash-copilot-taste-profile-label">Not into</span>
                                        <div class="stash-copilot-taste-profile-pills"></div>
                                    </div>
                                </div>
                                <div class="stash-copilot-train-exploration-slider">
                                    <div class="stash-copilot-train-exploration-header">
                                        <span class="stash-copilot-train-exploration-title">Scene Diversity</span>
                                        <span class="stash-copilot-train-exploration-value">20%</span>
                                    </div>
                                    <div class="stash-copilot-train-exploration-track">
                                        <span class="stash-copilot-train-exploration-label">Focused</span>
                                        <input type="range" min="0" max="100" value="20"
                                               class="stash-copilot-train-exploration-input"
                                               title="Higher = more diverse scenes, lower = faster convergence">
                                        <span class="stash-copilot-train-exploration-label">Diverse</span>
                                    </div>
                                    <p class="stash-copilot-train-exploration-hint">Higher diversity shows more varied scenes; lower focuses on refining your taste profile faster.</p>
                                </div>
                                <div class="stash-copilot-train-pure-random">
                                    <label class="stash-copilot-train-pure-random-label">
                                        <input type="checkbox" class="stash-copilot-train-pure-random-checkbox">
                                        <span class="stash-copilot-train-pure-random-text">Pure Random Mode</span>
                                    </label>
                                    <p class="stash-copilot-train-pure-random-hint">Start fresh with completely random scenes. No bias from viewing history, clusters, or engagement data.</p>
                                </div>
                                <button class="stash-copilot-train-start-btn">Start Training</button>
                                <button class="stash-copilot-train-reset-btn" style="display: none;">Reset Model</button>
                            </div>
                            <div class="stash-copilot-train-session" style="display: none;">
                                <div class="stash-copilot-train-progress">
                                    <div class="stash-copilot-train-progress-bar">
                                        <div class="stash-copilot-train-progress-fill" style="width: 0%"></div>
                                    </div>
                                    <span class="stash-copilot-train-progress-text">0 / 0</span>
                                </div>
                                <div class="stash-copilot-train-confidence-meter">
                                    <div class="stash-copilot-train-confidence-bar">
                                        <div class="stash-copilot-train-confidence-fill" style="width: 0%"></div>
                                    </div>
                                    <span class="stash-copilot-train-confidence-text">0% confident</span>
                                </div>
                                <div class="stash-copilot-train-card-area">
                                    <div class="stash-copilot-train-loading">
                                        <div class="stash-copilot-spinner"></div>
                                        <span>Loading scenes...</span>
                                    </div>
                                </div>
                                <div class="stash-copilot-train-actions" style="display: none;">
                                    <button class="stash-copilot-train-action-btn dislike" data-action="dislike" title="Dislike (Left Arrow)">
                                        <span>👎</span>
                                    </button>
                                    <button class="stash-copilot-train-action-btn skip" data-action="skip" title="Skip (Down Arrow)">
                                        <span>⏭</span>
                                    </button>
                                    <button class="stash-copilot-train-action-btn like" data-action="like" title="Like (Right Arrow)">
                                        <span>👍</span>
                                    </button>
                                    <button class="stash-copilot-train-action-btn super-like" data-action="super_like" title="Super Like (Up Arrow)">
                                        <span>🔥</span>
                                    </button>
                                </div>
                                <button class="stash-copilot-train-end-btn">End Session</button>
                            </div>
                            <div class="stash-copilot-train-complete" style="display: none;">
                                <div class="stash-copilot-train-complete-icon">✨</div>
                                <h3>Session Complete!</h3>
                                <div class="stash-copilot-train-summary">
                                    <p><strong data-stat="final-comparisons">0</strong> comparisons recorded</p>
                                    <p><strong data-stat="final-confidence">0%</strong> model confidence</p>
                                    <p>Phase: <strong data-stat="final-phase">-</strong></p>
                                </div>
                                <div class="stash-copilot-taste-profile" style="display: none;">
                                    <div class="stash-copilot-taste-profile-section stash-copilot-taste-profile-likes">
                                        <span class="stash-copilot-taste-profile-label">Into</span>
                                        <div class="stash-copilot-taste-profile-pills"></div>
                                    </div>
                                    <div class="stash-copilot-taste-profile-section stash-copilot-taste-profile-dislikes">
                                        <span class="stash-copilot-taste-profile-label">Not into</span>
                                        <div class="stash-copilot-taste-profile-pills"></div>
                                    </div>
                                </div>
                                <div class="stash-copilot-train-complete-actions">
                                    <button class="stash-copilot-train-restart-btn">Train More</button>
                                </div>
                            </div>
                            <div class="stash-copilot-train-recs" style="display: none;">
                                <div class="stash-copilot-train-recs-header">
                                    <h4>Your Recommendations</h4>
                                    <button class="stash-copilot-train-recs-refresh" title="Refresh recommendations">&#x21bb; Refresh</button>
                                </div>
                                <div class="stash-copilot-train-recs-grid"></div>
                                <div class="stash-copilot-train-recs-loading" style="display: none;">
                                    <div class="stash-copilot-spinner"></div>
                                    <span>Loading recommendations...</span>
                                </div>
                                <div class="stash-copilot-train-recs-empty" style="display: none;">
                                    <p>Complete at least one training session to see recommendations here.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Tag Gaps Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'tag_gaps' ? 'active' : ''}" data-tab="tag_gaps">
                        <div class="stash-copilot-tag-gaps-container">
                            <div class="stash-copilot-tag-gaps-controls">
                                <button class="btn btn-primary stash-copilot-tag-gaps-detect-btn">Detect Tag Gaps</button>
                                <label class="stash-copilot-tag-gaps-force-label" title="Recompute all scenes, ignoring cached results">
                                    <input type="checkbox" class="stash-copilot-tag-gaps-force-check" /> Force recompute
                                </label>
                                <span class="stash-copilot-tag-gaps-status"></span>
                            </div>
                            <div class="stash-copilot-tag-gaps-progress" style="display:none">
                                <div class="stash-copilot-tag-gaps-progress-bar-container">
                                    <div class="stash-copilot-tag-gaps-progress-bar" style="width:0%"></div>
                                </div>
                                <span class="stash-copilot-tag-gaps-progress-text"></span>
                            </div>
                            <div class="stash-copilot-tag-gaps-summary" style="display:none">
                                <div class="stash-copilot-tag-gaps-stats"></div>
                            </div>
                            <div class="stash-copilot-tag-gaps-results" style="display:none">
                                <div class="stash-copilot-tag-gaps-scene-list"></div>
                            </div>
                            <div class="stash-copilot-tag-gaps-empty">
                                <div class="stash-copilot-empty-state">
                                    <div class="stash-copilot-empty-state-icon">🏷️</div>
                                    <div class="stash-copilot-empty-state-desc">
                                        <h4>Tag Gap Detection</h4>
                                        <p>Detect visual content in your scenes that isn't covered by any existing tag. Uses frame-level embeddings to find gaps in your tag library.</p>
                                    </div>
                                    <div class="stash-copilot-empty-state-tip">
                                        <p>Click <strong>Detect Tag Gaps</strong> to analyze your library. Requires scenes to be embedded first.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        return modal;
    }

    function openInsightsModal() {
        if (state.insightsModalOpen) return;

        const modal = createInsightsModal();

        // Force reflow before adding open class (ensures CSS transition plays)
        modal.offsetHeight;

        // Open with animation
        requestAnimationFrame(() => {
            modal.classList.add('open');
        });

        state.insightsModalOpen = true;
        document.body.style.overflow = 'hidden';

        // Update navbar button state
        const navBtn = document.getElementById('stash-copilot-nav-btn');
        if (navBtn) navBtn.classList.add('active');

        // Setup event listeners
        setupInsightsModalEvents(modal);

        // Load data for active tab
        loadInsightsData(modal);
    }

    function closeInsightsModal() {
        const modal = document.getElementById('stash-copilot-insights-modal');
        if (!modal) return;

        // Add closing class for exit animation
        modal.classList.remove('open');
        modal.classList.add('closing');

        // Update navbar button state
        const navBtn = document.getElementById('stash-copilot-nav-btn');
        if (navBtn) navBtn.classList.remove('active');

        // Remove after animation completes
        setTimeout(() => {
            modal.remove();
            state.insightsModalOpen = false;
            document.body.style.overflow = '';
        }, 350);
    }

    // ============================================================================
    // Vision Details Modal
    // ============================================================================

    /**
     * Open the Vision Details modal for a scene
     */
    async function openVisionDetailsModal(sceneId) {
        // Fetch vision history data
        try {
            const response = await fetch(`${SCENE_VISION_PATH}/vision_history_${sceneId}.json?t=${Date.now()}`);
            if (!response.ok) {
                log(`No vision history found for scene ${sceneId}`, 'warn');
                // No analysis data yet - button will be disabled or show empty state
                return;
            }
            const data = await response.json();
            createVisionDetailsModal(sceneId, data);
        } catch (error) {
            log(`Failed to load vision details: ${error.message}`, 'error');
        }
    }

    /**
     * Create and display the Vision Details modal
     */
    function createVisionDetailsModal(sceneId, data) {
        // Remove existing modal if present
        const existing = document.getElementById('stash-copilot-vision-details-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'stash-copilot-vision-details-modal';
        modal.className = 'stash-copilot-vision-details-modal';

        modal.innerHTML = `
            <div class="stash-copilot-vision-details-content">
                <div class="stash-copilot-vision-details-header">
                    <h3>Analysis Details</h3>
                    <button class="stash-copilot-vision-details-close">&times;</button>
                </div>
                <div class="stash-copilot-vision-details-tabs">
                    <button class="stash-copilot-vision-details-tab active" data-tab="tools">Tool Calls</button>
                    <button class="stash-copilot-vision-details-tab" data-tab="frames">Frames</button>
                    <button class="stash-copilot-vision-details-tab" data-tab="debug">Debug</button>
                </div>
                <div class="stash-copilot-vision-details-body">
                    <div class="stash-copilot-vision-details-panel active" data-tab="tools">
                        ${renderToolCallsTab(data)}
                    </div>
                    <div class="stash-copilot-vision-details-panel" data-tab="frames">
                        ${renderFramesTab(data)}
                    </div>
                    <div class="stash-copilot-vision-details-panel" data-tab="debug">
                        ${renderDebugTab(data)}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Force reflow before opening animation
        modal.offsetHeight;
        requestAnimationFrame(() => {
            modal.classList.add('open');
        });

        // Setup event listeners
        setupVisionDetailsModalEvents(modal);
    }

    /**
     * Render the Tool Calls tab content
     */
    function renderToolCallsTab(data) {
        const toolCalls = data.tool_calls || [];

        if (toolCalls.length === 0) {
            return `
                <div class="stash-copilot-vision-details-empty">
                    <span class="stash-copilot-vision-details-empty-icon">🔧</span>
                    <p>No tool calls were made during this analysis.</p>
                    <p class="stash-copilot-vision-details-empty-hint">
                        Tool calls occur when the VLM uses timestamp lookup or frame search functions.
                        This requires a model that supports function calling.
                    </p>
                </div>
            `;
        }

        const toolCallsHtml = toolCalls.map((tc, index) => {
            const successIcon = tc.success ? '✓' : '✗';
            const successClass = tc.success ? 'success' : 'failure';
            const timestamp = tc.timestamp ? new Date(tc.timestamp).toLocaleTimeString() : '';

            return `
                <div class="stash-copilot-tool-call-card ${successClass}">
                    <div class="stash-copilot-tool-call-header">
                        <span class="stash-copilot-tool-call-name">🔧 ${escapeHtmlChars(tc.tool_name)}</span>
                        <span class="stash-copilot-tool-call-status ${successClass}">${successIcon}</span>
                    </div>
                    <div class="stash-copilot-tool-call-details">
                        <div class="stash-copilot-tool-call-row">
                            <span class="stash-copilot-tool-call-label">Arguments:</span>
                            <code class="stash-copilot-tool-call-value">${escapeHtmlChars(JSON.stringify(tc.arguments || {}))}</code>
                        </div>
                        <div class="stash-copilot-tool-call-row">
                            <span class="stash-copilot-tool-call-label">Result:</span>
                            <code class="stash-copilot-tool-call-value">${escapeHtmlChars(JSON.stringify(tc.result || {}))}</code>
                        </div>
                        <div class="stash-copilot-tool-call-meta">
                            <span>Stage: ${escapeHtmlChars(tc.stage || 'unknown')}</span>
                            ${timestamp ? `<span>Time: ${timestamp}</span>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        return `
            <div class="stash-copilot-tool-calls-summary">
                <span class="stash-copilot-tool-calls-count">${toolCalls.length} tool call${toolCalls.length !== 1 ? 's' : ''}</span>
                <span class="stash-copilot-tool-calls-success">${toolCalls.filter(tc => tc.success).length} successful</span>
            </div>
            <div class="stash-copilot-tool-calls-list">
                ${toolCallsHtml}
            </div>
        `;
    }

    /**
     * Render the Frames tab content
     */
    function renderFramesTab(data) {
        const frameTimestamps = data.frame_timestamps || [];
        const frameSelections = data.frame_selections || [];
        const frameCount = frameTimestamps.length;
        const resolution = data.frame_resolution || 'Unknown';
        const selectionMethod = data.frame_selection_method || 'Unknown';
        const usingGrid = data.using_grid_mode || false;
        const totalFrames = data.total_frames || frameCount;

        // Build frame timeline with selection reason colors
        let timelineHtml = '';
        if (frameTimestamps.length > 0) {
            const maxTimestamp = Math.max(...frameTimestamps);
            timelineHtml = frameTimestamps.map((ts, index) => {
                const position = maxTimestamp > 0 ? (ts / maxTimestamp) * 100 : 0;
                const formattedTime = formatSecondsToTime(ts);
                // Get selection reason if available
                const selection = frameSelections[index];
                const reason = selection?.selection_reason || 'temporal';
                const novelty = selection?.novelty_score;
                const reasonClass = reason === 'novelty' ? 'novelty' : 'temporal';
                const tooltip = novelty !== undefined
                    ? `Frame ${index + 1}: ${formattedTime} (${reason}, novelty: ${(novelty * 100).toFixed(1)}%)`
                    : `Frame ${index + 1}: ${formattedTime}`;
                return `
                    <div class="stash-copilot-frame-marker ${reasonClass}"
                         style="left: ${position}%"
                         title="${tooltip}">
                        <span class="stash-copilot-frame-marker-dot"></span>
                        <span class="stash-copilot-frame-marker-label">${index + 1}</span>
                    </div>
                `;
            }).join('');
        }

        // Build frame selections section (collapsible)
        let frameSelectionsHtml = '';
        if (frameSelections.length > 0) {
            const temporalCount = frameSelections.filter(s => s.selection_reason === 'temporal').length;
            const noveltyCount = frameSelections.filter(s => s.selection_reason === 'novelty').length;

            frameSelectionsHtml = `
                <details class="stash-copilot-frame-selections-details">
                    <summary>
                        <span>Frame Selection Details</span>
                        <span class="stash-copilot-frame-selections-summary">
                            <span class="temporal">${temporalCount} temporal</span>
                            <span class="novelty">${noveltyCount} novelty</span>
                        </span>
                    </summary>
                    <div class="stash-copilot-frame-selections-content">
                        <div class="stash-copilot-frame-selections-legend">
                            <span class="temporal-legend"><span class="dot"></span> Temporal</span>
                            <span class="novelty-legend"><span class="dot"></span> Novelty</span>
                        </div>
                        <div class="stash-copilot-frame-selections-list">
                            ${frameSelections.map((sel, i) => {
                                const reasonClass = sel.selection_reason === 'novelty' ? 'novelty' : 'temporal';
                                const imagePath = sel.path ? `/plugin/stash-copilot/assets/${sel.path}` : '';
                                const noveltyText = sel.selection_reason === 'novelty' && sel.novelty_score !== undefined
                                    ? `${(sel.novelty_score * 100).toFixed(0)}%`
                                    : '';
                                return `
                                    <div class="stash-copilot-frame-selection-item ${reasonClass}">
                                        <div class="frame-thumb">
                                            ${imagePath ? `<img src="${imagePath}" alt="Frame ${i + 1}" loading="lazy" />` : ''}
                                        </div>
                                        <div class="frame-caption">
                                            <span class="frame-time">${formatSecondsToTime(sel.timestamp)}</span>
                                            ${noveltyText ? `<span class="frame-novelty">${noveltyText}</span>` : ''}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                </details>
            `;
        }

        return `
            <div class="stash-copilot-frames-info">
                <div class="stash-copilot-frames-stats">
                    <div class="stash-copilot-frames-stat">
                        <span class="stash-copilot-frames-stat-value">${frameCount}</span>
                        <span class="stash-copilot-frames-stat-label">Frames Extracted</span>
                    </div>
                    <div class="stash-copilot-frames-stat">
                        <span class="stash-copilot-frames-stat-value">${resolution}px</span>
                        <span class="stash-copilot-frames-stat-label">Resolution</span>
                    </div>
                    <div class="stash-copilot-frames-stat">
                        <span class="stash-copilot-frames-stat-value">${selectionMethod}</span>
                        <span class="stash-copilot-frames-stat-label">Selection Method</span>
                    </div>
                    <div class="stash-copilot-frames-stat">
                        <span class="stash-copilot-frames-stat-value">${usingGrid ? 'Yes' : 'No'}</span>
                        <span class="stash-copilot-frames-stat-label">Grid Mode</span>
                    </div>
                </div>
                ${frameTimestamps.length > 0 ? `
                    <div class="stash-copilot-frames-timeline-container">
                        <h4>Frame Timeline</h4>
                        <div class="stash-copilot-frames-timeline">
                            <div class="stash-copilot-frames-timeline-track">
                                ${timelineHtml}
                            </div>
                        </div>
                        <div class="stash-copilot-frames-timestamps">
                            ${frameTimestamps.map((ts, i) => `
                                <span class="stash-copilot-frame-timestamp">
                                    <strong>${i + 1}:</strong> ${formatSecondsToTime(ts)}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                ` : `
                    <div class="stash-copilot-vision-details-empty">
                        <p>No frame timestamp data available.</p>
                    </div>
                `}
                ${frameSelectionsHtml}
            </div>
        `;
    }

    /**
     * Render the Debug tab content
     */
    function renderDebugTab(data) {
        const debugInfo = data.debug_info;
        const models = {
            description: data.description_model || 'Unknown',
            tag: data.tag_model || 'Unknown',
        };

        if (!debugInfo) {
            return `
                <div class="stash-copilot-vision-details-empty">
                    <span class="stash-copilot-vision-details-empty-icon">🐛</span>
                    <p>No debug information available.</p>
                    <p class="stash-copilot-vision-details-empty-hint">
                        Debug info is captured when debug mode is enabled in plugin settings.
                    </p>
                </div>
            `;
        }

        // Determine if this is multistage mode (has classification data)
        const isMultistage = !!(debugInfo.classification_system_prompt || debugInfo.classification_duration_ms);
        // Determine if verification stage exists
        const hasVerification = !!(debugInfo.verification_system_prompt || debugInfo.verification_duration_ms);

        // Calculate stage numbers dynamically
        let stageNum = 1;
        const classificationStage = isMultistage ? stageNum++ : null;
        const descriptionStage = stageNum++;
        const verificationStage = hasVerification ? stageNum++ : null;
        const taggingStage = stageNum;

        return `
            <div class="stash-copilot-debug-info">
                <div class="stash-copilot-debug-section">
                    <h4>Models Used</h4>
                    <div class="stash-copilot-debug-grid">
                        <div class="stash-copilot-debug-item">
                            <span class="label">Description (VLM):</span>
                            <span class="value">${escapeHtmlChars(models.description)}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Tagging (LLM):</span>
                            <span class="value">${escapeHtmlChars(models.tag)}</span>
                        </div>
                    </div>
                </div>

                ${isMultistage ? `
                <div class="stash-copilot-debug-section">
                    <h4>Stage ${classificationStage}: Classification</h4>
                    <div class="stash-copilot-debug-grid">
                        <div class="stash-copilot-debug-item">
                            <span class="label">Prompt Tokens:</span>
                            <span class="value">${debugInfo.classification_prompt_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Response Tokens:</span>
                            <span class="value">${debugInfo.classification_response_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Duration:</span>
                            <span class="value">${formatDurationMs(debugInfo.classification_duration_ms || 0)}</span>
                        </div>
                    </div>
                    ${debugInfo.classification_system_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>System Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.classification_system_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.classification_user_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>User Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.classification_user_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.classification_response ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>Response</summary>
                            <pre>${escapeHtmlChars(debugInfo.classification_response)}</pre>
                        </details>
                    ` : ''}
                </div>
                ` : ''}

                <div class="stash-copilot-debug-section">
                    <h4>Stage ${descriptionStage}: Description</h4>
                    <div class="stash-copilot-debug-grid">
                        <div class="stash-copilot-debug-item">
                            <span class="label">Frame Count:</span>
                            <span class="value">${debugInfo.description_frame_count || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Total Frame Bytes:</span>
                            <span class="value">${formatBytes(debugInfo.description_total_frame_bytes || 0)}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Prompt Tokens:</span>
                            <span class="value">${debugInfo.description_prompt_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Response Tokens:</span>
                            <span class="value">${debugInfo.description_response_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Duration:</span>
                            <span class="value">${formatDurationMs(debugInfo.description_duration_ms || 0)}</span>
                        </div>
                    </div>
                    ${debugInfo.description_system_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>System Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.description_system_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.description_user_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>User Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.description_user_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.description_response ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>Response</summary>
                            <pre>${escapeHtmlChars(debugInfo.description_response)}</pre>
                        </details>
                    ` : ''}
                </div>

                ${hasVerification ? `
                <div class="stash-copilot-debug-section">
                    <h4>Stage ${verificationStage}: Verification</h4>
                    <div class="stash-copilot-debug-grid">
                        <div class="stash-copilot-debug-item">
                            <span class="label">Prompt Tokens:</span>
                            <span class="value">${debugInfo.verification_prompt_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Response Tokens:</span>
                            <span class="value">${debugInfo.verification_response_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Duration:</span>
                            <span class="value">${formatDurationMs(debugInfo.verification_duration_ms || 0)}</span>
                        </div>
                    </div>
                    ${debugInfo.verification_system_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>System Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.verification_system_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.verification_user_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>User Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.verification_user_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.verification_response ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>Response</summary>
                            <pre>${escapeHtmlChars(debugInfo.verification_response)}</pre>
                        </details>
                    ` : ''}
                </div>
                ` : ''}

                <div class="stash-copilot-debug-section">
                    <h4>Stage ${taggingStage}: Tagging</h4>
                    <div class="stash-copilot-debug-grid">
                        <div class="stash-copilot-debug-item">
                            <span class="label">Prompt Tokens:</span>
                            <span class="value">${debugInfo.tag_prompt_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Response Tokens:</span>
                            <span class="value">${debugInfo.tag_response_tokens || 0}</span>
                        </div>
                        <div class="stash-copilot-debug-item">
                            <span class="label">Duration:</span>
                            <span class="value">${formatDurationMs(debugInfo.tag_duration_ms || 0)}</span>
                        </div>
                    </div>
                    ${debugInfo.tag_system_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>System Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.tag_system_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.tag_user_prompt ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>User Prompt</summary>
                            <pre>${escapeHtmlChars(debugInfo.tag_user_prompt)}</pre>
                        </details>
                    ` : ''}
                    ${debugInfo.tag_response ? `
                        <details class="stash-copilot-debug-expandable">
                            <summary>Response</summary>
                            <pre>${escapeHtmlChars(debugInfo.tag_response)}</pre>
                        </details>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Format seconds to MM:SS or HH:MM:SS
     */
    function formatSecondsToTime(seconds) {
        if (!seconds && seconds !== 0) return '--:--';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Format bytes to human readable
     */
    function formatBytes(bytes) {
        if (!bytes || bytes < 0) return '0 B';
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1);
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Format milliseconds to human readable duration
     */
    function formatDurationMs(ms) {
        if (!ms && ms !== 0) return '0ms';
        if (ms < 1000) return `${ms}ms`;
        const secs = ms / 1000;
        if (secs < 60) return `${secs.toFixed(1)}s`;
        const mins = secs / 60;
        return `${mins.toFixed(1)}m`;
    }

    /**
     * Escape HTML characters
     */
    function escapeHtmlChars(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Setup event listeners for Vision Details modal
     */
    function setupVisionDetailsModalEvents(modal) {
        // Close button
        modal.querySelector('.stash-copilot-vision-details-close').addEventListener('click', closeVisionDetailsModal);

        // Backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeVisionDetailsModal();
        });

        // Escape key
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                closeVisionDetailsModal();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);

        // Tab switching
        modal.querySelectorAll('.stash-copilot-vision-details-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;

                // Update tab buttons
                modal.querySelectorAll('.stash-copilot-vision-details-tab').forEach(t => {
                    t.classList.toggle('active', t.dataset.tab === tabName);
                });

                // Update panels
                modal.querySelectorAll('.stash-copilot-vision-details-panel').forEach(p => {
                    p.classList.toggle('active', p.dataset.tab === tabName);
                });
            });
        });
    }

    /**
     * Close the Vision Details modal
     */
    function closeVisionDetailsModal() {
        const modal = document.getElementById('stash-copilot-vision-details-modal');
        if (!modal) return;

        modal.classList.remove('open');
        modal.classList.add('closing');

        setTimeout(() => {
            modal.remove();
        }, 300);
    }

    // ============================================================================
    // End Vision Details Modal
    // ============================================================================

    function setupInsightsModalEvents(modal) {
        // Close button
        modal.querySelector('.stash-copilot-insights-close').addEventListener('click', closeInsightsModal);

        // Backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeInsightsModal();
        });

        // Escape key
        const escHandler = (e) => {
            if (e.key === 'Escape' && state.insightsModalOpen) {
                closeInsightsModal();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);

        // Tab switching
        modal.querySelectorAll('.stash-copilot-insights-tab').forEach(tab => {
            tab.addEventListener('click', () => switchInsightsTab(modal, tab.dataset.tab));
        });

        // Generate Summary button
        const generateBtn = modal.querySelector('.stash-copilot-generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', async () => {
                if (state.isGenerating) return;
                state.isGenerating = true;
                generateBtn.disabled = true;
                generateBtn.textContent = 'Generating...';

                try {
                    await runPluginTask('Generate Summary', {});
                    state.generationStartTime = Date.now();
                    pollForSummary(modal);
                } catch (error) {
                    log('Failed to start summary generation: ' + error.message, 'error');
                    state.isGenerating = false;
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate Summary';
                }
            });
        }

        // Refresh button
        const refreshBtn = modal.querySelector('.stash-copilot-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => loadInsightsData(modal));
        }

        // Chat send button
        const chatSendBtn = modal.querySelector('.stash-copilot-chat-send');
        const chatInput = modal.querySelector('.stash-copilot-chat-input');
        if (chatSendBtn && chatInput) {
            chatSendBtn.addEventListener('click', () => {
                const message = chatInput.value.trim();
                if (message) {
                    sendChatMessage(modal, message);
                    chatInput.value = '';
                }
            });

            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    chatSendBtn.click();
                }
            });
        }

        // Clear chat button
        const clearChatBtn = modal.querySelector('.stash-copilot-clear-chat');
        if (clearChatBtn) {
            clearChatBtn.addEventListener('click', () => clearChatHistory(modal));
        }

        // Recommendations generate button
        const recGenerateBtn = modal.querySelector('.stash-copilot-rec-generate-btn');
        if (recGenerateBtn) {
            recGenerateBtn.addEventListener('click', () => generateRecommendations(modal));
        }

        // Recommendations view filter switching (All / New / Re-watch)
        const viewFilterBtns = modal.querySelectorAll('.stash-copilot-rec-viewfilter');
        viewFilterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const filter = btn.dataset.filter;
                if (!filter) return;

                // Update active state
                viewFilterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Update state
                state.recommendationsViewFilter = filter;
                state.recommendationsPage = 1;

                // Apply filter and re-render if we have results
                if (state.recommendationsMergedResults && state.recommendationsMergedResults.length > 0) {
                    applyModalRecsViewFilter();
                    renderRecommendationsResults(modal, {
                        results: state.recommendationsResults,
                        profile: state.recommendationsProfile,
                        generatedAt: state.recommendationsGeneratedAt
                    });
                }
            });
        });

        // Recommendations decay select
        const decaySelect = modal.querySelector('.stash-copilot-rec-decay-select');
        if (decaySelect) {
            decaySelect.addEventListener('change', (e) => {
                const value = e.target.value;
                state.recommendationsTimeDecayDays = value === 'session' ? 'session' : parseInt(value);
                localStorage.setItem('stash-copilot-time-decay-days', value);

                // Toggle clear session button visibility
                const clearSessionBtn = modal.querySelector('.stash-copilot-rec-clear-session-btn');
                if (clearSessionBtn) {
                    clearSessionBtn.style.display = value === 'session' ? 'inline-flex' : 'none';
                }
            });
        }

        // Clear session button
        const clearSessionBtn = modal.querySelector('.stash-copilot-rec-clear-session-btn');
        if (clearSessionBtn) {
            clearSessionBtn.addEventListener('click', () => {
                clearSessionScenes();
                // Update the session option text
                const sessionOption = decaySelect?.querySelector('option[value="session"]');
                if (sessionOption) {
                    sessionOption.textContent = `This Session (0)`;
                }
            });
        }

        // Peak generate button
        const peakGenerateBtn = modal.querySelector('.stash-copilot-peak-generate-btn');
        if (peakGenerateBtn) {
            peakGenerateBtn.addEventListener('click', () => generatePeakMoments(modal));
        }

        // Peak embed button
        const peakEmbedBtn = modal.querySelector('.stash-copilot-peak-embed-btn');
        if (peakEmbedBtn) {
            peakEmbedBtn.addEventListener('click', async () => {
                peakEmbedBtn.disabled = true;
                peakEmbedBtn.textContent = 'Starting...';
                try {
                    await runPluginTask('Embed O-Moments', {});
                    peakEmbedBtn.textContent = 'Running...';
                    // Poll for completion
                    setTimeout(() => {
                        peakEmbedBtn.disabled = false;
                        peakEmbedBtn.textContent = '🎬 Embed Markers';
                        loadPeakStats(modal);
                    }, 5000);
                } catch (error) {
                    log('Failed to start O-Moment embedding: ' + error.message, 'error');
                    peakEmbedBtn.disabled = false;
                    peakEmbedBtn.textContent = '🎬 Embed Markers';
                }
            });
        }
    }

    function switchInsightsTab(modal, tabName) {
        setActiveTab(tabName);
        modal.setAttribute('data-active-tab', tabName);

        // Update tab buttons
        modal.querySelectorAll('.stash-copilot-insights-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update panels
        modal.querySelectorAll('.stash-copilot-insights-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.tab === tabName);
        });

        // Clear data-empty for tabs that don't use it
        if (tabName === 'summary' || tabName === 'chat' || tabName === 'tools' || tabName === 'taste_map' || tabName === 'train' || tabName === 'tag_gaps') {
            modal.removeAttribute('data-empty');
        }

        // Load tab-specific data (recommendations and peak handle data-empty internally)
        if (tabName === 'chat') loadChatHistory(modal);
        if (tabName === 'tools') renderToolsList(modal);
        if (tabName === 'recommendations') loadRecommendationsResults(modal);
        if (tabName === 'peak') loadPeakStats(modal);
        if (tabName === 'taste_map') loadTasteMapData(modal);
        if (tabName === 'train') loadTrainData(modal);
        if (tabName === 'tag_gaps') loadTagGapsData(modal);
    }

    function loadInsightsData(modal) {
        // Load stats and summary for Summary tab
        loadDropdownData(modal);

        // Load data based on active tab
        const activeTab = getActiveTab();
        if (activeTab === 'chat') loadChatHistory(modal);
        if (activeTab === 'tools') renderToolsList(modal);
        if (activeTab === 'recommendations') loadRecommendationsResults(modal);
        if (activeTab === 'peak') loadPeakStats(modal);
        if (activeTab === 'taste_map') loadTasteMapData(modal);
        if (activeTab === 'train') loadTrainData(modal);
        if (activeTab === 'tag_gaps') loadTagGapsData(modal);
    }

    // ====================================================================
    // Taste Map Functions
    // ====================================================================

    const CLUSTER_COLORS = [
        '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'
    ];

    function hexToRgb(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `${r}, ${g}, ${b}`;
    }

    function lerp3(a, b, t) {
        return {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            z: a.z + (b.z - a.z) * t,
        };
    }

    const DEFAULT_CAMERA = {
        eye: { x: 1.5, y: 1.5, z: 1.2 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 },
    };

    /**
     * Convert data coordinates (e.g. UMAP values) to Plotly's normalized
     * camera coordinate space where (0,0,0) is the data bounding-box center
     * and axes are scaled by the computed aspect ratio.
     */
    function dataToCameraCoords(chartEl, dataPoint) {
        const sl = chartEl._fullLayout.scene;
        const xr = sl.xaxis.range;
        const yr = sl.yaxis.range;
        const zr = sl.zaxis.range;
        const aspect = sl.aspectratio || { x: 1, y: 1, z: 1 };

        const mx = (xr[0] + xr[1]) / 2;
        const my = (yr[0] + yr[1]) / 2;
        const mz = (zr[0] + zr[1]) / 2;
        const rx = xr[1] - xr[0];
        const ry = yr[1] - yr[0];
        const rz = zr[1] - zr[0];

        return {
            x: ((dataPoint.x - mx) / rx) * aspect.x,
            y: ((dataPoint.y - my) / ry) * aspect.y,
            z: ((dataPoint.z - mz) / rz) * aspect.z,
        };
    }

    function animateCameraToTarget(chartEl, endCenter, endEye, duration, zoomOutBulge) {
        // Cancel any in-progress animation
        if (chartEl._cameraAnimId) {
            cancelAnimationFrame(chartEl._cameraAnimId);
            chartEl._cameraAnimId = null;
        }

        const bulge = zoomOutBulge || 0;
        const scene = chartEl._fullLayout.scene._scene;
        const camera = scene.getCamera();
        const startCenter = { ...camera.center };
        const startEye = { ...camera.eye };

        const startTime = performance.now();
        function step(now) {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3); // cubic ease-out

            const center = lerp3(startCenter, endCenter, ease);
            const baseEye = lerp3(startEye, endEye, ease);

            let eye = baseEye;
            if (bulge > 0) {
                // Push eye away from center with a sine curve peaking at t=0.5
                const b = Math.sin(Math.PI * t) * bulge;
                const dx = baseEye.x - center.x;
                const dy = baseEye.y - center.y;
                const dz = baseEye.z - center.z;
                eye = {
                    x: center.x + dx * (1 + b),
                    y: center.y + dy * (1 + b),
                    z: center.z + dz * (1 + b),
                };
            }

            Plotly.relayout(chartEl, {
                'scene.camera.center': center,
                'scene.camera.eye': eye,
            });

            if (t < 1) {
                chartEl._cameraAnimId = requestAnimationFrame(step);
            } else {
                chartEl._cameraAnimId = null;
            }
        }
        chartEl._cameraAnimId = requestAnimationFrame(step);
    }

    async function buildTasteMap(modal) {
        if (state.tasteMapLoading) return;

        state.tasteMapLoading = true;
        const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
        const statusEl = modal.querySelector('.stash-copilot-taste-map-status');

        buildBtn.disabled = true;
        buildBtn.innerHTML = '<span class="stash-copilot-spinner"></span>';
        statusEl.textContent = 'Building taste map...';

        const requestId = `taste_map_${Date.now()}`;
        state.tasteMapRequestId = requestId;

        try {
            const taskArgs = { request_id: requestId };
            const kInput = modal.querySelector('.stash-copilot-taste-map-k-input');
            const kValue = kInput ? parseInt(kInput.value) : NaN;
            if (!isNaN(kValue) && kValue >= 2) {
                taskArgs.num_clusters = kValue;
            }
            await runPluginTask('Build Taste Map', taskArgs);
            pollTasteMapResults(modal, requestId);
        } catch (e) {
            log(`Build Taste Map error: ${e.message}`, 'error');
            state.tasteMapLoading = false;
            buildBtn.disabled = false;
            buildBtn.textContent = 'Build Taste Map';
            statusEl.textContent = `Error: ${e.message}`;
        }
    }

    function pollTasteMapResults(modal, requestId) {
        const resultFile = `/plugin/stash-copilot/assets/taste_map_${requestId}.json`;

        const interval = setInterval(async () => {
            if (state.tasteMapRequestId !== requestId) {
                clearInterval(interval);
                return;
            }

            try {
                const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        clearInterval(interval);
                        state.tasteMapData = data;
                        state.tasteMapLoading = false;
                        renderTasteMap(modal, data);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        state.tasteMapLoading = false;
                        const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
                        const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
                        if (buildBtn) { buildBtn.disabled = false; buildBtn.textContent = 'Build Taste Map'; }
                        if (statusEl) statusEl.textContent = `Error: ${data.error}`;
                    }
                }
            } catch (e) {
                // 404 expected while task is running
            }
        }, 500);

        setTimeout(() => {
            clearInterval(interval);
            if (state.tasteMapLoading) {
                state.tasteMapLoading = false;
                const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
                if (statusEl) statusEl.textContent = 'Timed out waiting for results';
            }
        }, 900000); // 15 min timeout (12K+ scene queries take ~10 min)
    }

    async function renderTasteMap(modal, data) {
        const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
        const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
        const emptyEl = modal.querySelector('.stash-copilot-taste-map-empty');
        const contentEl = modal.querySelector('.stash-copilot-taste-map-content');

        if (buildBtn) { buildBtn.disabled = false; buildBtn.textContent = 'Rebuild'; }
        if (statusEl) statusEl.textContent = `${data.clusters.length} clusters, ${data.scenes.length} scenes (silhouette: ${data.silhouette_score.toFixed(2)})`;

        if (emptyEl) emptyEl.style.display = 'none';
        if (contentEl) contentEl.style.display = 'flex';

        // Render scatter plot
        await loadPlotly();
        renderTasteMapChart(modal, data);

        // Render cluster sidebar
        renderClusterSidebar(modal, data);
    }

    function renderTasteMapChart(modal, data) {
        const chartContainer = modal.querySelector('#taste-map-chart');
        if (!chartContainer) return;

        // Purge old chart if exists
        if (state.tasteMapChart) {
            if (state.tasteMapChart._cameraAnimId) {
                cancelAnimationFrame(state.tasteMapChart._cameraAnimId);
                state.tasteMapChart._cameraAnimId = null;
            }
            if (state.tasteMapChart._tmMouseCleanup) state.tasteMapChart._tmMouseCleanup();
            Plotly.purge(state.tasteMapChart);
        }

        state.tasteMapChart = chartContainer;

        // Build cluster label lookup
        const clusterLabels = {};
        for (const c of data.clusters) {
            clusterLabels[c.cluster_id] = c.auto_label;
        }

        // One trace per cluster — all scenes, no profile/non-profile distinction
        const traces = [];
        const traceClusterIds = [];
        for (const cluster of data.clusters) {
            const clusterScenes = data.scenes.filter(s => s.cluster_id === cluster.cluster_id);
            if (clusterScenes.length === 0) continue;

            const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
            const colorRgb = hexToRgb(color);

            traces.push({
                name: cluster.auto_label,
                type: 'scatter3d',
                mode: 'markers',
                x: clusterScenes.map(s => s.x),
                y: clusterScenes.map(s => s.y),
                z: clusterScenes.map(s => s.z || 0),
                customdata: clusterScenes.map(s => ({
                    id: s.scene_id,
                    title: s.title || `Scene ${s.scene_id}`,
                    plays: s.play_count || 0,
                    oCnt: s.o_counter || 0,
                    eng: s.engagement_score || 0,
                    clusterId: cluster.cluster_id,
                })),
                marker: {
                    size: clusterScenes.map(s => {
                        const eng = s.engagement_score || 0;
                        return Math.max(3, Math.min(12, 3 + Math.log(1 + eng) * 2));
                    }),
                    color: color,
                    opacity: clusterScenes.map(s => {
                        const eng = s.engagement_score || 0;
                        return eng > 0 ? 0.85 : 0.4;
                    }),
                    line: {
                        color: clusterScenes.map(s => {
                            const eng = s.engagement_score || 0;
                            const glowAlpha = Math.min(0.3, 0.05 + Math.log(1 + eng) * 0.04);
                            return `rgba(${colorRgb}, ${glowAlpha})`;
                        }),
                        width: clusterScenes.map(s => {
                            const eng = s.engagement_score || 0;
                            return Math.max(0, Math.min(4, Math.log(1 + eng) * 0.8));
                        }),
                    },
                },
                hoverinfo: 'none',
                hovertemplate: null,
            });
            traceClusterIds.push(cluster.cluster_id);
        }

        // Store trace-to-cluster mapping for sidebar interaction
        chartContainer._traceClusterIds = traceClusterIds;

        const layout = {
            paper_bgcolor: 'transparent',
            margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
                bgcolor: 'rgba(10, 10, 15, 0.5)',
                xaxis: { visible: false },
                yaxis: { visible: false },
                zaxis: { visible: false },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 },
                    center: { x: 0, y: 0, z: 0 },
                    up: { x: 0, y: 0, z: 1 },
                },
                dragmode: 'orbit',
            },
            showlegend: false,
            hovermode: 'closest',
        };

        const config = {
            responsive: true,
            scrollZoom: 'gl3d',
            displayModeBar: true,
            modeBarButtonsToRemove: [
                'toImage', 'orbitRotation', 'tableRotation',
                'zoom3d', 'pan3d', 'resetCameraLastSave3d',
            ],
            displaylogo: false,
        };

        Plotly.newPlot(chartContainer, traces, layout, config);

        // Lock axis ranges so camera relayout calls don't trigger autorange
        const computedScene = chartContainer._fullLayout.scene;
        Plotly.relayout(chartContainer, {
            'scene.xaxis.autorange': false,
            'scene.xaxis.range': [...computedScene.xaxis.range],
            'scene.yaxis.autorange': false,
            'scene.yaxis.range': [...computedScene.yaxis.range],
            'scene.zaxis.autorange': false,
            'scene.zaxis.range': [...computedScene.zaxis.range],
        });

        // Debug overlay — shows camera center and hovered point coords
        // Set to true to enable (useful for diagnosing camera/coordinate issues)
        const TASTE_MAP_DEBUG = false;
        let debugOverlay = null;
        if (TASTE_MAP_DEBUG) {
            debugOverlay = chartContainer.querySelector('.stash-copilot-tm-debug');
            if (!debugOverlay) {
                debugOverlay = document.createElement('div');
                debugOverlay.className = 'stash-copilot-tm-debug';
                debugOverlay.style.cssText = 'position:absolute;top:8px;left:8px;z-index:10;'
                    + 'background:rgba(0,0,0,0.75);color:#0f0;font:11px/1.4 monospace;'
                    + 'padding:6px 10px;border-radius:6px;pointer-events:none;white-space:pre;';
                chartContainer.style.position = 'relative';
                chartContainer.appendChild(debugOverlay);
            }
        }
        function updateDebug(hoverData) {
            if (!TASTE_MAP_DEBUG || !debugOverlay) return;
            const sceneObj = chartContainer._fullLayout.scene._scene;
            const cam = sceneObj.getCamera();
            const c = cam.center;
            const e = cam.eye;
            const sl = chartContainer._fullLayout.scene;
            const xr = sl.xaxis.range;
            const yr = sl.yaxis.range;
            const zr = sl.zaxis.range;
            const asp = sl.aspectratio || { x: 1, y: 1, z: 1 };

            let txt = `center  ${c.x.toFixed(3)}, ${c.y.toFixed(3)}, ${c.z.toFixed(3)}\n`;
            txt += `eye     ${e.x.toFixed(3)}, ${e.y.toFixed(3)}, ${e.z.toFixed(3)}\n`;
            txt += `xrange  [${xr[0].toFixed(2)}, ${xr[1].toFixed(2)}]\n`;
            txt += `yrange  [${yr[0].toFixed(2)}, ${yr[1].toFixed(2)}]\n`;
            txt += `zrange  [${zr[0].toFixed(2)}, ${zr[1].toFixed(2)}]\n`;
            txt += `aspect  ${asp.x.toFixed(3)}, ${asp.y.toFixed(3)}, ${asp.z.toFixed(3)}`;
            const gl = sceneObj.glplot;
            if (gl) {
                if (gl.dataScale) txt += `\ndScale  ${gl.dataScale.map(v => v.toFixed(4)).join(', ')}`;
                if (gl.dataCenter) txt += `\ndCenter ${gl.dataCenter.map(v => v.toFixed(3)).join(', ')}`;
            }
            if (hoverData) {
                const dc = hoverData.data;
                const cc = hoverData.cam;
                txt += `\nhover   data  ${dc.x.toFixed(3)}, ${dc.y.toFixed(3)}, ${dc.z.toFixed(3)}`;
                txt += `\n        cam   ${cc.x.toFixed(3)}, ${cc.y.toFixed(3)}, ${cc.z.toFixed(3)}`;
            }
            debugOverlay.textContent = txt;
        }
        if (TASTE_MAP_DEBUG) {
            const debugInterval = setInterval(() => {
                if (!document.contains(chartContainer)) { clearInterval(debugInterval); return; }
                updateDebug(chartContainer._lastHoverDebug || null);
            }, 200);
        }

        // Custom tooltip — unified card tooltip style, cursor-following
        let tooltip = document.querySelector('.stash-copilot-card-tooltip[data-theme="taste-map"]');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'stash-copilot-card-tooltip';
            tooltip.dataset.theme = 'taste-map';
            document.body.appendChild(tooltip);
        }

        // Track mouse globally (WebGL canvas swallows events)
        let mouseX = 0, mouseY = 0;
        function trackMouse(e) { mouseX = e.clientX; mouseY = e.clientY; }
        document.addEventListener('mousemove', trackMouse);
        const origPurge = chartContainer._tmMouseCleanup;
        chartContainer._tmMouseCleanup = () => {
            document.removeEventListener('mousemove', trackMouse);
            if (tooltip) tooltip.classList.remove('visible');
            if (origPurge) origPurge();
        };

        chartContainer.on('plotly_hover', function(eventData) {
            if (!eventData.points || !eventData.points.length) return;
            const pt = eventData.points[0];
            const d = pt.customdata;
            if (!d) return;

            // Update debug overlay with hovered point coords
            const hoverDataCoords = { x: pt.x, y: pt.y, z: pt.z };
            const hoverCamCoords = dataToCameraCoords(chartContainer, hoverDataCoords);
            chartContainer._lastHoverDebug = { data: hoverDataCoords, cam: hoverCamCoords };
            updateDebug(chartContainer._lastHoverDebug);

            const clusterLabel = clusterLabels[d.clusterId] || '';
            const clusterColor = CLUSTER_COLORS[d.clusterId % CLUSTER_COLORS.length];

            tooltip.innerHTML = `
                <img class="stash-copilot-card-tooltip-thumb" src="/scene/${d.id}/screenshot" alt="">
                <div class="stash-copilot-card-tooltip-header">
                    <div class="stash-copilot-card-tooltip-title">${escapeHtml(d.title)}</div>
                    ${clusterLabel ? `<div class="stash-copilot-card-tooltip-studio" style="color: ${clusterColor}">${escapeHtml(clusterLabel)}</div>` : ''}
                </div>
                <div class="stash-copilot-card-tooltip-stats">
                    <span class="${d.plays > 0 ? 'has-value' : ''}">▶ ${d.plays} plays</span>
                    <span class="${d.oCnt > 0 ? 'has-value' : ''}">💦 ${d.oCnt}</span>
                    <span class="stash-copilot-card-tooltip-score">Eng: ${d.eng.toFixed(1)}</span>
                </div>
            `;

            const tooltipWidth = 260;
            const margin = 12;
            let left = mouseX + margin;
            let top = mouseY - margin;

            if (left + tooltipWidth > window.innerWidth) {
                left = mouseX - tooltipWidth - margin;
            }
            if (top < margin) top = margin;
            if (top + 120 > window.innerHeight) {
                top = window.innerHeight - 120 - margin;
            }

            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
            tooltip.classList.add('visible');

            // Chart → sidebar: highlight corresponding cluster card
            const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');
            cards.forEach(card => {
                if (parseInt(card.dataset.clusterId) === d.clusterId) {
                    card.classList.add('chart-hover');
                } else {
                    card.classList.remove('chart-hover');
                }
            });
        });

        chartContainer.on('plotly_unhover', function() {
            tooltip.classList.remove('visible');
            chartContainer._lastHoverDebug = null;
            updateDebug(null);
            const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');
            cards.forEach(card => card.classList.remove('chart-hover'));
        });

        // Click point → re-center orbit + zoom in
        chartContainer.on('plotly_click', function(eventData) {
            if (!eventData.points || !eventData.points.length) return;
            const pt = eventData.points[0];

            // Convert data coordinates to Plotly's camera coordinate space
            const target = dataToCameraCoords(chartContainer, { x: pt.x, y: pt.y, z: pt.z });

            const scene = chartContainer._fullLayout.scene._scene;
            const camera = scene.getCamera();
            const dx = camera.eye.x - target.x;
            const dy = camera.eye.y - target.y;
            const dz = camera.eye.z - target.z;
            const zoomFactor = 0.45;
            const endEye = {
                x: target.x + dx * zoomFactor,
                y: target.y + dy * zoomFactor,
                z: target.z + dz * zoomFactor,
            };

            animateCameraToTarget(chartContainer, target, endEye, 500);
        });

        // Double-click empty space → reset to default overview
        chartContainer.on('plotly_doubleclick', function() {
            animateCameraToTarget(
                chartContainer,
                { ...DEFAULT_CAMERA.center },
                { ...DEFAULT_CAMERA.eye },
                600
            );
            return false; // suppress Plotly's default axis reset
        });

        // Resize observer
        if (chartContainer._resizeObserver) {
            chartContainer._resizeObserver.disconnect();
        }
        const resizeObserver = new ResizeObserver(() => {
            Plotly.Plots.resize(chartContainer);
        });
        resizeObserver.observe(chartContainer);
        chartContainer._resizeObserver = resizeObserver;
    }

    function renderClusterSidebar(modal, data) {
        const container = modal.querySelector('.stash-copilot-taste-map-clusters');
        if (!container) return;

        container.innerHTML = data.clusters.map(cluster => {
            const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
            const thumbs = (cluster.representative_scenes || []).map(sid => {
                const scene = data.scenes.find(s => s.scene_id === sid);
                const thumb = scene?.thumbnail || `/scene/${sid}/screenshot`;
                return `<img src="${thumb}" class="stash-copilot-taste-map-cluster-thumb" alt="" loading="lazy">`;
            }).join('');

            return `
                <div class="stash-copilot-taste-map-cluster-card"
                     data-cluster-id="${cluster.cluster_id}"
                     style="--cluster-color: ${color}; --cluster-color-rgb: ${hexToRgb(color)}">
                    <div class="stash-copilot-taste-map-cluster-header">
                        <span class="stash-copilot-taste-map-cluster-dot" style="background: ${color}"></span>
                        <span class="stash-copilot-taste-map-cluster-label">${escapeHtml(cluster.auto_label)}</span>
                        <span class="stash-copilot-taste-map-cluster-toggle">▼</span>
                    </div>
                    <div class="stash-copilot-taste-map-cluster-body">
                        <div class="stash-copilot-taste-map-cluster-stats">
                            <span>${cluster.scene_ids.length} scenes &middot; ${((cluster.engagement_share || 0) * 100).toFixed(0)}%</span>
                        </div>
                        <div class="stash-copilot-taste-map-cluster-thumbs">${thumbs}</div>
                    </div>
                </div>
            `;
        }).join('');

        // Setup sidebar card interactions
        setupClusterCardEvents(modal, data);
    }

    function setupClusterCardEvents(modal, data) {
        const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');

        cards.forEach(card => {
            const clusterId = parseInt(card.dataset.clusterId);
            const header = card.querySelector('.stash-copilot-taste-map-cluster-header');

            // Click header to collapse/expand + center camera on cluster
            header.addEventListener('click', () => {
                card.classList.toggle('collapsed');

                if (!state.tasteMapChart) return;
                const el = state.tasteMapChart;
                const clusterScenes = data.scenes.filter(s => s.cluster_id === clusterId);
                if (clusterScenes.length === 0) return;

                // Compute cluster centroid in data coordinates
                const cx = clusterScenes.reduce((s, sc) => s + sc.x, 0) / clusterScenes.length;
                const cy = clusterScenes.reduce((s, sc) => s + sc.y, 0) / clusterScenes.length;
                const cz = clusterScenes.reduce((s, sc) => s + (sc.z || 0), 0) / clusterScenes.length;
                const target = dataToCameraCoords(el, { x: cx, y: cy, z: cz });

                const scene = el._fullLayout.scene._scene;
                const camera = scene.getCamera();
                const dx = camera.eye.x - target.x;
                const dy = camera.eye.y - target.y;
                const dz = camera.eye.z - target.z;
                const zoomFactor = 0.6; // gentler zoom than single-point click
                const endEye = {
                    x: target.x + dx * zoomFactor,
                    y: target.y + dy * zoomFactor,
                    z: target.z + dz * zoomFactor,
                };
                animateCameraToTarget(el, target, endEye, 1000, 0.7);
            });

            // Hover card → desaturate other clusters in chart
            card.addEventListener('mouseenter', () => {
                if (!state.tasteMapChart) return;
                const el = state.tasteMapChart;
                const traceClusterIds = el._traceClusterIds || [];
                const traceCount = traceClusterIds.length;
                if (traceCount === 0) return;

                const indices = Array.from({ length: traceCount }, (_, i) => i);
                const colors = [];
                const opacityArrays = [];
                for (let i = 0; i < traceCount; i++) {
                    const tcId = traceClusterIds[i];
                    const clusterScenes = data.scenes.filter(s => s.cluster_id === tcId);
                    if (tcId === clusterId) {
                        colors.push(CLUSTER_COLORS[tcId % CLUSTER_COLORS.length]);
                        opacityArrays.push(clusterScenes.map(() => 0.9));
                    } else {
                        colors.push('rgba(150, 150, 150, 0.4)');
                        opacityArrays.push(clusterScenes.map(() => 0.15));
                    }
                }
                Plotly.restyle(el, { 'marker.color': colors, 'marker.opacity': opacityArrays }, indices);
            });

            // Mouse leave → restore all clusters
            card.addEventListener('mouseleave', () => {
                if (!state.tasteMapChart) return;
                const el = state.tasteMapChart;
                const traceClusterIds = el._traceClusterIds || [];
                const traceCount = traceClusterIds.length;
                if (traceCount === 0) return;

                const indices = Array.from({ length: traceCount }, (_, i) => i);
                const colors = [];
                const opacityArrays = [];
                for (let i = 0; i < traceCount; i++) {
                    const tcId = traceClusterIds[i];
                    colors.push(CLUSTER_COLORS[tcId % CLUSTER_COLORS.length]);
                    const clusterScenes = data.scenes.filter(s => s.cluster_id === tcId);
                    opacityArrays.push(clusterScenes.map(s => {
                        const eng = s.engagement_score || 0;
                        return eng > 0 ? 0.85 : 0.4;
                    }));
                }
                Plotly.restyle(el, { 'marker.color': colors, 'marker.opacity': opacityArrays }, indices);
            });
        });
    }

    function setupTasteMapEvents(modal) {
        const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
        if (buildBtn) {
            buildBtn.addEventListener('click', () => buildTasteMap(modal));
        }

        const sidebarHeader = modal.querySelector('.stash-copilot-taste-map-sidebar-header');
        if (sidebarHeader) {
            sidebarHeader.addEventListener('click', () => {
                const sidebar = modal.querySelector('.stash-copilot-taste-map-sidebar');
                sidebar.classList.toggle('collapsed');
                const toggle = sidebarHeader.querySelector('.stash-copilot-taste-map-sidebar-toggle');
                if (toggle) toggle.textContent = sidebar.classList.contains('collapsed') ? '▶' : '◀';
                if (state.tasteMapChart) {
                    setTimeout(() => Plotly.Plots.resize(state.tasteMapChart), 300);
                }
            });
        }
    }

    async function loadTasteMapData(modal) {
        // Setup events on first load
        if (!modal._tasteMapEventsSetup) {
            setupTasteMapEvents(modal);
            modal._tasteMapEventsSetup = true;
        }

        // In-memory data takes priority
        if (state.tasteMapData) {
            renderTasteMap(modal, state.tasteMapData);
            return;
        }

        // Attempt to load persisted taste map
        try {
            const resp = await fetch(
                `/plugin/stash-copilot/assets/taste_map_latest.json?t=${Date.now()}`,
                { cache: 'no-store' }
            );
            if (resp.ok) {
                const data = await resp.json();
                const has3D = data.scenes?.[0]?.z !== undefined;
                if (data.status === 'complete' && data.scenes?.length > 0 && has3D) {
                    state.tasteMapData = data;
                    renderTasteMap(modal, data);
                    return;
                }
            }
        } catch (e) {
            // No persisted data — show empty state
        }
    }

    function loadTrainData(modal) {
        const panel = modal.querySelector('.stash-copilot-insights-panel[data-tab="train"]');
        const container = panel ? panel.querySelector('.stash-copilot-train') : null;
        if (!container) return;

        if (!modal._trainEventsSetup) {
            setupTrainListeners(container);
            modal._trainEventsSetup = true;
        }

        // If there's an active session, restore the session view
        if (preferenceState.isTraining && preferenceState.pairs.length > 0) {
            container.querySelector('.stash-copilot-train-intro').style.display = 'none';
            container.querySelector('.stash-copilot-train-complete').style.display = 'none';
            const sessionEl = container.querySelector('.stash-copilot-train-session');
            sessionEl.style.display = '';
            updateTrainProgress(container);
            if (preferenceState.pairIndex < preferenceState.pairs.length) {
                renderTrainCard(container, preferenceState.pairs[preferenceState.pairIndex]);
                container.querySelector('.stash-copilot-train-actions').style.display = '';
            }
        } else {
            loadExistingTrainStats(container);
            loadPreferenceRecs(container);
        }
    }

    // Create navbar button (opens AI Insights modal)
    function createNavbarButton() {
        // Check if already exists
        if (document.getElementById('stash-copilot-nav-btn')) {
            return;
        }

        // Create the nav button
        const navBtn = document.createElement('button');
        navBtn.id = 'stash-copilot-nav-btn';
        navBtn.className = 'stash-copilot-nav-btn';
        navBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            <span>AI Insights</span>
        `;
        navBtn.title = 'AI Library Insights';

        // Open modal on button click
        navBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (state.insightsModalOpen) {
                closeInsightsModal();
            } else {
                openInsightsModal();
            }
        });

        // Find the always-visible navbar buttons (not the collapsible mobile menu)
        const navbarButtons = document.querySelector('.navbar-buttons.ml-auto');

        if (navbarButtons) {
            // Find the donate button within this section
            const donateBtn = navbarButtons.querySelector('a[href*="opencollective"]');

            if (donateBtn) {
                // Insert right before the donate button
                navbarButtons.insertBefore(navBtn, donateBtn);
                log('Navbar button added before donate');
            } else {
                // Fallback: insert at the beginning
                navbarButtons.insertBefore(navBtn, navbarButtons.firstChild);
                log('Navbar button added to navbar-buttons');
            }
        } else {
            log('Navbar not found, retrying...', 'warn');
            return;
        }
    }

    /**
     * Create and inject the semantic search button in Stash navbar (next to Tags)
     */
    function createSearchNavButton() {
        // Check if already exists
        if (document.getElementById('stash-copilot-search-nav-item')) {
            return;
        }

        // Find the Tags nav-item (li element containing the Tags link)
        const tagsLink = document.querySelector('.navbar-nav a[href="/tags"]');
        if (!tagsLink) {
            log('Tags link not found for search button placement', 'warn');
            return;
        }

        // Get the parent nav-item (li element)
        const tagsNavItem = tagsLink.closest('.nav-item') || tagsLink.parentElement;
        if (!tagsNavItem) {
            log('Tags nav-item not found', 'warn');
            return;
        }

        // Create nav-item wrapper (li element to match Stash structure)
        const navItem = document.createElement('li');
        navItem.className = 'nav-item';
        navItem.id = 'stash-copilot-search-nav-item';

        // Create search button as a nav-link
        const searchBtn = document.createElement('a');
        searchBtn.id = 'stash-copilot-search-nav-btn';
        searchBtn.className = 'stash-copilot-search-nav-btn nav-link';
        searchBtn.href = '/plugins/stash-copilot/search';
        searchBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35"/>
            </svg>
            <span>AI Search</span>
            <span class="stash-copilot-search-sparkle">✨</span>
        `;
        searchBtn.title = 'Semantic Search - Find scenes with natural language';

        // Add button to nav-item
        navItem.appendChild(searchBtn);

        // Insert the nav-item after Tags nav-item
        tagsNavItem.parentNode.insertBefore(navItem, tagsNavItem.nextSibling);

        // Handle click - prevent default and use SPA navigation
        searchBtn.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToSearchPage();
        });

        log('Search nav button created next to Tags');
    }

    /**
     * Create and inject the image labeling button in Stash navbar (next to Tags)
     */
    function createLabelingNavButton() {
        // Check if already exists
        if (document.getElementById('stash-copilot-labeling-nav-item')) {
            return;
        }

        // Find the search nav-item (li element) to insert after it
        const searchNavItem = document.getElementById('stash-copilot-search-nav-item');
        let insertAfterElement = searchNavItem;

        if (!insertAfterElement) {
            // Fallback: Find the Tags nav-item
            const tagsLink = document.querySelector('.navbar-nav a[href="/tags"]');
            if (!tagsLink) {
                log('Tags link not found for labeling button placement', 'warn');
                return;
            }
            insertAfterElement = tagsLink.closest('.nav-item') || tagsLink.parentElement;
        }

        if (!insertAfterElement) {
            log('Could not find insertion point for labeling button', 'warn');
            return;
        }

        // Create nav-item wrapper (li element to match Stash structure)
        const navItem = document.createElement('li');
        navItem.className = 'nav-item';
        navItem.id = 'stash-copilot-labeling-nav-item';

        // Create labeling button as a nav-link
        const labelingBtn = document.createElement('a');
        labelingBtn.id = 'stash-copilot-labeling-nav-btn';
        labelingBtn.className = 'stash-copilot-labeling-nav-btn nav-link';
        labelingBtn.href = '/plugins/stash-copilot/label';
        labelingBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
            </svg>
            <span>Image Labeling</span>
        `;
        labelingBtn.title = 'Label images for model training';

        // Add button to nav-item
        navItem.appendChild(labelingBtn);

        // Insert the nav-item after search nav-item (or Tags if search doesn't exist)
        insertAfterElement.parentNode.insertBefore(navItem, insertAfterElement.nextSibling);

        // Handle click - prevent default and use SPA navigation
        labelingBtn.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToLabelingPage();
        });

        log('Labeling nav button created next to AI Search');
    }

    /**
     * Navigate to labeling page using SPA-style navigation
     */
    function navigateToLabelingPage() {
        // Update URL without full page reload
        history.pushState({ stashCopilotLabeling: true }, '', '/plugins/stash-copilot/label');

        // Render labeling page
        renderLabelingPage();
    }

    /**
     * Navigate to search page using SPA-style navigation
     */
    function navigateToSearchPage() {
        // Update URL without full page reload
        history.pushState({ stashCopilotSearch: true }, '', '/plugins/stash-copilot/search');

        // Render search page
        renderSearchPage();
    }

    /**
     * Handle browser back/forward navigation and Stash nav link clicks
     */
    function setupSearchNavigationHandler() {
        // Handle browser back/forward
        window.addEventListener('popstate', () => {
            const path = window.location.pathname;

            if (path === '/plugins/stash-copilot/search') {
                // Navigating forward to search page
                renderSearchPage();
            } else {
                // Navigating away from search page - need to restore Stash content
                // Force a page reload to let Stash's router handle it properly
                if (document.querySelector('.stash-copilot-search-page')) {
                    window.location.reload();
                }
            }
        });

        // Intercept clicks on Stash nav links when search page is visible
        // This is needed because Stash uses pushState which doesn't trigger popstate
        document.addEventListener('click', (e) => {
            // Only care if search page is currently visible
            if (!document.querySelector('.stash-copilot-search-page')) {
                return;
            }

            // Check if click was on a Stash nav link (or inside one)
            const navLink = e.target.closest('.navbar-nav a.nav-link, .navbar-nav a[href]');
            if (navLink) {
                const href = navLink.getAttribute('href');
                // Skip our own search link
                if (href && href !== '/plugins/stash-copilot/search' && !href.startsWith('#')) {
                    // Prevent Stash's SPA navigation and do full reload to the target page
                    e.preventDefault();
                    e.stopPropagation();
                    window.location.href = href;
                }
            }
        }, true); // Use capture phase to intercept before Stash's handlers
    }

    /**
     * Handle browser back/forward navigation and Stash nav link clicks for labeling page
     */
    function setupLabelingNavigationHandler() {
        // Handle browser back/forward
        window.addEventListener('popstate', () => {
            const path = window.location.pathname;

            if (path === '/plugins/stash-copilot/label') {
                // Navigating forward to labeling page
                renderLabelingPage();
            } else {
                // Navigating away from labeling page - need to restore Stash content
                // Force a page reload to let Stash's router handle it properly
                if (document.querySelector('.stash-copilot-labeling-page')) {
                    window.location.reload();
                }
            }
        });

        // Intercept clicks on Stash nav links when labeling page is visible
        // This is needed because Stash uses pushState which doesn't trigger popstate
        document.addEventListener('click', (e) => {
            // Only care if labeling page is currently visible
            if (!document.querySelector('.stash-copilot-labeling-page')) {
                return;
            }

            // Check if click was on a Stash nav link (or inside one)
            const navLink = e.target.closest('.navbar-nav a.nav-link, .navbar-nav a[href]');
            if (navLink) {
                const href = navLink.getAttribute('href');
                // Skip our own labeling link
                if (href && href !== '/plugins/stash-copilot/label' && !href.startsWith('#')) {
                    // Prevent Stash's SPA navigation and do full reload to the target page
                    e.preventDefault();
                    e.stopPropagation();
                    window.location.href = href;
                }
            }
        }, true); // Use capture phase to intercept before Stash's handlers

        // Check if we're already on the labeling page (direct URL navigation)
        if (window.location.pathname === '/plugins/stash-copilot/label') {
            // Delay slightly to ensure DOM is ready
            setTimeout(() => renderLabelingPage(), 100);
        }
    }

    /**
     * Render the full-page semantic search interface
     */
    function renderSearchPage() {
        log('Rendering semantic search page...');

        // Get main content area - Stash uses different structures
        let mainContent = document.querySelector('.main');
        if (!mainContent) {
            mainContent = document.querySelector('#root > div:last-child');
        }
        if (!mainContent) {
            // Try to find any reasonable container
            mainContent = document.querySelector('.container-fluid') || document.querySelector('#root');
        }
        if (!mainContent) {
            log('Could not find main content area', 'error');
            return;
        }

        // Create search page container
        mainContent.innerHTML = `
            <div class="stash-copilot-search-page">
                <div class="stash-copilot-search-header">
                    <h1 class="stash-copilot-search-title">
                        <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                        AI Semantic Search
                    </h1>
                    <p class="stash-copilot-search-subtitle">
                        Find scenes using natural language. Describe what you're looking for.
                    </p>
                    <div class="stash-copilot-search-stats">
                        <span class="stash-copilot-search-embeddings-count">Semantic search powered by scene embeddings</span>
                    </div>
                    <div class="stash-copilot-search-model-selector">
                        <label for="stash-copilot-model-select">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                                <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                                <line x1="12" y1="22.08" x2="12" y2="12"/>
                            </svg>
                            Embedding Model:
                        </label>
                        <select id="stash-copilot-model-select" class="stash-copilot-model-dropdown">
                            <option value="">Loading models...</option>
                        </select>
                        <span class="stash-copilot-model-info"></span>
                    </div>
                    <div class="stash-copilot-search-mode-toggle">
                        <span class="stash-copilot-toggle-label">Search Mode:</span>
                        <div class="stash-copilot-toggle-buttons">
                            <button class="stash-copilot-toggle-btn ${!searchState.frameSearch ? 'active' : ''}"
                                    data-mode="scene">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="2" y="2" width="20" height="20" rx="2"/>
                                    <path d="M7 2v20M17 2v20M2 12h20"/>
                                </svg>
                                Scene
                            </button>
                            <button class="stash-copilot-toggle-btn ${searchState.frameSearch ? 'active' : ''}"
                                    data-mode="frame">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="2" y="2" width="20" height="20" rx="2"/>
                                    <circle cx="12" cy="12" r="3"/>
                                    <path d="M2 12h4M18 12h4M12 2v4M12 18v4"/>
                                </svg>
                                Frame
                            </button>
                        </div>
                    </div>
                </div>

                <div class="stash-copilot-search-input-container">
                    <input type="text"
                           class="stash-copilot-search-input"
                           placeholder="e.g., 'blonde in red lingerie', 'outdoor nature scene', 'POV with brunette'..."
                           value="${escapeHtml(searchState.lastQuery)}"
                           autofocus>
                    <button class="stash-copilot-search-btn">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                        Search
                    </button>
                </div>

                <div class="stash-copilot-search-loading" style="display: none;">
                    <div class="stash-copilot-spinner"></div>
                    <span>Searching...</span>
                </div>

                <div class="stash-copilot-search-results">
                    <div class="stash-copilot-search-results-header" style="display: none;">
                        <span class="stash-copilot-search-results-count"></span>
                        <span class="stash-copilot-search-results-query"></span>
                    </div>
                    <div class="stash-copilot-search-grid"></div>
                </div>

                <div class="stash-copilot-search-empty" style="display: none;">
                    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="M21 21l-4.35-4.35"/>
                        <path d="M8 8l6 6M14 8l-6 6"/>
                    </svg>
                    <h3>No results found</h3>
                    <p>Try a different search term or ensure scenes are embedded.</p>
                </div>

                <div class="stash-copilot-search-error" style="display: none;"></div>

                <div class="stash-copilot-search-pagination" style="display: none;">
                    <button class="stash-copilot-search-prev" disabled>← Previous</button>
                    <span class="stash-copilot-search-page-info">Page 1</span>
                    <button class="stash-copilot-search-next">Next →</button>
                </div>
            </div>
        `;

        // Setup event handlers
        setupSearchPageEvents(mainContent);

        // Check if we have cached results for the current query
        if (searchState.allResults.length > 0 && searchState.query) {
            // Restore from cache - no need to re-run the search
            const input = mainContent.querySelector('.stash-copilot-search-input');
            if (input) {
                input.value = searchState.query;
            }
            // Display the cached results for the current page
            displayCurrentPage();
            log(`Restored ${searchState.allResults.length} cached results for "${searchState.query}"`);
        } else if (searchState.lastQuery && !searchState.query) {
            // We have a saved query from localStorage but no cached results
            // Just populate the input, don't auto-search
            const input = mainContent.querySelector('.stash-copilot-search-input');
            if (input) {
                input.value = searchState.lastQuery;
            }
        }
    }

    /**
     * Load available embedding models from backend
     */
    async function loadAvailableModels() {
        if (searchState.modelsLoaded) {
            // Already loaded, just update the dropdown
            updateModelDropdown();
            return;
        }

        const requestId = `models-${Date.now()}`;

        try {
            // Run the get_embedding_models task
            await runPluginTask('Get Embedding Models', {
                request_id: requestId
            });

            // Poll for results
            const resultUrl = `/plugin/stash-copilot/assets/embedding_models_${requestId}.json`;
            let attempts = 0;
            const maxAttempts = 30;

            while (attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 200));
                attempts++;

                try {
                    const response = await fetch(resultUrl);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.status === 'complete') {
                            searchState.availableModels = data.models || [];
                            searchState.modelsLoaded = true;

                            // If no model selected, use the current configured model or first available
                            if (!searchState.selectedModel && data.current_model_key) {
                                searchState.selectedModel = data.current_model_key;
                            } else if (!searchState.selectedModel && searchState.availableModels.length > 0) {
                                searchState.selectedModel = searchState.availableModels[0].model_key;
                            }

                            updateModelDropdown();
                            log(`Loaded ${searchState.availableModels.length} embedding models`);
                            return;
                        } else if (data.status === 'error') {
                            log(`Error loading models: ${data.error}`, 'error');
                            updateModelDropdownError(data.error);
                            return;
                        }
                    }
                } catch (e) {
                    // File not ready yet, continue polling
                }
            }

            log('Timeout loading embedding models', 'warn');
            updateModelDropdownError('Timeout loading models');
        } catch (e) {
            log(`Failed to load models: ${e}`, 'error');
            updateModelDropdownError('Failed to load models');
        }
    }

    /**
     * Update the model dropdown with available models
     */
    function updateModelDropdown() {
        const dropdown = document.querySelector('#stash-copilot-model-select');
        const infoSpan = document.querySelector('.stash-copilot-model-info');
        if (!dropdown) return;

        if (searchState.availableModels.length === 0) {
            dropdown.innerHTML = '<option value="">No models available</option>';
            if (infoSpan) infoSpan.textContent = 'Run "Embed All Scenes" first';
            return;
        }

        // Build options
        dropdown.innerHTML = searchState.availableModels.map(model => {
            const selected = model.model_key === searchState.selectedModel ? 'selected' : '';
            const dims = model.dimensions ? `${model.dimensions}d` : '';
            return `<option value="${escapeHtml(model.model_key)}" ${selected}>
                ${escapeHtml(model.model_key)} (${model.count} scenes, ${dims})
            </option>`;
        }).join('');

        // Update info for selected model
        updateModelInfo();
    }

    /**
     * Update model dropdown to show error state
     */
    function updateModelDropdownError(error) {
        const dropdown = document.querySelector('#stash-copilot-model-select');
        const infoSpan = document.querySelector('.stash-copilot-model-info');
        if (dropdown) {
            dropdown.innerHTML = '<option value="">Error loading models</option>';
        }
        if (infoSpan) {
            infoSpan.textContent = error;
            infoSpan.style.color = '#ef4444';
        }
    }

    /**
     * Update the model info display based on selected model
     */
    function updateModelInfo() {
        const infoSpan = document.querySelector('.stash-copilot-model-info');
        const embeddingsCount = document.querySelector('.stash-copilot-search-embeddings-count');
        if (!infoSpan) return;

        const selectedModel = searchState.availableModels.find(
            m => m.model_key === searchState.selectedModel
        );

        if (selectedModel) {
            infoSpan.textContent = '';
            infoSpan.style.color = '';
            // Update the embeddings count display
            if (embeddingsCount) {
                embeddingsCount.textContent = `${selectedModel.count.toLocaleString()} scenes embedded with this model`;
            }
        } else {
            infoSpan.textContent = '';
        }
    }

    /**
     * Handle model selection change
     */
    function handleModelChange(event) {
        const newModel = event.target.value;
        if (newModel !== searchState.selectedModel) {
            searchState.selectedModel = newModel;
            localStorage.setItem('stash-copilot-selected-model', newModel);
            updateModelInfo();

            // Clear cached results since we're switching models
            searchState.allResults = [];
            searchState.totalFetched = 0;
            searchState.hasMoreOnServer = true;

            log(`Switched to embedding model: ${newModel}`);

            // If there's a current query, re-run the search with the new model
            if (searchState.query) {
                performSearch(searchState.query, true);
            }
        }
    }

    /**
     * Setup event handlers for search page
     */
    function setupSearchPageEvents(container) {
        const input = container.querySelector('.stash-copilot-search-input');
        const searchBtn = container.querySelector('.stash-copilot-search-btn');
        const prevBtn = container.querySelector('.stash-copilot-search-prev');
        const nextBtn = container.querySelector('.stash-copilot-search-next');
        const modelDropdown = container.querySelector('#stash-copilot-model-select');

        // Load available models
        loadAvailableModels();

        // Model selection change
        if (modelDropdown) {
            modelDropdown.addEventListener('change', handleModelChange);
        }

        // Search mode toggle
        container.querySelectorAll('.stash-copilot-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                const newFrameSearch = mode === 'frame';

                if (newFrameSearch !== searchState.frameSearch) {
                    searchState.frameSearch = newFrameSearch;
                    localStorage.setItem('stash-copilot-frame-search', newFrameSearch);

                    // Update button states
                    container.querySelectorAll('.stash-copilot-toggle-btn').forEach(b => {
                        b.classList.toggle('active', b.dataset.mode === mode);
                    });

                    // Clear cached results and re-search if we have a query
                    searchState.allResults = [];
                    searchState.totalFetched = 0;
                    searchState.hasMoreOnServer = true;

                    if (searchState.query) {
                        performSearch(searchState.query, true);
                    }

                    log(`Search mode changed to: ${mode}`);
                }
            });
        });

        // Search on Enter
        if (input) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performSearch(input.value.trim());
                }
            });
        }

        // Search button click
        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                performSearch(input.value.trim());
            });
        }

        // Pagination - uses goToSearchPage for instant cached navigation
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                goToSearchPage(searchState.currentPage - 1);
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                goToSearchPage(searchState.currentPage + 1);
            });
        }
    }

    /**
     * Perform semantic search - fetches 10 pages at once for fast pagination
     */
    async function performSearch(query, isNewSearch = true) {
        if (!query) return;

        // For new searches, reset state
        if (isNewSearch) {
            searchState.query = query;
            searchState.currentPage = 1;
            searchState.allResults = [];
            searchState.totalFetched = 0;
            searchState.hasMoreOnServer = true;

            // Save to localStorage
            localStorage.setItem('stash-copilot-last-search', query);
            searchState.lastQuery = query;
        }

        searchState.isSearching = true;

        // Generate request ID
        const requestId = `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
        searchState.requestId = requestId;

        // Update UI - show loading
        const container = document.querySelector('.stash-copilot-search-page');
        if (!container) return;

        const loadingDiv = container.querySelector('.stash-copilot-search-loading');
        const emptyDiv = container.querySelector('.stash-copilot-search-empty');
        const errorDiv = container.querySelector('.stash-copilot-search-error');

        if (loadingDiv) loadingDiv.style.display = 'flex';
        if (emptyDiv) emptyDiv.style.display = 'none';
        if (errorDiv) errorDiv.style.display = 'none';

        if (isNewSearch) {
            const gridDiv = container.querySelector('.stash-copilot-search-grid');
            const headerDiv = container.querySelector('.stash-copilot-search-results-header');
            if (gridDiv) gridDiv.innerHTML = '';
            if (headerDiv) headerDiv.style.display = 'none';
        }

        try {
            // Fetch 10 pages worth of results at once
            const batchSize = searchState.perPage * searchState.pagesPerBatch;
            const offset = searchState.totalFetched;

            // Build task args - include model_key if a specific model is selected
            const taskArgs = {
                query: query,
                limit: String(batchSize),
                offset: String(offset),
                request_id: requestId
            };

            // Add model_key if user has selected a specific model
            if (searchState.selectedModel) {
                taskArgs.model_key = searchState.selectedModel;
            }

            // Add frame search flag
            if (searchState.frameSearch) {
                taskArgs.frame_search = 'true';
            }

            await runPluginTask('Search Scenes by Text', taskArgs);

            // Poll for results
            pollSearchResults(requestId);

        } catch (error) {
            log(`Search error: ${error}`, 'error');
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (errorDiv) {
                errorDiv.style.display = 'block';
                errorDiv.innerHTML = `<p>Error: ${escapeHtml(error.message || 'Search failed')}</p>`;
            }
            searchState.isSearching = false;
        }
    }

    /**
     * Poll for search results
     */
    function pollSearchResults(requestId) {
        // Clear any existing poll
        if (searchState.pollInterval) {
            clearInterval(searchState.pollInterval);
        }

        const resultUrl = `/plugin/stash-copilot/assets/search_results_${requestId}.json`;
        let attempts = 0;
        const maxAttempts = 120; // 60 seconds max

        searchState.pollInterval = setInterval(async () => {
            attempts++;

            if (attempts > maxAttempts) {
                clearInterval(searchState.pollInterval);
                showSearchError('Search timed out. Please try again.');
                return;
            }

            try {
                const response = await fetch(`${resultUrl}?t=${Date.now()}`);
                if (!response.ok) return; // Not ready yet

                const data = await response.json();

                // Validate request ID
                if (data.request_id && data.request_id !== requestId) {
                    return; // Stale result
                }

                clearInterval(searchState.pollInterval);

                if (data.status === 'error') {
                    showSearchError(data.error);
                } else if (data.status === 'complete') {
                    handleSearchResults(data);
                }

            } catch (e) {
                // File not ready yet, continue polling
            }
        }, 500);
    }

    /**
     * Handle batch search results - stores all results and displays current page
     */
    function handleSearchResults(data) {
        searchState.isSearching = false;

        const newResults = data.results || [];

        // Append new results to our cache
        searchState.allResults = searchState.allResults.concat(newResults);
        searchState.totalFetched = searchState.allResults.length;
        searchState.hasMoreOnServer = data.has_more || false;

        // Display the current page
        displayCurrentPage();
    }

    /**
     * Display the current page from cached results (instant pagination)
     */
    function displayCurrentPage() {
        const container = document.querySelector('.stash-copilot-search-page');
        if (!container) return;

        const loadingDiv = container.querySelector('.stash-copilot-search-loading');
        const gridDiv = container.querySelector('.stash-copilot-search-grid');
        const headerDiv = container.querySelector('.stash-copilot-search-results-header');
        const emptyDiv = container.querySelector('.stash-copilot-search-empty');
        const paginationDiv = container.querySelector('.stash-copilot-search-pagination');

        if (loadingDiv) loadingDiv.style.display = 'none';

        // Calculate page slice
        const startIdx = (searchState.currentPage - 1) * searchState.perPage;
        const endIdx = startIdx + searchState.perPage;
        const pageResults = searchState.allResults.slice(startIdx, endIdx);

        // Calculate total pages available
        const totalCachedPages = Math.ceil(searchState.allResults.length / searchState.perPage);
        const hasNextPage = searchState.currentPage < totalCachedPages || searchState.hasMoreOnServer;

        if (searchState.allResults.length === 0) {
            if (emptyDiv) emptyDiv.style.display = 'flex';
            if (paginationDiv) paginationDiv.style.display = 'none';
            if (headerDiv) headerDiv.style.display = 'none';
            return;
        }

        // Show header
        if (headerDiv) {
            headerDiv.style.display = 'flex';
            const countSpan = container.querySelector('.stash-copilot-search-results-count');
            const querySpan = container.querySelector('.stash-copilot-search-results-query');
            const totalLabel = searchState.hasMoreOnServer ? `${searchState.allResults.length}+` : searchState.allResults.length;
            if (countSpan) countSpan.textContent = `${totalLabel} results`;
            if (querySpan) querySpan.textContent = `for "${searchState.query}"`;
        }

        // Render cards for current page
        if (gridDiv) {
            gridDiv.innerHTML = pageResults.map((item, index) => {
                // Determine thumbnail source for frame search results
                let overrideThumbnail = null;
                let matchTimestamp = null;

                if (searchState.frameSearch && item.frame_path) {
                    // Use frame thumbnail for frame search results
                    overrideThumbnail = `/plugin/stash-copilot/assets/${item.frame_path}`;
                    matchTimestamp = item.best_timestamp;
                }

                return buildSceneCard({
                    scene: item.scene,
                    score: item.similarity,
                    cardIndex: index,
                    theme: 'search',
                    scoreLabel: 'relevance',
                    overrideThumbnail: overrideThumbnail,
                    matchTimestamp: matchTimestamp
                });
            }).join('');

            // Setup card events
            setupSceneCardEvents(gridDiv, {
                theme: 'search',
                tooltipMode: 'fixed'
            });
        }

        // Update pagination
        if (paginationDiv) {
            paginationDiv.style.display = 'flex';
            const pageInfo = container.querySelector('.stash-copilot-search-page-info');
            const prevBtn = container.querySelector('.stash-copilot-search-prev');
            const nextBtn = container.querySelector('.stash-copilot-search-next');

            if (pageInfo) pageInfo.textContent = `Page ${searchState.currentPage}`;
            if (prevBtn) prevBtn.disabled = searchState.currentPage <= 1;
            if (nextBtn) nextBtn.disabled = !hasNextPage;
        }

        // Check if we need to pre-fetch the next batch
        // Trigger when user reaches the last 2 pages of cached results
        const pagesUntilEnd = totalCachedPages - searchState.currentPage;
        if (pagesUntilEnd <= 2 && searchState.hasMoreOnServer && !searchState.isSearching) {
            log(`Pre-fetching next batch (${searchState.pagesPerBatch} pages)...`);
            performSearch(searchState.query, false);
        }
    }

    /**
     * Navigate to a specific page (instant if cached)
     */
    function goToSearchPage(page) {
        if (page < 1) return;

        const totalCachedPages = Math.ceil(searchState.allResults.length / searchState.perPage);

        // Check if we have this page cached
        if (page <= totalCachedPages) {
            // Instant navigation - just render from cache
            searchState.currentPage = page;
            displayCurrentPage();
        } else if (searchState.hasMoreOnServer) {
            // Need to fetch more results
            searchState.currentPage = page;
            performSearch(searchState.query, false);
        }
    }

    /**
     * Show search error
     */
    function showSearchError(message) {
        searchState.isSearching = false;

        const container = document.querySelector('.stash-copilot-search-page');
        if (!container) return;

        const loadingDiv = container.querySelector('.stash-copilot-search-loading');
        const errorDiv = container.querySelector('.stash-copilot-search-error');

        if (loadingDiv) loadingDiv.style.display = 'none';
        if (errorDiv) {
            errorDiv.style.display = 'block';
            errorDiv.innerHTML = `<p>Error: ${escapeHtml(message)}</p>`;
        }
    }

    // Load dropdown data
    async function loadDropdownData(dropdown) {
        const statsGrid = dropdown.querySelector('.stash-copilot-stats');
        const summaryContent = dropdown.querySelector('.stash-copilot-summary-content');

        // Load stats
        statsGrid.innerHTML = '<div class="stash-copilot-spinner"></div>';
        const statsData = await getLibraryStats();

        if (statsData && statsData.stats) {
            const stats = statsData.stats;
            statsGrid.innerHTML = `
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${stats.scene_count.toLocaleString()}</div>
                    <div class="stash-copilot-stat-label">Scenes</div>
                </div>
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${formatDuration(stats.scenes_duration)}</div>
                    <div class="stash-copilot-stat-label">Duration</div>
                </div>
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${stats.total_play_count.toLocaleString()}</div>
                    <div class="stash-copilot-stat-label">Plays</div>
                </div>
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${formatDuration(stats.total_play_duration)}</div>
                    <div class="stash-copilot-stat-label">Watch Time</div>
                </div>
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${stats.performer_count.toLocaleString()}</div>
                    <div class="stash-copilot-stat-label">Performers</div>
                </div>
                <div class="stash-copilot-stat-item">
                    <div class="stash-copilot-stat-value">${stats.tag_count.toLocaleString()}</div>
                    <div class="stash-copilot-stat-label">Tags</div>
                </div>
            `;
        } else {
            statsGrid.innerHTML = '<p class="stash-copilot-error">Failed to load stats</p>';
        }

        // If generation is in progress, show generating message instead of old summary
        if (state.isGenerating) {
            summaryContent.innerHTML = `
                <div class="stash-copilot-info">
                    <span class="stash-copilot-spinner"></span> Generating summary... This may take a moment.
                </div>
            `;
            return;
        }

        // Load summary (only if not generating)
        const summary = await fetchLastSummary();
        if (summary && summary.summary) {
            const generatedAt = new Date(summary.generated_at).toLocaleString();
            const isStreaming = summary.status === 'streaming';
            summaryContent.innerHTML = `
                <div class="stash-copilot-summary-text">${renderMarkdown(summary.summary)}${isStreaming ? '<span class="stash-copilot-cursor"></span>' : ''}</div>
                <div class="stash-copilot-summary-meta">${isStreaming ? 'Generating...' : 'Generated: ' + generatedAt}</div>
            `;
        }
    }

    // ====================================================================
    // Tag Gaps Functions
    // ====================================================================

    async function detectTagGaps(modal) {
        if (state.tagGapsLoading) return;
        state.tagGapsLoading = true;
        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
        const emptyEl = modal.querySelector('.stash-copilot-tag-gaps-empty');
        detectBtn.disabled = true;
        detectBtn.innerHTML = '<span class="stash-copilot-spinner"></span> Detecting...';
        statusEl.textContent = 'Analyzing frames against tag embeddings...';
        if (emptyEl) emptyEl.style.display = 'none';
        const requestId = `tag_gaps_${Date.now()}`;
        state.tagGapsRequestId = requestId;
        try {
            const forceCheck = modal.querySelector('.stash-copilot-tag-gaps-force-check');
            const force = forceCheck && forceCheck.checked ? 'true' : 'false';
            await runPluginTask('Detect Tag Gaps', { request_id: requestId, force: force });
            pollTagGapsResults(modal, requestId);
        } catch (e) {
            log(`Detect Tag Gaps error: ${e.message}`, 'error');
            state.tagGapsLoading = false;
            detectBtn.disabled = false;
            detectBtn.textContent = 'Detect Tag Gaps';
            statusEl.textContent = `Error: ${e.message}`;
        }
    }

    function pollTagGapsResults(modal, requestId) {
        const resultFile = `/plugin/stash-copilot/assets/tag_gaps_${requestId}.json`;
        const interval = setInterval(async () => {
            if (state.tagGapsRequestId !== requestId) {
                clearInterval(interval);
                return;
            }
            try {
                const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        clearInterval(interval);
                        state.tagGapsData = data;
                        state.tagGapsLoading = false;
                        const progressEl = modal.querySelector('.stash-copilot-tag-gaps-progress');
                        if (progressEl) progressEl.style.display = 'none';
                        await renderTagGapsResults(modal, data);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        state.tagGapsLoading = false;
                        const progressEl = modal.querySelector('.stash-copilot-tag-gaps-progress');
                        if (progressEl) progressEl.style.display = 'none';
                        const btn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
                        const st = modal.querySelector('.stash-copilot-tag-gaps-status');
                        if (btn) { btn.disabled = false; btn.textContent = 'Detect Tag Gaps'; }
                        if (st) st.textContent = `Error: ${data.error || 'Unknown'}`;
                    } else if (data.status === 'processing') {
                        const progressEl = modal.querySelector('.stash-copilot-tag-gaps-progress');
                        const barEl = modal.querySelector('.stash-copilot-tag-gaps-progress-bar');
                        const textEl = modal.querySelector('.stash-copilot-tag-gaps-progress-text');
                        if (progressEl) progressEl.style.display = 'flex';
                        if (barEl) barEl.style.width = `${data.progress || 0}%`;
                        if (textEl) {
                            let msg = data.status_message || 'Processing...';
                            if (data.scenes_done != null && data.scenes_total != null) {
                                msg += ` (${data.scenes_done}/${data.scenes_total} scenes)`;
                            }
                            textEl.textContent = msg;
                        }
                        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
                        if (statusEl) statusEl.textContent = data.status_message || 'Processing...';
                    }
                }
            } catch (e) { /* not ready yet */ }
        }, 1500);

        setTimeout(() => {
            clearInterval(interval);
            if (state.tagGapsLoading) {
                state.tagGapsLoading = false;
                const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
                if (statusEl) statusEl.textContent = 'Timed out waiting for results';
            }
        }, 600000); // 10 min timeout
    }

    async function renderTagGapsResults(modal, data) {
        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
        const summaryEl = modal.querySelector('.stash-copilot-tag-gaps-summary');
        const statsEl = modal.querySelector('.stash-copilot-tag-gaps-stats');
        const resultsEl = modal.querySelector('.stash-copilot-tag-gaps-results');
        const sceneListEl = modal.querySelector('.stash-copilot-tag-gaps-scene-list');
        const emptyEl = modal.querySelector('.stash-copilot-tag-gaps-empty');
        if (detectBtn) { detectBtn.disabled = false; detectBtn.textContent = 'Re-detect'; }
        if (statusEl) statusEl.textContent = '';
        if (emptyEl) emptyEl.style.display = 'none';
        if (summaryEl && statsEl) {
            const avgPct = Math.round(data.avg_coverage * 100);
            statsEl.innerHTML = `
                <div class="stash-copilot-tag-gaps-stat-row">
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${avgPct}%</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Avg Coverage</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.flagged_scenes}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Scenes Flagged</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.total_scenes}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Total Scenes</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.threshold.toFixed(3)}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Threshold</span>
                    </div>
                </div>`;
            summaryEl.style.display = '';
        }
        if (resultsEl && sceneListEl) {
            const flagged = data.scenes.filter(s => s.coverage_ratio < 1.0);
            flagged.sort((a, b) => a.coverage_ratio - b.coverage_ratio);
            if (flagged.length === 0) {
                sceneListEl.innerHTML = '<p class="stash-copilot-info">All scenes are fully covered by existing tags.</p>';
            } else {
                const displayScenes = flagged.slice(0, 50);
                // Batch-fetch scene metadata from Stash GraphQL
                const sceneIds = displayScenes.map(s => s.scene_id);
                let sceneMap = {};
                try {
                    const gqlResult = await callGQL(`
                        query FindScenes($ids: [Int!]!) {
                            findScenes(scene_filter: { id: { modifier: INCLUDES, value: $ids } }, filter: { per_page: -1 }) {
                                scenes {
                                    id title date
                                    files { path duration height size fingerprints { type value } }
                                    performers { id name }
                                    studio { id name }
                                    tags { id name }
                                    play_count o_counter rating100 interactive
                                }
                            }
                        }`, { ids: sceneIds });
                    const scenes = gqlResult?.findScenes?.scenes || [];
                    scenes.forEach(s => { sceneMap[s.id] = s; });
                } catch (e) {
                    log(`Tag gaps: failed to fetch scene details: ${e.message}`, 'error');
                }

                sceneListEl.innerHTML = displayScenes.map((scene, idx) => {
                    const meta = sceneMap[String(scene.scene_id)] || { id: scene.scene_id };
                    const tagHints = (scene.top_uncovered_tags || []).slice(0, 3)
                        .map(t => `<span class="stash-copilot-tag-gaps-hint">${escapeHtml(t.tag)} ${(t.avg_similarity * 100).toFixed(0)}%</span>`)
                        .join('');

                    const cardHtml = buildSceneCard({
                        scene: meta,
                        score: scene.coverage_ratio,
                        cardIndex: idx,
                        theme: 'tag-gaps',
                        scoreLabel: 'covered'
                    });

                    return `<div class="stash-copilot-tag-gaps-card-wrap">
                        ${cardHtml}
                        ${tagHints ? `<div class="stash-copilot-tag-gaps-card-hints">${tagHints}</div>` : ''}
                    </div>`;
                }).join('');

                setupSceneCardEvents(sceneListEl, { theme: 'tag-gaps', tooltipMode: 'fixed' });
            }
            resultsEl.style.display = '';
        }
    }

    function setupTagGapsEvents(modal) {
        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        if (detectBtn) {
            detectBtn.addEventListener('click', () => detectTagGaps(modal));
        }
    }

    async function loadTagGapsData(modal) {
        // Setup events on first load
        if (!modal._tagGapsEventsSetup) {
            setupTagGapsEvents(modal);
            modal._tagGapsEventsSetup = true;
        }

        // In-memory data takes priority
        if (state.tagGapsData) {
            await renderTagGapsResults(modal, state.tagGapsData);
            return;
        }

        // If currently loading (polling), don't fetch cached data
        if (state.tagGapsLoading) return;

        // Attempt to load persisted tag gaps data
        try {
            const resp = await fetch(
                `/plugin/stash-copilot/assets/tag_gaps_latest.json?t=${Date.now()}`,
                { cache: 'no-store' }
            );
            if (resp.ok) {
                const data = await resp.json();
                if (data.status === 'complete' && data.scenes?.length > 0) {
                    state.tagGapsData = data;
                    await renderTagGapsResults(modal, data);
                }
            }
        } catch (e) {
            // No persisted data - show empty state
        }
    }

    // ===== Scene Vision Functions =====

    // Custom prompts storage
    const PROMPT_STORAGE_KEY = 'stash-copilot-vision-prompts';

    function loadCustomPrompts() {
        try {
            const stored = localStorage.getItem(PROMPT_STORAGE_KEY);
            return stored ? JSON.parse(stored) : {};
        } catch (e) {
            return {};
        }
    }

    function saveCustomPrompts(prompts) {
        try {
            localStorage.setItem(PROMPT_STORAGE_KEY, JSON.stringify(prompts));
        } catch (e) {
            log('Failed to save prompts to localStorage', 'error');
        }
    }

    function clearCustomPrompts() {
        localStorage.removeItem(PROMPT_STORAGE_KEY);
    }

    // Create the vision analysis modal
    function createVisionModal() {
        // Remove existing modal if present
        const existing = document.getElementById('stash-copilot-vision-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'stash-copilot-vision-modal';
        modal.className = 'stash-copilot-vision-modal';
        modal.innerHTML = `
            <div class="stash-copilot-vision-content">
                <div class="stash-copilot-vision-header">
                    <h3>Scene Vision Analysis</h3>
                    <div class="stash-copilot-vision-header-actions">
                        <button class="stash-copilot-vision-settings" title="Edit Prompts">⚙</button>
                        <button class="stash-copilot-vision-reanalyze" title="Re-analyze scene (uses cached frames)">↻</button>
                        <button class="stash-copilot-vision-close">&times;</button>
                    </div>
                </div>
                <div class="stash-copilot-vision-body">
                    <div class="stash-copilot-vision-analysis">
                        <div class="stash-copilot-vision-loading">
                            <div class="stash-copilot-spinner"></div>
                            <span class="stash-copilot-vision-status">Initializing...</span>
                            <div class="stash-copilot-vision-progress-container">
                                <div class="stash-copilot-vision-progress-bar" style="width: 0%"></div>
                            </div>
                            <span class="stash-copilot-vision-progress-text"></span>
                        </div>
                    </div>
                    <div class="stash-copilot-vision-tags-section" style="display: none;">
                        <div class="stash-copilot-section-header" data-section="tags">
                            <h4>Suggested Tags</h4>
                            <button class="stash-copilot-collapse-btn" title="Toggle visibility">
                                <span class="stash-copilot-collapse-icon">▼</span>
                            </button>
                        </div>
                        <div class="stash-copilot-vision-tags"></div>
                    </div>
                    <div class="stash-copilot-vision-prompts" style="display: none;">
                        <div class="stash-copilot-vision-prompts-header">
                            <h4>Edit Prompts</h4>
                        </div>
                        <div class="stash-copilot-vision-prompts-content">
                            <div class="stash-copilot-prompt-field">
                                <label>System Prompt</label>
                                <textarea class="stash-copilot-prompt-system" rows="4" placeholder="System prompt for the VLM..."></textarea>
                            </div>
                            <div class="stash-copilot-prompt-field">
                                <label>Description Prompt <small>(use {frame_count}, {performer_context})</small></label>
                                <textarea class="stash-copilot-prompt-description" rows="10" placeholder="Description prompt template..."></textarea>
                            </div>
                            <div class="stash-copilot-prompt-actions">
                                <button class="stash-copilot-prompt-save">Save & Re-analyze</button>
                                <button class="stash-copilot-prompt-reset">Reset to Defaults</button>
                            </div>
                        </div>
                    </div>
                    <div class="stash-copilot-vision-debug" style="display: none;">
                        <div class="stash-copilot-vision-debug-header">
                            <h4>Debug Info</h4>
                            <button class="stash-copilot-vision-debug-toggle">▼</button>
                        </div>
                        <div class="stash-copilot-vision-debug-content">
                            <div class="stash-copilot-debug-stage" data-stage="description">
                                <h5>Stage 1: Description (VLM)</h5>
                                <div class="stash-copilot-debug-tokens"></div>
                                <details>
                                    <summary>System Prompt</summary>
                                    <pre class="stash-copilot-debug-system-prompt"></pre>
                                </details>
                                <details>
                                    <summary>User Prompt</summary>
                                    <pre class="stash-copilot-debug-user-prompt"></pre>
                                </details>
                                <details>
                                    <summary>Frames (<span class="frame-count">0</span>)</summary>
                                    <div class="stash-copilot-debug-frames"></div>
                                </details>
                            </div>
                            <div class="stash-copilot-debug-stage" data-stage="tagging">
                                <h5>Stage 2: Tagging (Text LLM)</h5>
                                <div class="stash-copilot-debug-tokens"></div>
                                <details>
                                    <summary>System Prompt</summary>
                                    <pre class="stash-copilot-debug-system-prompt"></pre>
                                </details>
                                <details>
                                    <summary>User Prompt</summary>
                                    <pre class="stash-copilot-debug-user-prompt"></pre>
                                </details>
                            </div>
                            <div class="stash-copilot-debug-totals"></div>
                        </div>
                    </div>
                    <div class="stash-copilot-vision-chat">
                        <div class="stash-copilot-vision-messages"></div>
                        <div class="stash-copilot-vision-input-container">
                            <input type="text" class="stash-copilot-vision-input"
                                   placeholder="Ask about this scene..."
                                   maxlength="500">
                            <button class="btn btn-primary stash-copilot-vision-send">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add event listeners
        modal.querySelector('.stash-copilot-vision-close').addEventListener('click', closeVisionModal);
        modal.querySelector('.stash-copilot-vision-reanalyze').addEventListener('click', () => {
            if (!visionState.isAnalyzing && visionState.sceneId) {
                reanalyzeScene(visionState.sceneId);
            }
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeVisionModal();
        });

        const input = modal.querySelector('.stash-copilot-vision-input');
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendVisionMessage(input.value);
            }
        });

        modal.querySelector('.stash-copilot-vision-send').addEventListener('click', () => {
            sendVisionMessage(input.value);
        });

        // Debug toggle
        const debugToggle = modal.querySelector('.stash-copilot-vision-debug-toggle');
        const debugContent = modal.querySelector('.stash-copilot-vision-debug-content');
        debugToggle.addEventListener('click', () => {
            const isHidden = debugContent.style.display === 'none';
            debugContent.style.display = isHidden ? 'block' : 'none';
            debugToggle.textContent = isHidden ? '▲' : '▼';
        });

        // Prompt editor toggle
        const settingsBtn = modal.querySelector('.stash-copilot-vision-settings');
        const promptsSection = modal.querySelector('.stash-copilot-vision-prompts');
        settingsBtn.addEventListener('click', () => {
            const isHidden = promptsSection.style.display === 'none';
            promptsSection.style.display = isHidden ? 'block' : 'none';
            settingsBtn.classList.toggle('active', isHidden);
        });

        // Prompt editor - Save & Re-analyze
        modal.querySelector('.stash-copilot-prompt-save').addEventListener('click', () => {
            const systemPrompt = modal.querySelector('.stash-copilot-prompt-system').value;
            const descriptionPrompt = modal.querySelector('.stash-copilot-prompt-description').value;
            saveCustomPrompts({ system_prompt: systemPrompt, description_prompt: descriptionPrompt });
            log('Custom prompts saved to localStorage');
            // Trigger re-analysis with new prompts
            if (visionState.sceneId) {
                reanalyzeScene(visionState.sceneId);
            }
        });

        // Prompt editor - Reset to defaults
        modal.querySelector('.stash-copilot-prompt-reset').addEventListener('click', () => {
            clearCustomPrompts();
            modal.querySelector('.stash-copilot-prompt-system').value = '';
            modal.querySelector('.stash-copilot-prompt-description').value = '';
            log('Custom prompts cleared, using defaults');
            // Trigger re-analysis with default prompts
            if (visionState.sceneId) {
                reanalyzeScene(visionState.sceneId);
            }
        });

        return modal;
    }

    // Re-analyze scene (clear conversation history, keep cached frames)
    async function reanalyzeScene(sceneId) {
        log(`Re-analyzing scene ${sceneId} (clearing conversation history, keeping cached frames)`);

        // Delete the history file by triggering task with clear_history flag
        const historyFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;

        // Reset UI to loading state
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (modal) {
            const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');
            const tagsSection = modal.querySelector('.stash-copilot-vision-tags-section');
            const messagesDiv = modal.querySelector('.stash-copilot-vision-messages');

            analysisDiv.innerHTML = `
                <div class="stash-copilot-vision-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span class="stash-copilot-vision-status">Re-analyzing scene...</span>
                    <div class="stash-copilot-vision-progress-container">
                        <div class="stash-copilot-vision-progress-bar" style="width: 0%"></div>
                    </div>
                    <span class="stash-copilot-vision-progress-text"></span>
                </div>
            `;
            tagsSection.style.display = 'none';
            messagesDiv.innerHTML = '';
        }

        // Reset state
        visionState.messages = [];
        visionState.description = null;
        visionState.suggestedTags = [];
        visionState.tagTimestamps = {};
        visionState.conversationId = null;

        // Start fresh analysis with clear_history flag
        startVisionAnalysis(sceneId, true);
    }

    // Open vision modal and start analysis
    async function openVisionModal(sceneId) {
        visionState.sceneId = sceneId;
        visionState.modalOpen = true;
        visionState.messages = [];
        visionState.description = null;
        visionState.suggestedTags = [];
        visionState.tagTimestamps = {};

        const modal = createVisionModal();
        modal.classList.add('open');

        // First, try to load cached result
        const cachedResult = await loadCachedVisionResult(sceneId);
        if (cachedResult && cachedResult.description) {
            log('Found cached vision result, displaying immediately');
            visionState.isAnalyzing = false;
            visionState.conversationId = cachedResult.conversation_id;
            visionState.description = cachedResult.description;
            visionState.suggestedTags = cachedResult.suggested_tags || [];
            visionState.tagTimestamps = cachedResult.tag_timestamps || {};
            visionState.messages = cachedResult.messages || [];
            renderVisionResult(cachedResult);
        } else {
            // No cached result, start fresh analysis
            startVisionAnalysis(sceneId);
        }
    }

    // Load cached vision result without starting a new task
    async function loadCachedVisionResult(sceneId) {
        const historyFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;
        try {
            const cacheBuster = `?t=${Date.now()}`;
            const response = await fetch(historyFile + cacheBuster, { cache: 'no-store' });
            if (response.ok) {
                const data = await response.json();
                // Return data if it has a valid description
                if (data.description && data.messages?.length > 0) {
                    return data;
                }
            }
        } catch (e) {
            log(`No cached result found: ${e.message}`);
        }
        return null;
    }

    // Close vision modal
    function closeVisionModal() {
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (modal) {
            modal.classList.remove('open');
            setTimeout(() => modal.remove(), 300);
        }
        visionState.modalOpen = false;
    }

    // Start vision analysis task
    async function startVisionAnalysis(sceneId, clearHistory = false) {
        if (visionState.isAnalyzing) return;

        visionState.isAnalyzing = true;
        visionState.analysisStartTime = Date.now();
        log(`Starting vision analysis for scene ${sceneId}${clearHistory ? ' (clearing history)' : ''}`);

        // Build args
        const args = [
            { key: 'mode', value: { str: 'scene_vision' } },
            { key: 'scene_id', value: { str: sceneId } }
        ];

        if (clearHistory) {
            args.push({ key: 'clear_history', value: { str: 'true' } });
        }

        // Add custom prompts if saved in localStorage
        const customPrompts = loadCustomPrompts();
        if (customPrompts?.system_prompt) {
            args.push({ key: 'custom_system_prompt', value: { str: customPrompts.system_prompt } });
        }
        if (customPrompts?.description_prompt) {
            args.push({ key: 'custom_description_prompt', value: { str: customPrompts.description_prompt } });
        }

        try {
            const result = await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: args
            });

            if (result && result.runPluginTask) {
                log('Vision task started, polling for results...');
                pollVisionResult(sceneId);
            } else {
                showVisionError('Failed to start vision analysis');
            }
        } catch (error) {
            log(`Vision analysis error: ${error}`, 'error');
            showVisionError('Error starting vision analysis');
        }
    }

    /**
     * Get human-readable status message for analysis stages
     */
    function getAnalysisStatusMessage(status) {
        const messages = {
            'pending': 'Initializing...',
            'extracting': 'Extracting frames...',
            'classifying': 'Stage 1/3: Classifying scene...',
            'describing': 'Stage 2/3: Generating description...',
            'verifying': 'Stage 3/3: Verifying claims...',
            'tagging': 'Suggesting tags...',
            'complete': 'Analysis complete',
            'partial': 'Partial results (see below)',
            'error': 'Analysis failed',
        };
        return messages[status] || status;
    }

    // Update progress display in modal
    function updateVisionProgress(data) {
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const statusEl = modal.querySelector('.stash-copilot-vision-status');
        const progressBar = modal.querySelector('.stash-copilot-vision-progress-bar');
        const progressText = modal.querySelector('.stash-copilot-vision-progress-text');
        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');
        const tagsSection = modal.querySelector('.stash-copilot-vision-tags-section');
        const tagsDiv = modal.querySelector('.stash-copilot-vision-tags');

        if (statusEl) {
            statusEl.textContent = data.status_message || getAnalysisStatusMessage(data.status);
        }

        if (progressBar && typeof data.progress === 'number') {
            progressBar.style.width = `${data.progress}%`;
        }

        if (progressText && data.total_frames > 0) {
            if (data.status === 'extracting' || data.stage === 'extracting') {
                const currentFrame = Math.round((data.progress - 10) / 60 * data.total_frames);
                progressText.textContent = `Frame ${currentFrame}/${data.total_frames}`;
            } else if (data.status === 'describing' || data.stage === 'describing') {
                // Show model name and elapsed time
                let text = data.description_model || `${data.total_frames} frames`;
                if (data.stage_start_time) {
                    const elapsed = Math.round((Date.now() - new Date(data.stage_start_time).getTime()) / 1000);
                    text += ` • ${elapsed}s`;
                }
                progressText.textContent = text;
            } else if (data.status === 'tagging' || data.stage === 'tagging') {
                // Show tag model name and elapsed time
                let text = data.tag_model || 'Suggesting tags';
                if (data.stage_start_time) {
                    const elapsed = Math.round((Date.now() - new Date(data.stage_start_time).getTime()) / 1000);
                    text += ` • ${elapsed}s`;
                }
                progressText.textContent = text;
            } else {
                progressText.textContent = '';
            }
        }

        // Progressive display: Show description as soon as it's ready
        if (data.description_complete && data.description && analysisDiv) {
            // Check if we're still showing the loading state
            const loadingDiv = analysisDiv.querySelector('.stash-copilot-vision-loading');
            if (loadingDiv) {
                // Render classification badges if available
                const badgesHtml = renderClassificationBadges(data.classification);

                // Replace loading with badges and description
                analysisDiv.innerHTML = `
                    ${badgesHtml}
                    <div class="stash-copilot-vision-description">
                        ${renderMarkdown(data.description)}
                    </div>
                `;
            }
        }

        // Show tags loading while Stage 2 runs
        if (data.description_complete && !data.tags_complete && tagsSection && tagsDiv) {
            tagsSection.style.display = 'block';
            // Only update if not already showing loading
            if (!tagsDiv.querySelector('.stash-copilot-vision-tags-loading')) {
                tagsDiv.innerHTML = `
                    <div class="stash-copilot-vision-tags-loading">
                        <span class="stash-copilot-spinner"></span>
                        Generating tag suggestions...
                    </div>
                `;
            }
        }

        // Show tags when complete
        if (data.tags_complete && data.suggested_tags && tagsSection && tagsDiv) {
            renderVisionTags(data.suggested_tags, data.tag_timestamps || {}, tagsDiv, tagsSection, data.tag_sources || {}, data.tag_confidences || {});
        }
    }

    /**
     * Render the image labeling page
     */
    function renderLabelingPage() {
        log('Rendering labeling page...');

        let mainContent = document.querySelector('.main');
        if (!mainContent) {
            mainContent = document.querySelector('#root > div:last-child');
        }
        if (!mainContent) {
            mainContent = document.querySelector('.container-fluid') || document.querySelector('#root');
        }
        if (!mainContent) {
            log('Could not find main content area', 'error');
            return;
        }

        mainContent.innerHTML = `
            <div class="stash-copilot-label-page">
                <div class="stash-copilot-label-header">
                    <div class="stash-copilot-label-header-left">
                        <a href="/scenes" class="stash-copilot-label-back-btn">← Back</a>
                        <h1 class="stash-copilot-label-title">
                            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
                                <line x1="7" y1="7" x2="7.01" y2="7"/>
                            </svg>
                            Image Labeling
                        </h1>
                    </div>
                    <div class="stash-copilot-label-header-center">
                        <span class="stash-copilot-label-progress-text">No session</span>
                        <div class="stash-copilot-label-progress-bar">
                            <div class="stash-copilot-label-progress-fill"></div>
                        </div>
                    </div>
                    <div class="stash-copilot-label-header-right">
                        <button class="stash-copilot-label-view-toggle" data-mode="single" title="Toggle view (G)">
                            <span class="view-single active">▣</span>
                            <span class="view-grid">⊞</span>
                        </button>
                        <button class="stash-copilot-label-export-btn" title="Export dataset">Export</button>
                        <button class="stash-copilot-label-settings-btn" title="Settings">⚙</button>
                    </div>
                </div>

                <div class="stash-copilot-label-body">
                    <div class="stash-copilot-label-intro">
                        <div class="stash-copilot-label-intro-icon">🏷️</div>
                        <h2>Start Labeling Session</h2>
                        <p>Label images with tags to create training data for embedding model fine-tuning.
                           Uses uncertainty sampling to show you images where the model needs the most help.</p>
                        <div class="stash-copilot-label-session-controls">
                            <label>Batch size:
                                <input type="number" class="stash-copilot-label-batch-input" value="200" min="10" max="1000" step="10">
                            </label>
                            <button class="stash-copilot-label-start-btn">Start New Session</button>
                        </div>
                        <div class="stash-copilot-label-previous-sessions"></div>
                    </div>

                    <div class="stash-copilot-label-loading" style="display: none;">
                        <div class="stash-copilot-spinner"></div>
                        <span class="stash-copilot-label-loading-status">Preparing session...</span>
                    </div>

                    <div class="stash-copilot-label-single" style="display: none;">
                        <div class="stash-copilot-label-image-area">
                            <img class="stash-copilot-label-image" src="" alt="Frame to label">
                            <div class="stash-copilot-label-image-meta"></div>
                        </div>
                        <div class="stash-copilot-label-tag-panel">
                            <h3 class="stash-copilot-label-section-title">Suggested Tags</h3>
                            <div class="stash-copilot-label-suggestions"></div>
                            <h3 class="stash-copilot-label-section-title">Scene Tags</h3>
                            <div class="stash-copilot-label-scene-tags"></div>
                            <h3 class="stash-copilot-label-section-title">Add Tag</h3>
                            <div class="stash-copilot-label-autocomplete">
                                <input type="text" class="stash-copilot-label-tag-input"
                                       placeholder="Type to search tags... (/)"
                                       autocomplete="off">
                                <div class="stash-copilot-label-autocomplete-dropdown"></div>
                            </div>
                            <div class="stash-copilot-label-manual-tags"></div>
                        </div>
                    </div>

                    <div class="stash-copilot-label-grid" style="display: none;">
                        <div class="stash-copilot-label-grid-images"></div>
                        <div class="stash-copilot-label-tag-panel"></div>
                    </div>
                </div>

                <div class="stash-copilot-label-footer" style="display: none;">
                    <div class="stash-copilot-label-nav">
                        <button class="stash-copilot-label-prev-btn" title="Previous (←)">← Prev</button>
                        <span class="stash-copilot-label-position">0 / 0</span>
                        <button class="stash-copilot-label-next-btn" title="Next (→)">Next →</button>
                    </div>
                    <div class="stash-copilot-label-actions">
                        <button class="stash-copilot-label-skip-btn" title="Skip (S)">Skip</button>
                        <button class="stash-copilot-label-save-btn" title="Save & Next (Enter)">Save & Next</button>
                    </div>
                </div>
            </div>
        `;

        setupLabelingEvents(mainContent);
        loadPreviousSessions(mainContent);
    }

    function setupLabelingEvents(container) {
        const startBtn = container.querySelector('.stash-copilot-label-start-btn');
        startBtn.addEventListener('click', () => {
            const batchInput = container.querySelector('.stash-copilot-label-batch-input');
            const batchSize = parseInt(batchInput.value, 10) || 200;
            startLabelingSession(container, batchSize);
        });

        const prevBtn = container.querySelector('.stash-copilot-label-prev-btn');
        const nextBtn = container.querySelector('.stash-copilot-label-next-btn');
        prevBtn.addEventListener('click', () => navigateLabeling(container, -1));
        nextBtn.addEventListener('click', () => navigateLabeling(container, 1));

        const skipBtn = container.querySelector('.stash-copilot-label-skip-btn');
        const saveBtn = container.querySelector('.stash-copilot-label-save-btn');
        skipBtn.addEventListener('click', () => skipFrame(container));
        saveBtn.addEventListener('click', () => saveAndNext(container));

        const viewToggle = container.querySelector('.stash-copilot-label-view-toggle');
        viewToggle.addEventListener('click', () => toggleViewMode(container));

        const exportBtn = container.querySelector('.stash-copilot-label-export-btn');
        exportBtn.addEventListener('click', () => exportDataset(container));

        document.addEventListener('keydown', (e) => handleLabelingKeyboard(e, container));
    }

    async function startLabelingSession(container, batchSize) {
        const introEl = container.querySelector('.stash-copilot-label-intro');
        const loadingEl = container.querySelector('.stash-copilot-label-loading');

        introEl.style.display = 'none';
        loadingEl.style.display = 'flex';

        const requestId = `label_${Date.now()}`;
        labelingState.isLoading = true;

        try {
            await runPluginTask('Prepare Labeling Session', {
                batch_size: String(batchSize),
                request_id: requestId,
            });

            await pollLabelingSession(container, requestId);
        } catch (error) {
            log(`Error starting session: ${error.message}`, 'error');
            loadingEl.style.display = 'none';
            introEl.style.display = 'flex';
        }
    }

    async function pollLabelingSession(container, requestId) {
        const loadingEl = container.querySelector('.stash-copilot-label-loading');
        const statusEl = loadingEl.querySelector('.stash-copilot-label-loading-status');

        const maxAttempts = 300; // 5 min — large libraries need time for sampling
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const resp = await fetch(
                    `/plugin/stash-copilot/assets/labeling_session_${requestId}.json?t=${Date.now()}`,
                    { cache: 'no-store' }
                );
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        labelingState.sessionId = data.session_id;
                        labelingState.batch = data.batch;
                        labelingState.vocabulary = data.vocabulary;
                        labelingState.currentIndex = 0;
                        labelingState.annotations = {};
                        labelingState.isLoading = false;

                        loadingEl.style.display = 'none';
                        showLabelingUI(container);
                        renderCurrentFrame(container);
                        return;
                    } else if (data.status === 'error' || data.status === 'no_embeddings') {
                        loadingEl.style.display = 'none';
                        const introEl = container.querySelector('.stash-copilot-label-intro');
                        introEl.style.display = 'flex';
                        alert(data.error || 'Failed to prepare session');
                        return;
                    }
                    if (statusEl) statusEl.textContent = 'Computing uncertainty scores...';
                }
            } catch (e) {
                // File not ready yet
            }
            await new Promise(r => setTimeout(r, 1000));
        }

        loadingEl.style.display = 'none';
        const introEl = container.querySelector('.stash-copilot-label-intro');
        introEl.style.display = 'flex';
        alert('Session preparation timed out');
    }

    function showLabelingUI(container) {
        const singleView = container.querySelector('.stash-copilot-label-single');
        const footer = container.querySelector('.stash-copilot-label-footer');

        if (labelingState.viewMode === 'single') {
            singleView.style.display = 'flex';
        }
        footer.style.display = 'flex';
        updateProgress(container);
    }

    async function loadPreviousSessions(container) {
        const sessionsEl = container.querySelector('.stash-copilot-label-previous-sessions');
        if (!sessionsEl) return;

        const requestId = `sessions_${Date.now()}`;
        try {
            await runPluginTask('Get Labeling Sessions', { request_id: requestId });

            // Poll for result
            for (let i = 0; i < 30; i++) {
                try {
                    const resp = await fetch(
                        `/plugin/stash-copilot/assets/labeling_sessions_${requestId}.json?t=${Date.now()}`,
                        { cache: 'no-store' }
                    );
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.status === 'complete' && data.sessions && data.sessions.length > 0) {
                            sessionsEl.innerHTML = '<h3 style="color: #999; font-size: 14px; margin-top: 24px;">Previous Sessions</h3>' +
                                data.sessions.map(s => `
                                    <div class="stash-copilot-label-session-card" data-session-id="${s.session_id}">
                                        <span class="session-status ${s.status}">${s.status}</span>
                                        <span class="session-progress">${s.labeled_count}/${s.total_frames} labeled</span>
                                        <span class="session-date">${new Date(s.created_at).toLocaleDateString()}</span>
                                    </div>
                                `).join('');
                            return;
                        }
                        if (data.status === 'complete') return; // no sessions
                    }
                } catch (e) { /* not ready */ }
                await new Promise(r => setTimeout(r, 500));
            }
        } catch (e) {
            log(`Error loading sessions: ${e.message}`, 'error');
        }
    }

    // Task 8: renderCurrentFrame, toggleTagState, autocomplete, addManualTag
    function renderCurrentFrame(container) {
        const item = labelingState.batch[labelingState.currentIndex];
        if (!item) return;

        const imageEl = container.querySelector('.stash-copilot-label-image');
        const metaEl = container.querySelector('.stash-copilot-label-image-meta');
        const suggestionsEl = container.querySelector('.stash-copilot-label-suggestions');
        const sceneTagsEl = container.querySelector('.stash-copilot-label-scene-tags');
        const manualTagsEl = container.querySelector('.stash-copilot-label-manual-tags');

        // Load image
        const framePath = item.frame_path.replace(/^.*?assets\//, '/plugin/stash-copilot/assets/');
        imageEl.src = framePath;
        imageEl.alt = `Scene ${item.scene_id} - Frame ${item.frame_index}`;

        // Meta info
        metaEl.innerHTML = `
            <span class="stash-copilot-label-meta-scene">
                <a href="/scenes/${item.scene_id}" target="_blank">${escapeHtml(item.scene_title)}</a>
            </span>
            <span class="stash-copilot-label-meta-time">${item.timestamp}</span>
            <span class="stash-copilot-label-meta-uncertainty" title="Uncertainty score">⚡ ${item.uncertainty_score}</span>
        `;

        // Get existing annotations for this frame
        const frameKey = `${item.scene_id}_${item.frame_index}`;
        const existing = labelingState.annotations[frameKey] || {};

        // Render suggested tags
        suggestionsEl.innerHTML = item.suggested_tags.map((tag, idx) => {
            const label = existing[tag.tag_text] || 'undecided';
            return `
                <div class="stash-copilot-label-tag-row" data-tag="${escapeHtml(tag.tag_text)}" data-state="${label}">
                    <button class="stash-copilot-label-tag-toggle" data-key="${idx + 1}">
                        <span class="tag-state-icon">${label === 'confirmed' ? '✓' : label === 'rejected' ? '✗' : '?'}</span>
                    </button>
                    <span class="stash-copilot-label-tag-name">${escapeHtml(tag.tag_text)}</span>
                    <span class="stash-copilot-label-tag-sim">${(tag.similarity * 100).toFixed(0)}%</span>
                    <span class="stash-copilot-label-tag-key">${idx + 1}</span>
                </div>
            `;
        }).join('');

        // Setup tag toggle events
        suggestionsEl.querySelectorAll('.stash-copilot-label-tag-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const row = btn.closest('.stash-copilot-label-tag-row');
                toggleTagState(row, frameKey);
            });
        });

        // Render scene tags (read-only)
        sceneTagsEl.innerHTML = item.scene_tags.map(tag =>
            `<span class="stash-copilot-label-scene-tag-pill">${escapeHtml(tag)}</span>`
        ).join('');

        // Render manually added tags
        const manualTags = Object.entries(existing)
            .filter(([text, label]) => label === 'confirmed' && !item.suggested_tags.some(s => s.tag_text === text))
            .map(([text]) => text);

        manualTagsEl.innerHTML = manualTags.map(tag =>
            `<span class="stash-copilot-label-manual-tag">
                ${escapeHtml(tag)}
                <button class="stash-copilot-label-remove-tag" data-tag="${escapeHtml(tag)}">×</button>
            </span>`
        ).join('');

        // Setup remove tag events
        manualTagsEl.querySelectorAll('.stash-copilot-label-remove-tag').forEach(btn => {
            btn.addEventListener('click', () => {
                const tagText = btn.dataset.tag;
                delete labelingState.annotations[frameKey][tagText];
                renderCurrentFrame(container);
            });
        });

        // Setup autocomplete
        setupLabelingAutocomplete(container);

        updateProgress(container);
    }

    function toggleTagState(row, frameKey) {
        const tagText = row.dataset.tag;
        const currentState = row.dataset.state;
        const states = ['undecided', 'confirmed', 'rejected'];
        const nextIdx = (states.indexOf(currentState) + 1) % states.length;
        const newState = states[nextIdx];

        row.dataset.state = newState;
        const icon = row.querySelector('.tag-state-icon');
        icon.textContent = newState === 'confirmed' ? '✓' : newState === 'rejected' ? '✗' : '?';

        if (!labelingState.annotations[frameKey]) {
            labelingState.annotations[frameKey] = {};
        }
        labelingState.annotations[frameKey][tagText] = newState;
    }

    function setupLabelingAutocomplete(container) {
        const input = container.querySelector('.stash-copilot-label-tag-input');
        const dropdown = container.querySelector('.stash-copilot-label-autocomplete-dropdown');
        if (!input || !dropdown) return;

        // Remove old listeners by replacing element
        const newInput = input.cloneNode(true);
        input.parentNode.replaceChild(newInput, input);

        newInput.addEventListener('input', () => {
            const query = newInput.value.trim().toLowerCase();
            if (query.length < 2) {
                dropdown.style.display = 'none';
                return;
            }

            const matches = labelingState.vocabulary
                .filter(tag => tag.toLowerCase().includes(query))
                .slice(0, 10);

            if (matches.length === 0) {
                dropdown.innerHTML = `
                    <div class="stash-copilot-label-ac-item stash-copilot-label-ac-new"
                         data-tag="${escapeHtml(newInput.value.trim())}">
                        + Create: "${escapeHtml(newInput.value.trim())}"
                    </div>
                `;
            } else {
                dropdown.innerHTML = matches.map(tag =>
                    `<div class="stash-copilot-label-ac-item" data-tag="${escapeHtml(tag)}">
                        ${escapeHtml(tag)}
                    </div>`
                ).join('');

                if (!matches.some(t => t.toLowerCase() === query)) {
                    dropdown.innerHTML += `
                        <div class="stash-copilot-label-ac-item stash-copilot-label-ac-new"
                             data-tag="${escapeHtml(newInput.value.trim())}">
                            + Create: "${escapeHtml(newInput.value.trim())}"
                        </div>
                    `;
                }
            }

            dropdown.style.display = 'block';

            dropdown.querySelectorAll('.stash-copilot-label-ac-item').forEach(item => {
                item.addEventListener('click', () => {
                    addManualTag(container, item.dataset.tag);
                    newInput.value = '';
                    dropdown.style.display = 'none';
                });
            });
        });

        newInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const highlighted = dropdown.querySelector('.stash-copilot-label-ac-item.highlighted');
                if (highlighted) {
                    addManualTag(container, highlighted.dataset.tag);
                } else if (dropdown.style.display === 'block' && dropdown.querySelector('.stash-copilot-label-ac-item')) {
                    const firstItem = dropdown.querySelector('.stash-copilot-label-ac-item');
                    addManualTag(container, firstItem.dataset.tag);
                } else if (newInput.value.trim().length > 0) {
                    addManualTag(container, newInput.value.trim());
                }
                newInput.value = '';
                dropdown.style.display = 'none';
            } else if (e.key === 'ArrowDown' && dropdown.style.display === 'block') {
                e.preventDefault();
                const items = dropdown.querySelectorAll('.stash-copilot-label-ac-item');
                const current = dropdown.querySelector('.highlighted');
                if (current) {
                    current.classList.remove('highlighted');
                    const next = current.nextElementSibling || items[0];
                    next.classList.add('highlighted');
                } else if (items.length) {
                    items[0].classList.add('highlighted');
                }
            } else if (e.key === 'ArrowUp' && dropdown.style.display === 'block') {
                e.preventDefault();
                const items = dropdown.querySelectorAll('.stash-copilot-label-ac-item');
                const current = dropdown.querySelector('.highlighted');
                if (current) {
                    current.classList.remove('highlighted');
                    const prev = current.previousElementSibling || items[items.length - 1];
                    prev.classList.add('highlighted');
                } else if (items.length) {
                    items[items.length - 1].classList.add('highlighted');
                }
            } else if (e.key === 'Escape') {
                dropdown.style.display = 'none';
            }
        });

        newInput.addEventListener('blur', () => {
            setTimeout(() => { dropdown.style.display = 'none'; }, 200);
        });
    }

    function addManualTag(container, tagText) {
        const item = labelingState.batch[labelingState.currentIndex];
        if (!item) return;

        const frameKey = `${item.scene_id}_${item.frame_index}`;
        if (!labelingState.annotations[frameKey]) {
            labelingState.annotations[frameKey] = {};
        }
        labelingState.annotations[frameKey][tagText] = 'confirmed';

        if (!labelingState.vocabulary.includes(tagText)) {
            labelingState.vocabulary.push(tagText);
            labelingState.vocabulary.sort();
        }

        renderCurrentFrame(container);
    }

    // Task 9: Navigation, sync, keyboard shortcuts, grid view, export
    function navigateLabeling(container, direction) {
        const newIndex = labelingState.currentIndex + direction;
        if (newIndex < 0 || newIndex >= labelingState.batch.length) return;

        labelingState.currentIndex = newIndex;
        if (labelingState.viewMode === 'single') {
            renderCurrentFrame(container);
        } else {
            renderGridView(container);
            renderCurrentFrame(container);
        }
    }

    function skipFrame(container) {
        const item = labelingState.batch[labelingState.currentIndex];
        if (!item) return;

        labelingState.pendingSync.push({
            type: 'progress',
            scene_id: item.scene_id,
            frame_index: item.frame_index,
            status: 'skipped',
        });

        navigateLabeling(container, 1);
        maybeSyncAnnotations();
    }

    function saveAndNext(container) {
        const item = labelingState.batch[labelingState.currentIndex];
        if (!item) return;

        const frameKey = `${item.scene_id}_${item.frame_index}`;
        const annotations = labelingState.annotations[frameKey] || {};

        for (const [tagText, label] of Object.entries(annotations)) {
            const suggested = item.suggested_tags.find(s => s.tag_text === tagText);
            labelingState.pendingSync.push({
                type: 'annotation',
                scene_id: item.scene_id,
                frame_index: item.frame_index,
                tag_text: tagText,
                tag_source: suggested ? 'suggested' : 'manual',
                label: label,
                similarity_score: suggested ? suggested.similarity : null,
            });
        }

        labelingState.pendingSync.push({
            type: 'progress',
            scene_id: item.scene_id,
            frame_index: item.frame_index,
            status: 'labeled',
        });

        navigateLabeling(container, 1);
        maybeSyncAnnotations();
    }

    function maybeSyncAnnotations() {
        if (labelingState.pendingSync.length >= 30) {
            syncAnnotationsNow();
        }
    }

    async function syncAnnotationsNow() {
        if (labelingState.pendingSync.length === 0) return;
        if (!labelingState.sessionId) return;

        const items = [...labelingState.pendingSync];
        labelingState.pendingSync = [];

        const annotations = items
            .filter(i => i.type === 'annotation')
            .map(({ type, ...rest }) => rest);
        const progress = items
            .filter(i => i.type === 'progress')
            .map(({ type, ...rest }) => rest);

        const payload = {
            session_id: labelingState.sessionId,
            annotations: annotations,
            progress: progress,
        };

        const requestId = `sync_${Date.now()}`;

        try {
            await runPluginTask('Sync Labeling Annotations', {
                request_id: requestId,
                payload: JSON.stringify(payload),
            });
            log(`Synced ${annotations.length} annotations, ${progress.length} progress updates`);
        } catch (e) {
            log(`Sync failed: ${e.message}`, 'error');
            labelingState.pendingSync.push(...items);
        }
    }

    function toggleViewMode(container) {
        labelingState.viewMode = labelingState.viewMode === 'single' ? 'grid' : 'single';
        const singleView = container.querySelector('.stash-copilot-label-single');
        const gridView = container.querySelector('.stash-copilot-label-grid');
        const toggle = container.querySelector('.stash-copilot-label-view-toggle');

        if (labelingState.viewMode === 'single') {
            singleView.style.display = 'flex';
            gridView.style.display = 'none';
            if (toggle) {
                const single = toggle.querySelector('.view-single');
                const grid = toggle.querySelector('.view-grid');
                if (single) single.classList.add('active');
                if (grid) grid.classList.remove('active');
            }
            renderCurrentFrame(container);
        } else {
            singleView.style.display = 'none';
            gridView.style.display = 'flex';
            if (toggle) {
                const single = toggle.querySelector('.view-single');
                const grid = toggle.querySelector('.view-grid');
                if (single) single.classList.remove('active');
                if (grid) grid.classList.add('active');
            }
            renderGridView(container);
        }
    }

    function renderGridView(container) {
        const gridImages = container.querySelector('.stash-copilot-label-grid-images');
        if (!gridImages) return;

        const startIdx = labelingState.currentIndex;
        const endIdx = Math.min(startIdx + 6, labelingState.batch.length);

        gridImages.innerHTML = '';
        for (let i = startIdx; i < endIdx; i++) {
            const item = labelingState.batch[i];
            const frameKey = `${item.scene_id}_${item.frame_index}`;
            const hasAnnotations = !!labelingState.annotations[frameKey];
            const isSelected = i === labelingState.currentIndex;
            const framePath = item.frame_path.replace(/^.*?assets\//, '/plugin/stash-copilot/assets/');

            const cell = document.createElement('div');
            cell.className = `stash-copilot-label-grid-cell${isSelected ? ' selected' : ''}${hasAnnotations ? ' labeled' : ''}`;
            cell.dataset.index = i;
            cell.innerHTML = `
                <img src="${framePath}" alt="Frame ${item.frame_index}">
                <div class="stash-copilot-label-grid-overlay">
                    <span>${escapeHtml(item.scene_title)}</span>
                    <span>${item.timestamp}</span>
                </div>
            `;
            cell.addEventListener('click', () => {
                labelingState.currentIndex = i;
                renderGridView(container);
                renderCurrentFrame(container);
            });
            gridImages.appendChild(cell);
        }
    }

    async function exportDataset(container) {
        await syncAnnotationsNow();

        const requestId = `export_${Date.now()}`;
        const exportBtn = container.querySelector('.stash-copilot-label-export-btn');
        if (exportBtn) {
            exportBtn.disabled = true;
            exportBtn.textContent = 'Exporting...';
        }

        try {
            await runPluginTask('Export Labeling Dataset', {
                request_id: requestId,
                include_negatives: 'true',
            });

            for (let i = 0; i < 60; i++) {
                try {
                    const resp = await fetch(
                        `/plugin/stash-copilot/assets/labeling_export_${requestId}.json?t=${Date.now()}`,
                        { cache: 'no-store' }
                    );
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.status === 'complete') {
                            alert(`Exported ${data.total_images} images with ${data.total_tags} tags to:\n${data.export_path}`);
                            break;
                        } else if (data.status === 'error') {
                            alert(`Export failed: ${data.error}`);
                            break;
                        }
                    }
                } catch (e) { /* not ready */ }
                await new Promise(r => setTimeout(r, 1000));
            }
        } catch (e) {
            alert(`Export failed: ${e.message}`);
        } finally {
            if (exportBtn) {
                exportBtn.disabled = false;
                exportBtn.textContent = 'Export';
            }
        }
    }

    function handleLabelingKeyboard(e, container) {
        if (!document.querySelector('.stash-copilot-label-page')) return;

        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            if (e.key === 'Escape') {
                e.target.blur();
                e.preventDefault();
            }
            return;
        }

        switch (e.key) {
            case 'ArrowRight':
            case 'd':
            case 'D':
                navigateLabeling(container, 1);
                e.preventDefault();
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                navigateLabeling(container, -1);
                e.preventDefault();
                break;
            case 'Enter':
                saveAndNext(container);
                e.preventDefault();
                break;
            case 's':
            case 'S':
                skipFrame(container);
                e.preventDefault();
                break;
            case '/':
                const tagInput = container.querySelector('.stash-copilot-label-tag-input');
                if (tagInput) tagInput.focus();
                e.preventDefault();
                break;
            case 'g':
            case 'G':
                toggleViewMode(container);
                e.preventDefault();
                break;
            case '1': case '2': case '3': case '4': case '5':
            case '6': case '7': case '8': case '9':
                toggleSuggestedTagByIndex(container, parseInt(e.key, 10) - 1);
                e.preventDefault();
                break;
        }
    }

    function toggleSuggestedTagByIndex(container, index) {
        const rows = container.querySelectorAll('.stash-copilot-label-tag-row');
        if (index < rows.length) {
            const item = labelingState.batch[labelingState.currentIndex];
            const frameKey = `${item.scene_id}_${item.frame_index}`;
            toggleTagState(rows[index], frameKey);
        }
    }

    // Sync on page unload
    window.addEventListener('beforeunload', () => {
        if (labelingState.pendingSync.length > 0) {
            syncAnnotationsNow();
        }
    });

    function updateProgress(container) {
        const total = labelingState.batch.length;
        const current = labelingState.currentIndex + 1;
        const labeled = Object.keys(labelingState.annotations).length;
        const pct = total > 0 ? Math.round((labeled / total) * 100) : 0;

        const posEl = container.querySelector('.stash-copilot-label-position');
        const progressText = container.querySelector('.stash-copilot-label-progress-text');
        const progressFill = container.querySelector('.stash-copilot-label-progress-fill');

        if (posEl) posEl.textContent = `${current} / ${total}`;
        if (progressText) progressText.textContent = `Session: ${labeled}/${total} labeled (${pct}%)`;
        if (progressFill) progressFill.style.width = `${pct}%`;
    }

    /**
     * Render classification badges from multi-stage analysis
     */
    function renderClassificationBadges(classification) {
        if (!classification) return '';

        const badges = [];

        // Scene type badge with tooltip
        const sceneTypeIcons = {
            'solo_female': '\u{1F464}',
            'solo_male': '\u{1F464}',
            'couple': '\u{1F46B}',
            'threesome': '\u{1F465}',
            'group': '\u{1F465}\u{1F465}',
        };
        const sceneTypeTooltips = {
            'solo_female': 'Scene type: Single female performer',
            'solo_male': 'Scene type: Single male performer',
            'couple': 'Scene type: Two performers',
            'threesome': 'Scene type: Three performers',
            'group': 'Scene type: Four or more performers',
        };
        const sceneIcon = sceneTypeIcons[classification.scene_type] || '\u{1F3AC}';
        const sceneLabel = classification.scene_type?.replace(/_/g, ' ') || 'Unknown';
        const sceneTooltip = sceneTypeTooltips[classification.scene_type] || `Scene type: ${sceneLabel}`;
        badges.push(`<span class="stash-copilot-badge scene-type" title="${sceneTooltip}">${sceneIcon} ${sceneLabel}</span>`);

        // Content type badge with tooltip
        if (classification.content_type === 'animated') {
            badges.push(`<span class="stash-copilot-badge content-type" title="Content type: Animated/cartoon/hentai/3D">\u{1F3A8} Animated</span>`);
        } else if (classification.content_type === 'live_action') {
            badges.push(`<span class="stash-copilot-badge content-type" title="Content type: Real people, live-action footage">\u{1F3AC} Live Action</span>`);
        }

        // Performer count with tooltip
        if (classification.performer_count) {
            const countTooltip = `Performer count: ${classification.performer_count} performer${classification.performer_count > 1 ? 's' : ''} detected`;
            badges.push(`<span class="stash-copilot-badge performer-count" title="${countTooltip}">\u{1F464}\u{00D7}${classification.performer_count}</span>`);
        }

        // Setting with tooltip
        const settingIcons = {
            'indoor_only': '\u{1F3E0}',
            'outdoor_only': '\u{1F333}',
            'outdoor_to_indoor': '\u{1F333}\u{2192}\u{1F3E0}',
            'indoor_to_outdoor': '\u{1F3E0}\u{2192}\u{1F333}',
            'mixed': '\u{1F3E0}\u{1F333}',
        };
        const settingTooltips = {
            'indoor_only': 'Setting: Entirely indoors',
            'outdoor_only': 'Setting: Entirely outdoors',
            'outdoor_to_indoor': 'Setting: Starts outdoors, moves indoors',
            'indoor_to_outdoor': 'Setting: Starts indoors, moves outdoors',
            'mixed': 'Setting: Mix of indoor and outdoor locations',
        };
        if (classification.setting_progression && settingIcons[classification.setting_progression]) {
            const settingTooltip = settingTooltips[classification.setting_progression] || 'Setting';
            badges.push(`<span class="stash-copilot-badge setting" title="${settingTooltip}">${settingIcons[classification.setting_progression]}</span>`);
        }

        // Activities (multiple badges, text only)
        // Support both new 'activities' array and legacy 'primary_activity' string
        let activities = classification.activities;
        if (!activities && classification.primary_activity && classification.primary_activity !== 'unknown') {
            // Backwards compatibility: convert single primary_activity to array
            activities = [classification.primary_activity];
        }
        if (activities && Array.isArray(activities) && activities.length > 0) {
            activities.forEach(act => {
                if (act && act !== 'unknown') {
                    const activityLabel = act.charAt(0).toUpperCase() + act.slice(1).replace(/_/g, ' ');
                    const activityTooltip = `Activity: ${activityLabel}`;
                    badges.push(`<span class="stash-copilot-badge activity" title="${activityTooltip}">${activityLabel}</span>`);
                }
            });
        }

        return `<div class="stash-copilot-classification-badges">${badges.join('')}</div>`;
    }

    // Separate function to render tags (extracted from renderVisionResult)
    function renderVisionTags(suggestedTags, tagTimestamps, tagsDiv, tagsSection, tagSources = {}, tagConfidences = {}) {
        // Preserve collapsed state if re-rendering
        const wasCollapsed = tagsSection?.classList.contains('collapsed') || false;

        const hasAnyTimestamps = Object.keys(tagTimestamps).length > 0;
        const hasSources = Object.keys(tagSources).length > 0;
        const hasConfidences = Object.keys(tagConfidences).length > 0;

        // Helper to get CSS class based on confidence level
        const getConfidenceClass = (confidence) => {
            if (confidence >= 70) return 'confidence-high';
            if (confidence >= 50) return 'confidence-medium';
            return 'confidence-low';
        };

        if (suggestedTags && suggestedTags.length > 0) {
            tagsSection.style.display = 'block';
            // Restore collapsed state if it was collapsed before
            if (wasCollapsed) {
                tagsSection.classList.add('collapsed');
            }

            // Sort tags by confidence if available (highest first)
            let sortedTags = [...suggestedTags];
            if (hasConfidences) {
                sortedTags.sort((a, b) => {
                    const confA = tagConfidences[a] || 50;
                    const confB = tagConfidences[b] || 50;
                    return confB - confA;
                });
            }

            // Build tags HTML with source and confidence indicators
            const tagsHtml = sortedTags.map(tag => {
                const timestamp = tagTimestamps[tag];
                const hasTimestamp = timestamp !== undefined && timestamp !== null;
                const timeStr = hasTimestamp ? formatTimestamp(timestamp) : '';

                // Get confidence (default 50 if not specified)
                const confidence = tagConfidences[tag] || 50;
                const confidenceClass = getConfidenceClass(confidence);
                const confidenceOpacity = 0.5 + (confidence / 100) * 0.5; // 0.5-1.0 opacity range

                // Get source indicator
                const source = tagSources[tag] || 'llm';
                let sourceClass = '';
                let sourceIndicator = '';
                let sourceTitle = '';

                if (hasSources) {
                    if (source === 'similar') {
                        sourceClass = 'source-similar';
                        sourceIndicator = ' <span class="stash-copilot-tag-source" title="From similar scenes">⋈</span>';
                        sourceTitle = ' (from similar scenes)';
                    } else if (source === 'both') {
                        sourceClass = 'source-both';
                        sourceIndicator = ' <span class="stash-copilot-tag-source" title="Suggested by LLM and found in similar scenes">✓⋈</span>';
                        sourceTitle = ' (LLM + similar scenes)';
                    }
                }

                // Build confidence badge
                const confidenceBadge = hasConfidences
                    ? `<span class="tag-confidence ${confidenceClass}" title="Confidence: ${confidence}%">${confidence}%</span>`
                    : '';

                return `
                    <span class="stash-copilot-vision-tag ${hasTimestamp ? 'has-timestamp' : ''} ${sourceClass} ${confidenceClass}"
                          data-tag="${escapeHtml(tag)}"
                          data-timestamp="${hasTimestamp ? timestamp : ''}"
                          data-confidence="${confidence}"
                          data-source="${source}"
                          style="--confidence-opacity: ${confidenceOpacity}"
                          title="${hasTimestamp ? 'Click timestamp to seek | Shift+click or right-click to apply tag' : 'Click to apply tag'}${sourceTitle} (${confidence}% confident)">
                        ${confidenceBadge}${escapeHtml(tag)}${hasTimestamp ? ` <span class="stash-copilot-tag-timestamp" data-seconds="${timestamp}">@${timeStr}</span>` : ''}${sourceIndicator}
                    </span>
                `;
            }).join('');

            // Build comprehensive legend
            let legendHtml = '<div class="stash-copilot-tags-legend">';
            legendHtml += '<div class="stash-copilot-legend-title">Legend</div>';
            legendHtml += '<div class="stash-copilot-legend-grid">';

            // Confidence section
            if (hasConfidences) {
                legendHtml += '<div class="stash-copilot-legend-section">';
                legendHtml += '<div class="stash-copilot-legend-section-title">Confidence</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-badge confidence-high">85%</span> High certainty (70%+)</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-badge confidence-medium">60%</span> Likely present (50-69%)</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-badge confidence-low">35%</span> Uncertain (&lt;50%)</div>';
                legendHtml += '</div>';
            }

            // Source section
            if (hasSources) {
                legendHtml += '<div class="stash-copilot-legend-section">';
                legendHtml += '<div class="stash-copilot-legend-section-title">Tag Source</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-tag source-llm">tag</span> From AI vision analysis</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-tag source-similar">tag <span class="stash-copilot-tag-source">⋈</span></span> From similar scenes</div>';
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-tag source-both">tag <span class="stash-copilot-tag-source">✓⋈</span></span> AI + similar scenes</div>';
                legendHtml += '</div>';
            }

            // Actions section
            legendHtml += '<div class="stash-copilot-legend-section">';
            legendHtml += '<div class="stash-copilot-legend-section-title">Actions</div>';
            legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-icon">👆</span> Click tag to apply it</div>';
            if (hasAnyTimestamps) {
                legendHtml += '<div class="stash-copilot-legend-item"><span class="stash-copilot-legend-icon">⏱️</span> Click <span class="stash-copilot-legend-timestamp">@0:45</span> to seek</div>';
            }
            legendHtml += '</div>';

            legendHtml += '</div></div>';

            tagsDiv.innerHTML = tagsHtml + legendHtml;

            // Add click handlers for tags
            tagsDiv.querySelectorAll('.stash-copilot-vision-tag').forEach(tagEl => {
                // Click on timestamp span to seek
                const timestampSpan = tagEl.querySelector('.stash-copilot-tag-timestamp');
                if (timestampSpan) {
                    timestampSpan.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const seconds = parseFloat(timestampSpan.dataset.seconds);
                        if (!isNaN(seconds)) {
                            seekToTimestamp(seconds);
                        }
                    });
                }

                tagEl.addEventListener('click', (e) => {
                    // If clicking on timestamp span, it's handled above
                    if (e.target.classList.contains('stash-copilot-tag-timestamp')) {
                        return;
                    }

                    const timestamp = tagEl.dataset.timestamp;
                    if (timestamp && e.shiftKey) {
                        // Shift+click to apply tag
                        applyTag(tagEl.dataset.tag);
                    } else if (!timestamp) {
                        // No timestamp, apply tag
                        applyTag(tagEl.dataset.tag);
                    } else {
                        // Has timestamp but not shift - also apply tag (timestamp click handled by span)
                        applyTag(tagEl.dataset.tag);
                    }
                });

                // Right-click to apply tag
                tagEl.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    applyTag(tagEl.dataset.tag);
                });
            });
        } else {
            tagsSection.style.display = 'block';
            tagsDiv.innerHTML = '<span class="stash-copilot-vision-no-tags">No additional tags suggested</span>';
        }

        // Add collapse toggle handler for tags section
        const tagsHeader = tagsSection.querySelector('.stash-copilot-section-header');
        if (tagsHeader && !tagsHeader.dataset.listenerAdded) {
            tagsHeader.dataset.listenerAdded = 'true';
            tagsHeader.addEventListener('click', () => {
                tagsSection.classList.toggle('collapsed');
            });
        }
    }

    // Show hosted provider confirmation dialog
    function showHostedProviderConfirmation(data, sceneId) {
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');
        const tagsSection = modal.querySelector('.stash-copilot-vision-tags-section');

        // Hide tags section
        if (tagsSection) {
            tagsSection.style.display = 'none';
        }

        const frameCount = data.calculated_frame_count || data.frame_count || 0;
        const maxFrames = data.max_frames || 10;
        const provider = data.provider || 'hosted provider';
        const reason = data.confirmation_reason || `Frame count (${frameCount}) exceeds the ${maxFrames}-frame limit for hosted providers.`;

        analysisDiv.innerHTML = `
            <div class="stash-copilot-vision-confirmation">
                <div class="stash-copilot-warning-icon">&#9888;</div>
                <h4>Hosted Provider Warning</h4>
                <p class="stash-copilot-confirmation-message">${escapeHtml(reason)}</p>
                <div class="stash-copilot-confirmation-details">
                    <p><strong>Provider:</strong> ${escapeHtml(provider)}</p>
                    <p><strong>Frames to analyze:</strong> ${frameCount}</p>
                    <p><strong>Configured limit:</strong> ${maxFrames}</p>
                </div>
                <p class="stash-copilot-confirmation-warning">
                    Choose how many frames to analyze:
                </p>
                <div class="stash-copilot-confirmation-actions">
                    <button class="btn btn-primary stash-copilot-confirm-limited" title="Uniformly sample ${maxFrames} frames (includes first and last)">
                        Use ${maxFrames} Frames (Recommended)
                    </button>
                    <button class="btn btn-warning stash-copilot-confirm-proceed" title="Send all frames to the API">
                        Use All ${frameCount} Frames
                    </button>
                    <button class="btn btn-secondary stash-copilot-confirm-cancel">
                        Cancel
                    </button>
                </div>
            </div>
        `;

        // Add event listeners
        modal.querySelector('.stash-copilot-confirm-limited').addEventListener('click', () => {
            proceedWithLimitedFrames(sceneId);
        });

        modal.querySelector('.stash-copilot-confirm-proceed').addEventListener('click', () => {
            proceedWithConfirmation(sceneId);
        });

        modal.querySelector('.stash-copilot-confirm-cancel').addEventListener('click', () => {
            closeVisionModal();
        });
    }

    // Proceed with analysis after user confirmation (all frames)
    async function proceedWithConfirmation(sceneId) {
        log(`User confirmed hosted provider limit for scene ${sceneId} - using ALL frames`);

        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');

        // Reset UI to loading state
        analysisDiv.innerHTML = `
            <div class="stash-copilot-vision-loading">
                <div class="stash-copilot-spinner"></div>
                <span class="stash-copilot-vision-status">Proceeding with all frames...</span>
                <div class="stash-copilot-vision-progress-container">
                    <div class="stash-copilot-vision-progress-bar" style="width: 0%"></div>
                </div>
                <span class="stash-copilot-vision-progress-text">0%</span>
            </div>
        `;

        visionState.isAnalyzing = true;
        visionState.analysisStartTime = Date.now();

        // Call task with user_confirmed flag
        try {
            await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: [
                    { key: 'mode', value: { str: 'scene_vision' } },
                    { key: 'scene_id', value: { str: sceneId } },
                    { key: 'user_confirmed', value: { str: 'true' } }
                ]
            });

            // Resume polling
            pollVisionResult(sceneId);
        } catch (error) {
            log(`Error proceeding with confirmation: ${error}`, 'error');
            showVisionError('Failed to proceed with analysis');
            visionState.isAnalyzing = false;
        }
    }

    // Proceed with analysis using limited (uniformly sampled) frames
    async function proceedWithLimitedFrames(sceneId) {
        log(`User chose limited frames for scene ${sceneId}`);

        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');

        // Reset UI to loading state
        analysisDiv.innerHTML = `
            <div class="stash-copilot-vision-loading">
                <div class="stash-copilot-spinner"></div>
                <span class="stash-copilot-vision-status">Proceeding with sampled frames...</span>
                <div class="stash-copilot-vision-progress-container">
                    <div class="stash-copilot-vision-progress-bar" style="width: 0%"></div>
                </div>
                <span class="stash-copilot-vision-progress-text">0%</span>
            </div>
        `;

        visionState.isAnalyzing = true;
        visionState.analysisStartTime = Date.now();

        // Call task with user_confirmed AND use_limited_frames flags
        try {
            await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: [
                    { key: 'mode', value: { str: 'scene_vision' } },
                    { key: 'scene_id', value: { str: sceneId } },
                    { key: 'user_confirmed', value: { str: 'true' } },
                    { key: 'use_limited_frames', value: { str: 'true' } }
                ]
            });

            // Resume polling
            pollVisionResult(sceneId);
        } catch (error) {
            log(`Error proceeding with limited frames: ${error}`, 'error');
            showVisionError('Failed to proceed with analysis');
            visionState.isAnalyzing = false;
        }
    }

    // Poll for vision analysis results
    async function pollVisionResult(sceneId) {
        const historyFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;

        const checkResult = async () => {
            try {
                // Add cache buster to prevent browser caching
                const cacheBuster = `?t=${Date.now()}`;
                const response = await fetch(historyFile + cacheBuster, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();
                    log(`Vision poll: status=${data.status}, progress=${data.progress}%`);

                    // Check if this result is from after we started (not an old cached result)
                    const resultTime = data.updated_at ? new Date(data.updated_at).getTime() : 0;
                    if (resultTime < visionState.analysisStartTime) {
                        log('Vision poll: result is older than start time, waiting...');
                        return false;
                    }

                    // Check for pending confirmation (hosted provider with too many frames)
                    if (data.status === 'pending_confirmation' || data.pending_confirmation) {
                        log('Hosted provider confirmation required');
                        showHostedProviderConfirmation(data, sceneId);
                        visionState.isAnalyzing = false;
                        return true; // Stop polling
                    }

                    // Update progress display (progressive updates for description/tags)
                    updateVisionProgress(data);

                    // Check if analysis is fully complete (both stages done)
                    const isComplete = (data.stage === 'complete' && data.description_complete && data.tags_complete)
                        || (data.status === 'complete' && (data.description || data.messages?.length > 0));

                    if (isComplete) {
                        // If we have a pending follow-up message, verify it's in the response
                        // This prevents rendering stale data from a previous analysis
                        if (visionState.pendingMessage) {
                            const pendingMsg = visionState.pendingMessage;
                            const messages = data.messages || [];
                            // Check if any user message contains our pending message
                            const hasPendingMessage = messages.some(m =>
                                m.role === 'user' && m.content && m.content.includes(pendingMsg)
                            );
                            if (!hasPendingMessage) {
                                log(`Vision poll: waiting for pending message "${pendingMsg.substring(0, 30)}..." to appear`);
                                return false; // Keep polling
                            }
                            // Clear pending message now that it's confirmed
                            visionState.pendingMessage = null;
                        }

                        visionState.isAnalyzing = false;
                        visionState.conversationId = data.conversation_id;
                        visionState.description = data.description;
                        visionState.suggestedTags = data.suggested_tags || [];
                        visionState.tagTimestamps = data.tag_timestamps || {};
                        visionState.messages = data.messages || [];

                        log('Vision analysis complete, rendering result');
                        renderVisionResult(data);
                        return true;
                    }

                    // Check for error status
                    if (data.status === 'error') {
                        showVisionError(data.status_message || 'Analysis failed');
                        visionState.isAnalyzing = false;
                        visionState.pendingMessage = null;
                        return true;
                    }
                } else {
                    log(`Vision poll: fetch returned ${response.status}`);
                }
            } catch (e) {
                // File may not exist yet
                log(`Vision poll: error - ${e.message}`);
            }
            return false;
        };

        // Poll every 500ms for up to 2 minutes
        const maxAttempts = 240;
        let attempts = 0;

        const poll = async () => {
            if (!visionState.modalOpen) return; // Stop if modal closed

            const done = await checkResult();
            if (!done && attempts < maxAttempts) {
                attempts++;
                setTimeout(poll, 500);
            } else if (!done) {
                showVisionError('Analysis timed out. The vision model may be slow to respond.');
                visionState.isAnalyzing = false;
                visionState.pendingMessage = null;
            }
        };

        poll();
    }

    // Render vision analysis result
    function renderVisionResult(data) {
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');
        const tagsSection = modal.querySelector('.stash-copilot-vision-tags-section');
        const tagsDiv = modal.querySelector('.stash-copilot-vision-tags');
        const messagesDiv = modal.querySelector('.stash-copilot-vision-messages');

        // Get description - either from description field or from first assistant message
        let description = data.description;
        let suggestedTags = data.suggested_tags || [];

        // If description is null/empty, try to extract from assistant messages
        if (!description && data.messages) {
            const assistantMsg = data.messages.find(m => m.role === 'assistant' && m.content);
            if (assistantMsg) {
                const content = assistantMsg.content;
                const suggestedMatch = content.match(/SUGGESTED_TAGS:\s*(.+?)(?:\n|$)/i);

                // Find where description ends
                let descEnd = suggestedMatch ? suggestedMatch.index : content.length;
                description = content.substring(0, descEnd).trim();

                // Extract tags if not already set
                if (suggestedTags.length === 0 && suggestedMatch) {
                    suggestedTags = suggestedMatch[1].split(',').map(t => t.trim().toLowerCase()).filter(t => t);
                }
            }
        }

        // Render classification badges if available
        const classificationBadgesHtml = renderClassificationBadges(data.classification);

        // Render description with collapsible functionality
        if (description) {
            // Preserve collapsed state if re-rendering
            const existingWrapper = analysisDiv.querySelector('.stash-copilot-vision-description-wrapper');
            const wasCollapsed = existingWrapper?.classList.contains('collapsed') || false;

            // Create a preview (first ~100 chars or first 2 lines)
            const plainText = description.replace(/[#*_`]/g, '').trim();
            const previewLines = plainText.split('\n').slice(0, 2);
            let preview = previewLines.join(' ').substring(0, 120);
            if (plainText.length > 120) preview += '...';

            analysisDiv.innerHTML = `
                ${classificationBadgesHtml}
                <div class="stash-copilot-vision-description-wrapper${wasCollapsed ? ' collapsed' : ''}">
                    <div class="stash-copilot-section-header" data-section="description">
                        <h4>Scene Analysis</h4>
                        <button class="stash-copilot-collapse-btn" title="Toggle visibility">
                            <span class="stash-copilot-collapse-icon">▼</span>
                        </button>
                    </div>
                    <div class="stash-copilot-vision-description-preview">${preview}</div>
                    <div class="stash-copilot-vision-description">
                        ${renderMarkdown(description)}
                    </div>
                </div>
            `;

            // Add toggle handler
            const wrapper = analysisDiv.querySelector('.stash-copilot-vision-description-wrapper');
            const header = wrapper.querySelector('.stash-copilot-section-header');
            header.addEventListener('click', () => {
                wrapper.classList.toggle('collapsed');
            });
        } else {
            analysisDiv.innerHTML = `
                <div class="stash-copilot-vision-error">
                    <p>Analysis completed but no description was returned.</p>
                </div>
            `;
        }

        // Render suggested tags (filtered to only show valid tags not already on scene)
        // Use the shared renderVisionTags function
        const tagTimestamps = data.tag_timestamps || {};
        const tagSources = data.tag_sources || {};
        const tagConfidences = data.tag_confidences || {};
        renderVisionTags(suggestedTags, tagTimestamps, tagsDiv, tagsSection, tagSources, tagConfidences);

        // Render follow-up messages (skip the initial prompt/response)
        const modalContent = modal.querySelector('.stash-copilot-vision-content');
        if (data.messages && data.messages.length > 2) {
            const followUpMessages = data.messages.slice(2);
            if (followUpMessages.length > 0) {
                messagesDiv.innerHTML = followUpMessages.map(msg => `
                    <div class="stash-copilot-vision-message ${msg.role}">
                        <div class="stash-copilot-vision-message-content">${renderMarkdown(msg.content || '')}</div>
                    </div>
                `).join('');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;

                // Expand chat area when there are follow-up messages
                messagesDiv.classList.add('has-messages');
                if (modalContent) {
                    modalContent.classList.add('stash-copilot-chat-expanded');
                }
            }
        } else {
            // Remove expanded classes if no follow-up messages
            messagesDiv.classList.remove('has-messages');
            if (modalContent) {
                modalContent.classList.remove('stash-copilot-chat-expanded');
            }
        }

        // Render debug info if available
        if (data.debug_info) {
            renderDebugInfo(modal, data.debug_info, visionState.sceneId);
        }

        // Display suggested question in chat input
        if (data.suggested_question) {
            const chatInput = modal.querySelector('.stash-copilot-vision-input');
            const inputContainer = modal.querySelector('.stash-copilot-vision-input-container');
            if (chatInput && inputContainer) {
                chatInput.value = data.suggested_question;
                chatInput.classList.add('has-suggestion');
                inputContainer.classList.add('has-suggestion');

                // Add sparkle indicator if not already present
                if (!inputContainer.querySelector('.suggestion-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'stash-copilot-suggestion-indicator';
                    indicator.innerHTML = '✨';
                    indicator.title = 'AI-suggested question (click to use or type your own)';
                    inputContainer.insertBefore(indicator, chatInput);
                }

                // Clear suggestion styling when user starts typing
                const clearSuggestion = () => {
                    chatInput.classList.remove('has-suggestion');
                    inputContainer.classList.remove('has-suggestion');
                    const indicator = inputContainer.querySelector('.suggestion-indicator');
                    if (indicator) indicator.remove();
                    chatInput.removeEventListener('input', clearSuggestion);
                    chatInput.removeEventListener('focus', selectAll);
                };

                // Select all text on focus so user can easily replace
                const selectAll = () => {
                    chatInput.select();
                };

                chatInput.addEventListener('input', clearSuggestion);
                chatInput.addEventListener('focus', selectAll);
            }
        }
    }

    // Render debug information panel
    function renderDebugInfo(modal, debugInfo, sceneId) {
        const debugSection = modal.querySelector('.stash-copilot-vision-debug');
        if (!debugSection || !debugInfo) return;

        // Only show debug section if there's actual meaningful debug data
        // (check for populated token counts which indicate debug mode was enabled)
        const hasDebugData = debugInfo.description_system_tokens > 0 ||
                             debugInfo.description_prompt_tokens > 0 ||
                             debugInfo.description_system_prompt;
        if (!hasDebugData) {
            debugSection.style.display = 'none';
            return;
        }

        // Show the debug section
        debugSection.style.display = 'block';

        // Stage 1: Description
        const descStage = debugSection.querySelector('[data-stage="description"]');
        if (descStage) {
            // Token info
            const descTokens = descStage.querySelector('.stash-copilot-debug-tokens');
            const totalTextTokens = (debugInfo.description_system_tokens || 0) + (debugInfo.description_prompt_tokens || 0);
            const totalInputTokens = totalTextTokens + (debugInfo.description_image_tokens || 0);
            descTokens.innerHTML = `
                <span>Text: ~${totalTextTokens.toLocaleString()} tokens</span>
                <span>Images: ~${(debugInfo.description_image_tokens || 0).toLocaleString()} tokens (${debugInfo.description_frame_count} frames)</span>
                <span>Response: ~${(debugInfo.description_response_tokens || 0).toLocaleString()} tokens</span>
                <span>Duration: ${((debugInfo.description_duration_ms || 0) / 1000).toFixed(1)}s</span>
            `;

            // Prompts
            descStage.querySelector('.stash-copilot-debug-system-prompt').textContent = debugInfo.description_system_prompt || '';
            descStage.querySelector('.stash-copilot-debug-user-prompt').textContent = debugInfo.description_user_prompt || '';

            // Frame count in summary
            const frameCountSpan = descStage.querySelector('.frame-count');
            if (frameCountSpan) {
                frameCountSpan.textContent = debugInfo.description_frame_count || 0;
            }

            // Frame thumbnails
            const framesDiv = descStage.querySelector('.stash-copilot-debug-frames');
            if (framesDiv && sceneId && debugInfo.description_frame_count > 0) {
                const frameCount = debugInfo.description_frame_count;
                const frameSizes = debugInfo.description_frame_sizes || [];
                let framesHtml = '';
                for (let i = 1; i <= frameCount; i++) {
                    const frameUrl = `/plugin/stash-copilot/assets/embedded_frames/scene_${sceneId}/frame_${String(i).padStart(4, '0')}.jpg`;
                    const sizeKb = frameSizes[i - 1] ? Math.round(frameSizes[i - 1] * 0.75 / 1024) : '?';
                    framesHtml += `<img src="${frameUrl}" class="stash-copilot-debug-frame" title="Frame ${i} (~${sizeKb}KB)" loading="lazy">`;
                }
                framesDiv.innerHTML = framesHtml;
            }
        }

        // Stage 2: Tagging
        const tagStage = debugSection.querySelector('[data-stage="tagging"]');
        if (tagStage) {
            const tagTokens = tagStage.querySelector('.stash-copilot-debug-tokens');
            const totalTagTextTokens = (debugInfo.tag_system_tokens || 0) + (debugInfo.tag_prompt_tokens || 0);
            tagTokens.innerHTML = `
                <span>Text: ~${totalTagTextTokens.toLocaleString()} tokens</span>
                <span>Response: ~${(debugInfo.tag_response_tokens || 0).toLocaleString()} tokens</span>
                <span>Duration: ${((debugInfo.tag_duration_ms || 0) / 1000).toFixed(1)}s</span>
            `;

            tagStage.querySelector('.stash-copilot-debug-system-prompt').textContent = debugInfo.tag_system_prompt || '';
            tagStage.querySelector('.stash-copilot-debug-user-prompt').textContent = debugInfo.tag_user_prompt || '';
        }

        // Totals
        const totalsDiv = debugSection.querySelector('.stash-copilot-debug-totals');
        if (totalsDiv) {
            const descTextTokens = (debugInfo.description_system_tokens || 0) + (debugInfo.description_prompt_tokens || 0);
            const descImageTokens = debugInfo.description_image_tokens || 0;
            const descResponseTokens = debugInfo.description_response_tokens || 0;
            const tagTextTokens = (debugInfo.tag_system_tokens || 0) + (debugInfo.tag_prompt_tokens || 0);
            const tagResponseTokens = debugInfo.tag_response_tokens || 0;
            const totalTokens = descTextTokens + descImageTokens + descResponseTokens + tagTextTokens + tagResponseTokens;
            const totalDuration = ((debugInfo.description_duration_ms || 0) + (debugInfo.tag_duration_ms || 0)) / 1000;

            totalsDiv.innerHTML = `
                <div class="stash-copilot-debug-total-line">
                    <strong>Total:</strong>
                    ~${totalTokens.toLocaleString()} tokens |
                    ${totalDuration.toFixed(1)}s
                </div>
            `;
        }

        // Populate prompt editor textareas with current prompts
        // Use localStorage values if set, otherwise use the prompts from debug info
        const customPrompts = loadCustomPrompts();
        const systemTextarea = modal.querySelector('.stash-copilot-prompt-system');
        const descTextarea = modal.querySelector('.stash-copilot-prompt-description');

        if (systemTextarea) {
            systemTextarea.value = customPrompts?.system_prompt || debugInfo.description_system_prompt || '';
        }
        if (descTextarea) {
            descTextarea.value = customPrompts?.description_prompt || debugInfo.description_user_prompt || '';
        }
    }

    // Show error in vision modal
    function showVisionError(message) {
        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const analysisDiv = modal.querySelector('.stash-copilot-vision-analysis');
        analysisDiv.innerHTML = `
            <div class="stash-copilot-vision-error">
                <p>${message}</p>
                <button class="btn btn-secondary" onclick="document.getElementById('stash-copilot-vision-modal').querySelector('.stash-copilot-vision-close').click()">Close</button>
            </div>
        `;
    }

    // Send follow-up message
    async function sendVisionMessage(message) {
        if (!message.trim() || visionState.isAnalyzing) return;

        const modal = document.getElementById('stash-copilot-vision-modal');
        if (!modal) return;

        const input = modal.querySelector('.stash-copilot-vision-input');
        input.value = '';

        // Set analysisStartTime so pollVisionResult waits for fresh data
        visionState.analysisStartTime = Date.now();
        // Track pending message to ensure it appears in response before rendering
        visionState.pendingMessage = message.trim();

        // Add user message to UI immediately
        const messagesDiv = modal.querySelector('.stash-copilot-vision-messages');
        messagesDiv.innerHTML += `
            <div class="stash-copilot-vision-message user">
                <div class="stash-copilot-vision-message-content">${renderMarkdown(message)}</div>
            </div>
            <div class="stash-copilot-vision-message assistant loading">
                <div class="stash-copilot-spinner"></div>
            </div>
        `;
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        visionState.isAnalyzing = true;

        try {
            await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: [
                    { key: 'mode', value: { str: 'scene_vision' } },
                    { key: 'scene_id', value: { str: visionState.sceneId } },
                    { key: 'message', value: { str: message } },
                    { key: 'conversation_id', value: { str: visionState.conversationId || '' } }
                ]
            });

            // Poll for updated result
            pollVisionResult(visionState.sceneId);
        } catch (error) {
            log(`Vision message error: ${error}`, 'error');
            // Remove loading message and show error
            const loadingMsg = messagesDiv.querySelector('.loading');
            if (loadingMsg) {
                loadingMsg.innerHTML = '<div class="stash-copilot-vision-message-content">Error sending message</div>';
                loadingMsg.classList.remove('loading');
            }
            visionState.isAnalyzing = false;
            visionState.pendingMessage = null;
        }
    }

    // Apply a suggested tag to the scene
    async function applyTag(tagName) {
        if (!visionState.sceneId) return;

        const tagEl = document.querySelector(`.stash-copilot-vision-tag[data-tag="${tagName}"]`);

        // Show applying state
        if (tagEl) {
            tagEl.classList.add('applying');
        }

        log(`Applying tag "${tagName}" to scene ${visionState.sceneId}`);

        try {
            // Find the existing tag
            const findResult = await callGQL(`
                query FindTags($filter: FindFilterType) {
                    findTags(filter: $filter) {
                        tags { id name }
                    }
                }
            `, {
                filter: { q: tagName, per_page: 10 }
            });

            // Find exact match (case-insensitive)
            const matchedTag = findResult?.findTags?.tags?.find(
                t => t.name.toLowerCase() === tagName.toLowerCase()
            );

            if (!matchedTag) {
                log(`Tag "${tagName}" not found in library`, 'error');
                if (tagEl) tagEl.classList.remove('applying');
                return;
            }

            const tagId = matchedTag.id;

            // Get current scene tags
            const sceneResult = await callGQL(`
                query FindScene($id: ID!) {
                    findScene(id: $id) {
                        tags { id }
                    }
                }
            `, { id: visionState.sceneId });

            const currentTagIds = sceneResult?.findScene?.tags?.map(t => t.id) || [];

            // Add tag if not already present
            if (!currentTagIds.includes(tagId)) {
                await callGQL(`
                    mutation SceneUpdate($input: SceneUpdateInput!) {
                        sceneUpdate(input: $input) { id }
                    }
                `, {
                    input: {
                        id: visionState.sceneId,
                        tag_ids: [...currentTagIds, tagId]
                    }
                });

                // Update UI to show tag was applied
                if (tagEl) {
                    tagEl.classList.remove('applying');
                    tagEl.classList.add('applied');
                }

                log(`Tag "${tagName}" applied successfully`);
            } else {
                log(`Tag "${tagName}" already exists on scene`);
                if (tagEl) {
                    tagEl.classList.remove('applying');
                    tagEl.classList.add('applied');
                }
            }
        } catch (error) {
            log(`Error applying tag: ${error}`, 'error');
            if (tagEl) tagEl.classList.remove('applying');
        }
    }

    // Check if auto-analyze is enabled
    async function shouldAutoAnalyze() {
        try {
            const settings = await getPluginSettings();
            // Default to false - require explicit opt-in
            const autoAnalyze = settings?.vision_auto_analyze;
            return autoAnalyze && autoAnalyze.toLowerCase() !== 'false';
        } catch (e) {
            return false;
        }
    }

    // Get plugin settings
    async function getPluginSettings() {
        try {
            const result = await callGQL(`
                query Configuration {
                    configuration {
                        plugins
                    }
                }
            `);
            return result?.configuration?.plugins?.['stash-copilot'] || {};
        } catch (e) {
            return {};
        }
    }

    // ===== End Scene Vision Functions =====

    // ===== Similar Scenes Functions =====

    // Similar scenes state - separate state per tab
    const savedVisualWeight = localStorage.getItem('stash-copilot-visual-weight');
    const similarState = {
        sceneId: null,
        modalOpen: false,
        activeTab: 'all',      // 'all' or 'different-performers'
        resultsPerPage: 12,    // How many to show per page
        fetchBatchSize: 120,   // How many to fetch from backend at once
        visualWeight: savedVisualWeight !== null ? parseFloat(savedVisualWeight) : 0.7,
        // Exclusion filters (applied to all tabs)
        excludePerformers: [],  // Array of performer names
        excludeTags: [],        // Array of tag names
        filtersOpen: false,
        requestId: null,        // Unique ID for current request to avoid stale results
        // Per-tab state
        tabs: {
            'all': {
                allResults: [],       // All fetched results from backend
                allSceneDetails: [],  // All fetched scene details
                currentPage: 1,
                backendOffset: 0,     // Offset for next backend fetch
                hasMoreBackend: true, // Whether backend has more results
                isSearching: false,
                isLoadingPage: false,
                loaded: false,
            },
            'different-performers': {
                allResults: [],
                allSceneDetails: [],
                currentPage: 1,
                backendOffset: 0,
                hasMoreBackend: true,
                isSearching: false,
                isLoadingPage: false,
                loaded: false,
            }
        }
    };

    // ===== Frame Search State =====
    const frameSearchState = {
        active: false,
        results: [],
        requestId: '',
        queryTimestamp: 0,
        pollInterval: null
    };

    // Get current tab state
    function getTabState(tab) {
        return similarState.tabs[tab || similarState.activeTab];
    }

    // Create the similar scenes modal
    function createSimilarModal() {
        // Remove existing modal if present
        const existing = document.getElementById('stash-copilot-similar-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'stash-copilot-similar-modal';
        modal.className = 'stash-copilot-similar-modal';
        modal.innerHTML = `
            <div class="stash-copilot-similar-content">
                <div class="stash-copilot-similar-header">
                    <h3>Similar Scenes</h3>
                    <button class="stash-copilot-similar-close">&times;</button>
                </div>
                <div class="stash-copilot-similar-tabs">
                    <button class="stash-copilot-similar-tab active" data-filter="all">All Similar</button>
                    <button class="stash-copilot-similar-tab" data-filter="different-performers">Different Performers</button>
                    <button class="stash-copilot-similar-filter-toggle" title="Filter options">
                        <svg class="stash-copilot-filter-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                        </svg>
                        <span>Filters</span>
                        <span class="stash-copilot-filter-badge"></span>
                    </button>
                </div>
                <div class="stash-copilot-weight-slider-container">
                    <span class="stash-copilot-weight-slider-endpoint">Metadata</span>
                    <div class="stash-copilot-weight-slider-track">
                        <input type="range" min="0" max="100" value="70" step="5"
                               class="stash-copilot-weight-slider" id="visual-weight-slider">
                    </div>
                    <span class="stash-copilot-weight-slider-endpoint">Visual</span>
                    <span class="stash-copilot-weight-slider-value">70%</span>
                </div>
                <div class="stash-copilot-similar-filters" style="display: none;">
                    <div class="stash-copilot-similar-filter-columns">
                        <div class="stash-copilot-similar-filter-column">
                            <label>Exclude Performers:</label>
                            <div class="stash-copilot-autocomplete-wrapper">
                                <input type="text" class="stash-copilot-similar-filter-input" id="exclude-performers-input"
                                       placeholder="Type to search..." autocomplete="off">
                                <div class="stash-copilot-autocomplete-dropdown" id="performers-autocomplete"></div>
                            </div>
                            <div class="stash-copilot-similar-filter-tags" id="exclude-performers-tags"></div>
                        </div>
                        <div class="stash-copilot-similar-filter-column">
                            <label>Exclude Tags:</label>
                            <div class="stash-copilot-autocomplete-wrapper">
                                <input type="text" class="stash-copilot-similar-filter-input" id="exclude-tags-input"
                                       placeholder="Type to search..." autocomplete="off">
                                <div class="stash-copilot-autocomplete-dropdown" id="tags-autocomplete"></div>
                            </div>
                            <div class="stash-copilot-similar-filter-tags" id="exclude-tags-tags"></div>
                        </div>
                    </div>
                    <div class="stash-copilot-similar-filter-actions">
                        <button class="stash-copilot-similar-filter-apply">Apply Filters</button>
                        <button class="stash-copilot-similar-filter-clear">Clear All</button>
                    </div>
                </div>
                <div class="stash-copilot-similar-body">
                    <div class="stash-copilot-similar-loading">
                        <div class="stash-copilot-spinner"></div>
                        <span class="stash-copilot-similar-status">Finding similar scenes...</span>
                    </div>
                    <div class="stash-copilot-similar-results" style="display: none;"></div>
                    <div class="stash-copilot-similar-pagination" style="display: none;">
                        <button class="stash-copilot-similar-page-btn prev" disabled>&larr; Previous</button>
                        <span class="stash-copilot-similar-page-info">Page 1 of 1</span>
                        <button class="stash-copilot-similar-page-btn next" disabled>Next &rarr;</button>
                    </div>
                    <div class="stash-copilot-similar-empty" style="display: none;">
                        <p>No similar scenes found.</p>
                        <p class="stash-copilot-similar-hint">This scene may not have an embedding yet. Run the "Embed Scene" task first.</p>
                    </div>
                    <div class="stash-copilot-similar-error" style="display: none;"></div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add event listeners
        modal.querySelector('.stash-copilot-similar-close').addEventListener('click', closeSimilarModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeSimilarModal();
        });

        // Tab event listeners
        modal.querySelectorAll('.stash-copilot-similar-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const newTab = tab.dataset.filter;
                if (newTab === similarState.activeTab) return;

                // Update active tab UI
                modal.querySelectorAll('.stash-copilot-similar-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Switch to new tab
                switchToTab(newTab);
            });
        });

        // Pagination event listeners - client-side pagination with lazy backend fetch
        modal.querySelector('.stash-copilot-similar-page-btn.prev').addEventListener('click', () => {
            const tabState = getTabState();
            if (tabState.currentPage > 1 && !tabState.isLoadingPage) {
                tabState.currentPage--;
                renderCurrentPage();  // Instant - already cached
            }
        });
        modal.querySelector('.stash-copilot-similar-page-btn.next').addEventListener('click', () => {
            const tabState = getTabState();
            if (tabState.isLoadingPage) return;

            const nextPageStart = tabState.currentPage * similarState.resultsPerPage;
            const hasNextPageCached = nextPageStart < tabState.allResults.length;
            const canFetchMore = tabState.hasMoreBackend;

            if (hasNextPageCached) {
                // Next page is already cached - instant render
                tabState.currentPage++;
                renderCurrentPage();
            } else if (canFetchMore) {
                // Need to fetch more from backend
                tabState.currentPage++;
                fetchMoreResults();
            }
        });

        // Filter toggle event listener
        modal.querySelector('.stash-copilot-similar-filter-toggle').addEventListener('click', () => {
            similarState.filtersOpen = !similarState.filtersOpen;
            const filtersSection = modal.querySelector('.stash-copilot-similar-filters');
            const toggleBtn = modal.querySelector('.stash-copilot-similar-filter-toggle');
            filtersSection.style.display = similarState.filtersOpen ? 'block' : 'none';
            toggleBtn.classList.toggle('active', similarState.filtersOpen);
        });

        // Visual weight slider event listener with debounce
        const weightSlider = modal.querySelector('#visual-weight-slider');
        const weightValueDisplay = modal.querySelector('.stash-copilot-weight-slider-value');
        let sliderDebounceTimer = null;

        weightSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value, 10);
            // Update display immediately
            weightValueDisplay.textContent = `${value}%`;

            // Debounce the actual search
            if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
            sliderDebounceTimer = setTimeout(() => {
                const newWeight = value / 100;
                if (newWeight !== similarState.visualWeight) {
                    similarState.visualWeight = newWeight;
                    localStorage.setItem('stash-copilot-visual-weight', newWeight);
                    log(`Visual weight changed to: ${newWeight}`);
                    // Reset all tabs and refetch
                    applyExclusionFilters();
                }
            }, 300);
        });

        // Setup autocomplete for performers
        const performerInput = modal.querySelector('#exclude-performers-input');
        const performerDropdown = modal.querySelector('#performers-autocomplete');
        setupAutocomplete(performerInput, performerDropdown, 'performers', (name) => {
            if (!similarState.excludePerformers.includes(name)) {
                similarState.excludePerformers.push(name);
                renderExclusionTags();
            }
        });

        // Setup autocomplete for tags
        const tagInput = modal.querySelector('#exclude-tags-input');
        const tagDropdown = modal.querySelector('#tags-autocomplete');
        setupAutocomplete(tagInput, tagDropdown, 'tags', (name) => {
            if (!similarState.excludeTags.includes(name)) {
                similarState.excludeTags.push(name);
                renderExclusionTags();
            }
        });

        // Apply filters button
        modal.querySelector('.stash-copilot-similar-filter-apply').addEventListener('click', () => {
            applyExclusionFilters();
        });

        // Clear filters button
        modal.querySelector('.stash-copilot-similar-filter-clear').addEventListener('click', () => {
            similarState.excludePerformers = [];
            similarState.excludeTags = [];
            renderExclusionTags();
            applyExclusionFilters();
        });

        return modal;
    }

    // Render the exclusion filter tags
    function renderExclusionTags() {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        // Render performer exclusion tags
        const performerTagsDiv = modal.querySelector('#exclude-performers-tags');
        performerTagsDiv.innerHTML = similarState.excludePerformers.map(name => `
            <span class="stash-copilot-filter-tag" data-type="performer" data-name="${escapeHtml(name)}">
                ${escapeHtml(name)}
                <button class="stash-copilot-filter-tag-remove">&times;</button>
            </span>
        `).join('');

        // Render tag exclusion tags
        const tagTagsDiv = modal.querySelector('#exclude-tags-tags');
        tagTagsDiv.innerHTML = similarState.excludeTags.map(name => `
            <span class="stash-copilot-filter-tag" data-type="tag" data-name="${escapeHtml(name)}">
                ${escapeHtml(name)}
                <button class="stash-copilot-filter-tag-remove">&times;</button>
            </span>
        `).join('');

        // Add click handlers for remove buttons
        modal.querySelectorAll('.stash-copilot-filter-tag-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tag = e.target.closest('.stash-copilot-filter-tag');
                const type = tag.dataset.type;
                const name = tag.dataset.name;
                if (type === 'performer') {
                    similarState.excludePerformers = similarState.excludePerformers.filter(n => n !== name);
                } else {
                    similarState.excludeTags = similarState.excludeTags.filter(n => n !== name);
                }
                renderExclusionTags();
            });
        });

        // Update filter badge count
        const totalFilters = similarState.excludePerformers.length + similarState.excludeTags.length;
        const badge = modal.querySelector('.stash-copilot-filter-badge');
        if (badge) {
            if (totalFilters > 0) {
                badge.textContent = totalFilters;
                badge.classList.add('visible');
            } else {
                badge.classList.remove('visible');
            }
        }
    }

    // Apply exclusion filters - re-fetch results with filters
    async function applyExclusionFilters() {
        // Reset similar scene tab states and re-fetch with new filters
        for (const tab of Object.keys(similarState.tabs)) {
            similarState.tabs[tab].allResults = [];
            similarState.tabs[tab].allSceneDetails = [];
            similarState.tabs[tab].currentPage = 1;
            similarState.tabs[tab].backendOffset = 0;
            similarState.tabs[tab].hasMoreBackend = true;
            similarState.tabs[tab].isSearching = false;
            similarState.tabs[tab].isLoadingPage = false;
            similarState.tabs[tab].loaded = false;
        }
        // Re-fetch current tab
        findSimilarScenes(similarState.sceneId);
    }

    // NOTE: escapeHtml is defined earlier in the file (line ~2813)
    // Removed duplicate definition here

    // Setup autocomplete for an input field
    function setupAutocomplete(input, dropdown, type, onSelect) {
        let debounceTimer = null;
        let selectedIndex = -1;

        // Search function
        async function search(query) {
            if (query.length < 1) {
                dropdown.style.display = 'none';
                return;
            }

            try {
                const gqlQuery = type === 'performers'
                    ? `query FindPerformers($filter: FindFilterType) {
                        findPerformers(filter: $filter) {
                            performers { id name }
                        }
                    }`
                    : `query FindTags($filter: FindFilterType) {
                        findTags(filter: $filter) {
                            tags { id name }
                        }
                    }`;

                const response = await fetch('/graphql', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: gqlQuery,
                        variables: {
                            filter: {
                                q: query,
                                per_page: 10,
                                sort: 'name',
                                direction: 'ASC'
                            }
                        }
                    })
                });

                const data = await response.json();
                const items = type === 'performers'
                    ? data.data?.findPerformers?.performers || []
                    : data.data?.findTags?.tags || [];

                renderDropdown(items);
            } catch (error) {
                log(`Autocomplete search error: ${error}`, 'error');
                dropdown.style.display = 'none';
            }
        }

        // Render dropdown
        function renderDropdown(items) {
            if (items.length === 0) {
                dropdown.style.display = 'none';
                return;
            }

            // Filter out already selected items
            const excluded = type === 'performers'
                ? similarState.excludePerformers
                : similarState.excludeTags;
            const filtered = items.filter(item => !excluded.includes(item.name));

            if (filtered.length === 0) {
                dropdown.style.display = 'none';
                return;
            }

            dropdown.innerHTML = filtered.map((item, idx) => `
                <div class="stash-copilot-autocomplete-item" data-index="${idx}" data-name="${escapeHtml(item.name)}">
                    ${escapeHtml(item.name)}
                </div>
            `).join('');
            dropdown.style.display = 'block';

            // Always highlight first item so user knows what Tab will complete
            selectedIndex = 0;
            updateSelection();

            // Add click handlers
            dropdown.querySelectorAll('.stash-copilot-autocomplete-item').forEach(item => {
                item.addEventListener('click', () => {
                    selectItem(item.dataset.name);
                });
                item.addEventListener('mouseenter', () => {
                    selectedIndex = parseInt(item.dataset.index);
                    updateSelection();
                });
            });
        }

        // Update visual selection
        function updateSelection() {
            dropdown.querySelectorAll('.stash-copilot-autocomplete-item').forEach((item, idx) => {
                item.classList.toggle('selected', idx === selectedIndex);
            });
        }

        // Select an item
        function selectItem(name) {
            onSelect(name);
            input.value = '';
            dropdown.style.display = 'none';
            selectedIndex = -1;
        }

        // Input event handler
        input.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                search(input.value.trim());
            }, 100);
        });

        // Keyboard navigation
        input.addEventListener('keydown', (e) => {
            const items = dropdown.querySelectorAll('.stash-copilot-autocomplete-item');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (dropdown.style.display === 'block' && items.length > 0) {
                    selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                    updateSelection();
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (dropdown.style.display === 'block' && items.length > 0) {
                    selectedIndex = Math.max(selectedIndex - 1, 0);
                    updateSelection();
                }
            } else if (e.key === 'Enter' || e.key === 'Tab') {
                // Tab/Enter completes the highlighted selection
                if (dropdown.style.display === 'block' && items.length > 0 && selectedIndex >= 0) {
                    e.preventDefault();
                    selectItem(items[selectedIndex].dataset.name);
                } else if (e.key === 'Enter' && input.value.trim()) {
                    e.preventDefault();
                    // Allow manual entry if dropdown not showing
                    selectItem(input.value.trim());
                }
            } else if (e.key === 'Escape') {
                dropdown.style.display = 'none';
                selectedIndex = -1;
            }
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!input.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.style.display = 'none';
                selectedIndex = -1;
            }
        });

        // Focus handler
        input.addEventListener('focus', () => {
            if (input.value.trim().length >= 1) {
                search(input.value.trim());
            }
        });
    }

    // Switch to a different tab
    function switchToTab(tab) {
        similarState.activeTab = tab;
        const tabState = getTabState();

        // If this tab hasn't been loaded yet, fetch data
        if (!tabState.loaded) {
            findSimilarScenes(similarState.sceneId);
        } else {
            // Tab already has data, just render current page from cache
            renderTabResults();
        }
    }

    // Render results for current tab (used when switching between loaded tabs)
    function renderTabResults() {
        const tabState = getTabState();
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const emptyDiv = modal.querySelector('.stash-copilot-similar-empty');
        const resultsDiv = modal.querySelector('.stash-copilot-similar-results');
        const loadingDiv = modal.querySelector('.stash-copilot-similar-loading');

        loadingDiv.style.display = 'none';

        if (!tabState.allResults.length) {
            resultsDiv.style.display = 'none';
            emptyDiv.style.display = 'block';
            emptyDiv.innerHTML = similarState.activeTab === 'different-performers'
                ? `<p>No similar scenes with different performers found.</p>
                   <p class="stash-copilot-similar-hint">All similar scenes share performers with this scene.</p>`
                : `<p>No similar scenes found.</p>
                   <p class="stash-copilot-similar-hint">This scene may not have an embedding yet. Run the "Embed Scene" task first.</p>`;
            updatePaginationButtons();
            return;
        }
        renderCurrentPage();
    }

    // Open similar scenes modal
    async function openSimilarModal(sceneId) {
        similarState.sceneId = sceneId;
        similarState.modalOpen = true;
        similarState.activeTab = 'all';

        // Reset all tab states
        for (const tab of Object.keys(similarState.tabs)) {
            similarState.tabs[tab] = {
                allResults: [],
                allSceneDetails: [],
                currentPage: 1,
                backendOffset: 0,
                hasMoreBackend: true,
                isSearching: false,
                isLoadingPage: false,
                loaded: false,
            };
        }

        const modal = createSimilarModal();
        modal.classList.add('open');

        // Sync slider with current visualWeight state
        const weightSlider = modal.querySelector('#visual-weight-slider');
        const weightValueDisplay = modal.querySelector('.stash-copilot-weight-slider-value');
        if (weightSlider && weightValueDisplay) {
            const weightPercent = Math.round(similarState.visualWeight * 100);
            weightSlider.value = weightPercent;
            weightValueDisplay.textContent = `${weightPercent}%`;
        }

        // Start search - fetches first batch
        findSimilarScenes(sceneId);
    }

    // Close similar modal
    function closeSimilarModal() {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (modal) {
            modal.classList.remove('open');
            setTimeout(() => modal.remove(), 300);
        }
        similarState.modalOpen = false;
    }

    // Find similar scenes (initial batch fetch)
    async function findSimilarScenes(sceneId) {
        const tabState = getTabState();
        if (tabState.isSearching) return;

        tabState.isSearching = true;
        const excludePerformers = similarState.activeTab === 'different-performers';

        // Generate unique request ID to track this specific request
        const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        similarState.requestId = requestId;
        log(`Finding scenes similar to: ${sceneId} (batch fetch, tab=${similarState.activeTab}, requestId=${requestId})`);

        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const loadingDiv = modal.querySelector('.stash-copilot-similar-loading');
        const resultsDiv = modal.querySelector('.stash-copilot-similar-results');
        const paginationDiv = modal.querySelector('.stash-copilot-similar-pagination');
        const emptyDiv = modal.querySelector('.stash-copilot-similar-empty');
        const errorDiv = modal.querySelector('.stash-copilot-similar-error');

        // Show loading
        loadingDiv.style.display = 'flex';
        resultsDiv.style.display = 'none';
        paginationDiv.style.display = 'none';
        emptyDiv.style.display = 'none';
        errorDiv.style.display = 'none';

        try {
            // Build task args - fetch a larger batch for client-side pagination
            const taskArgs = {
                scene_id: sceneId,
                limit: String(similarState.fetchBatchSize),
                offset: String(tabState.backendOffset),
                exclude_common_performers: excludePerformers ? 'true' : 'false',
                visual_weight: String(similarState.visualWeight)
            };

            // Add exclusion filters if set
            if (similarState.excludePerformers.length > 0) {
                taskArgs.exclude_performer_names = similarState.excludePerformers.join(',');
            }
            if (similarState.excludeTags.length > 0) {
                taskArgs.exclude_tag_names = similarState.excludeTags.join(',');
            }

            // Include request ID so backend can echo it back for validation
            taskArgs.request_id = requestId;

            // Run the find_similar task
            await runPluginTask('Find Similar Scenes', taskArgs);

            // Poll for results from the embedding database (pass requestId to validate)
            pollSimilarResults(sceneId, requestId);

        } catch (error) {
            log(`Similar search error: ${error}`, 'error');
            loadingDiv.style.display = 'none';
            errorDiv.style.display = 'block';
            errorDiv.innerHTML = `<p>Error: ${escapeHtml(error.message || 'Failed to find similar scenes')}</p>`;
            tabState.isSearching = false;
            tabState.isLoadingPage = false;
        }
    }

    // Fetch more results when user navigates past cached data
    async function fetchMoreResults() {
        const tabState = getTabState();
        if (tabState.isLoadingPage || !tabState.hasMoreBackend) return;

        tabState.isLoadingPage = true;
        const excludePerformers = similarState.activeTab === 'different-performers';

        // Generate unique request ID
        const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        similarState.requestId = requestId;
        log(`Fetching more results (offset=${tabState.backendOffset}, tab=${similarState.activeTab}, requestId=${requestId})`);

        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const resultsDiv = modal.querySelector('.stash-copilot-similar-results');
        const paginationDiv = modal.querySelector('.stash-copilot-similar-pagination');

        // Show loading overlay
        resultsDiv.classList.add('loading');
        paginationDiv.querySelectorAll('button').forEach(btn => btn.disabled = true);

        try {
            const taskArgs = {
                scene_id: similarState.sceneId,
                limit: String(similarState.fetchBatchSize),
                offset: String(tabState.backendOffset),
                exclude_common_performers: excludePerformers ? 'true' : 'false',
                visual_weight: String(similarState.visualWeight)
            };

            if (similarState.excludePerformers.length > 0) {
                taskArgs.exclude_performer_names = similarState.excludePerformers.join(',');
            }
            if (similarState.excludeTags.length > 0) {
                taskArgs.exclude_tag_names = similarState.excludeTags.join(',');
            }

            // Include request ID so backend can echo it back for validation
            taskArgs.request_id = requestId;

            await runPluginTask('Find Similar Scenes', taskArgs);
            pollSimilarResults(similarState.sceneId, requestId);

        } catch (error) {
            log(`Fetch more error: ${error}`, 'error');
            resultsDiv.classList.remove('loading');
            tabState.isLoadingPage = false;
            updatePaginationButtons();
        }
    }

    // Render current page from cache (instant, no backend call)
    function renderCurrentPage() {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const tabState = getTabState();
        const resultsDiv = modal.querySelector('.stash-copilot-similar-results');
        const paginationDiv = modal.querySelector('.stash-copilot-similar-pagination');
        const emptyDiv = modal.querySelector('.stash-copilot-similar-empty');

        // Calculate page slice
        const start = (tabState.currentPage - 1) * similarState.resultsPerPage;
        const end = start + similarState.resultsPerPage;
        const pageResults = tabState.allResults.slice(start, end);
        const pageSceneDetails = tabState.allSceneDetails.slice(start, end);

        if (pageResults.length === 0) {
            resultsDiv.style.display = 'none';
            emptyDiv.style.display = 'block';
            paginationDiv.style.display = 'none';
            updatePaginationButtons();
            return;
        }

        emptyDiv.style.display = 'none';
        resultsDiv.style.display = 'grid';
        paginationDiv.style.display = 'flex';

        // Render scene cards using unified card system
        resultsDiv.innerHTML = pageResults.map((result, idx) => {
            const scene = {
                ...(pageSceneDetails[idx] || {}),
                id: pageSceneDetails[idx]?.id || result.scene_id
            };
            return buildSceneCard({
                scene: scene,
                score: result.similarity,
                cardIndex: idx,
                theme: 'similar',
                scoreLabel: 'match'
            });
        }).join('');

        // Setup card interactions using unified event handler
        setupSceneCardEvents(resultsDiv, { theme: 'similar', tooltipMode: 'fixed' });
        updatePaginationButtons();
    }

    // Poll for similar scene results
    async function pollSimilarResults(sceneId, requestId) {
        const resultFile = `/plugin/stash-copilot/assets/similar_results_${sceneId}.json`;
        const tabState = getTabState();

        const checkResult = async () => {
            // Check if this request is still the current one
            if (requestId && similarState.requestId !== requestId) {
                log(`Similar poll: requestId mismatch, stopping poll (was ${requestId}, now ${similarState.requestId})`);
                return true;  // Stop polling - a new request superseded this one
            }

            try {
                const cacheBuster = `?t=${Date.now()}`;
                const response = await fetch(resultFile + cacheBuster, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();

                    // Check if this result matches our request ID (most reliable check)
                    if (requestId && data.request_id && data.request_id !== requestId) {
                        log(`Similar poll: got request_id ${data.request_id}, waiting for ${requestId}`);
                        return false;  // Keep polling for our request
                    }

                    // Check if this result matches our current tab
                    const expectedFilterMode = similarState.activeTab;
                    if (data.filter_mode && data.filter_mode !== expectedFilterMode) {
                        log(`Similar poll: got filter_mode ${data.filter_mode}, waiting for ${expectedFilterMode}`);
                        return false;  // Keep polling for the right filter mode
                    }

                    // Check if this result matches our current request offset
                    if (data.offset !== undefined && data.offset !== tabState.backendOffset) {
                        log(`Similar poll: got offset ${data.offset}, waiting for ${tabState.backendOffset}`);
                        return false;  // Keep polling for the right offset
                    }

                    log(`Similar poll: found ${data.results?.length || 0} results, has_more=${data.has_more}, filter_mode=${data.filter_mode}, request_id=${data.request_id}`);

                    if (data.status === 'complete' || data.results) {
                        tabState.isSearching = false;
                        tabState.isLoadingPage = false;
                        tabState.hasMoreBackend = data.has_more || false;
                        tabState.loaded = true;
                        await renderSimilarResults(data.results || [], data.has_more);
                        return true;
                    }

                    if (data.status === 'error') {
                        showSimilarError(data.error || 'Search failed');
                        tabState.isSearching = false;
                        tabState.isLoadingPage = false;
                        return true;
                    }
                }
            } catch (e) {
                log(`Similar poll: ${e.message}`);
            }
            return false;
        };

        // Poll every 150ms for up to 30 seconds (faster response)
        const maxAttempts = 200;
        let attempts = 0;

        const poll = async () => {
            if (!similarState.modalOpen) return;

            const done = await checkResult();
            if (!done && attempts < maxAttempts) {
                attempts++;
                setTimeout(poll, 150);
            } else if (!done) {
                showSimilarError('Search timed out. The embedding database may be empty or slow.');
                tabState.isSearching = false;
                tabState.isLoadingPage = false;
            }
        };

        poll();
    }

    // Render similar scenes results - appends to cache and shows current page
    async function renderSimilarResults(results, hasMore) {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const tabState = getTabState();
        const loadingDiv = modal.querySelector('.stash-copilot-similar-loading');
        const resultsDiv = modal.querySelector('.stash-copilot-similar-results');
        const paginationDiv = modal.querySelector('.stash-copilot-similar-pagination');
        const emptyDiv = modal.querySelector('.stash-copilot-similar-empty');

        loadingDiv.style.display = 'none';
        resultsDiv.classList.remove('loading');

        // If no results at all (first fetch returned empty)
        if ((!results || results.length === 0) && tabState.allResults.length === 0) {
            emptyDiv.style.display = 'block';
            emptyDiv.innerHTML = similarState.activeTab === 'different-performers'
                ? `<p>No similar scenes with different performers found.</p>
                   <p class="stash-copilot-similar-hint">All similar scenes share performers with this scene.</p>`
                : `<p>No similar scenes found.</p>
                   <p class="stash-copilot-similar-hint">This scene may not have an embedding yet. Run the "Embed Scene" task first.</p>`;
            paginationDiv.style.display = 'none';
            return;
        }

        // Scene details are now embedded in results from the backend (fetched via SQLite)
        const sceneDetails = results.map(r => r.scene || null);

        // Append to cache
        tabState.allResults = tabState.allResults.concat(results);
        tabState.allSceneDetails = tabState.allSceneDetails.concat(sceneDetails);
        tabState.backendOffset += results.length;
        tabState.hasMoreBackend = hasMore;

        // Render current page from cache
        renderCurrentPage();
    }


    // Update pagination button states
    function updatePaginationButtons() {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const paginationDiv = modal.querySelector('.stash-copilot-similar-pagination');
        const prevBtn = paginationDiv.querySelector('.prev');
        const nextBtn = paginationDiv.querySelector('.next');
        const pageInfo = paginationDiv.querySelector('.stash-copilot-similar-page-info');

        const tabState = getTabState();
        const { currentPage, allResults, hasMoreBackend, isLoadingPage } = tabState;

        // Calculate if there's a next page (either in cache or from backend)
        const nextPageStart = currentPage * similarState.resultsPerPage;
        const hasNextPageCached = nextPageStart < allResults.length;
        const hasMore = hasNextPageCached || hasMoreBackend;

        // Total pages we can navigate to (cached + potentially more)
        const cachedPages = Math.ceil(allResults.length / similarState.resultsPerPage);

        // Show pagination if there's more than one page
        const showPagination = currentPage > 1 || hasMore;
        paginationDiv.style.display = showPagination ? 'flex' : 'none';

        if (showPagination) {
            prevBtn.disabled = currentPage <= 1 || isLoadingPage;
            nextBtn.disabled = !hasMore || isLoadingPage;
            pageInfo.textContent = `Page ${currentPage}${hasMore ? '' : ' (last)'}`;
        }
    }

    // Show error in similar modal
    function showSimilarError(message) {
        const modal = document.getElementById('stash-copilot-similar-modal');
        if (!modal) return;

        const loadingDiv = modal.querySelector('.stash-copilot-similar-loading');
        const errorDiv = modal.querySelector('.stash-copilot-similar-error');

        loadingDiv.style.display = 'none';
        errorDiv.style.display = 'block';
        errorDiv.innerHTML = `<p>${escapeHtml(message)}</p>`;
    }

    // ===== End Similar Scenes Functions =====

    // ===== Scene Sidebar Tabs =====

    // Labeling page state
    const labelingState = {
        initialized: false,
        sessionId: null,
        batch: [],
        vocabulary: [],
        currentIndex: 0,
        annotations: {},       // key: "sceneId_frameIndex", value: {tagText: label}
        pendingSync: [],       // Annotations waiting to be synced
        viewMode: 'single',   // 'single' | 'grid'
        gridSelection: null,   // Currently selected grid item index
        syncTimer: null,
        isLoading: false,
    };

    // Sidebar tab state
    const sidebarTabState = {
        sceneId: null,
        initialized: false,
        activeTab: null,  // 'analyze', 'similar', 'recs', 'gaps', 'tags'
        contentLoaded: {
            analyze: false,
            similar: false,
            recs: false,
            gaps: false,
            tags: false,
        }
    };

    /**
     * Injects AI tabs into Stash's native scene page sidebar
     * @param {string} sceneId - The scene ID
     */
    async function injectSceneTabs(sceneId) {
        log('Injecting scene tabs...');

        try {
            // Wait for the native Stash tab container
            const sceneTabs = await waitForElement('.scene-tabs', 5000);
            const navTabs = sceneTabs.querySelector('.nav-tabs');
            const tabContent = sceneTabs.querySelector('.tab-content');

            if (!navTabs || !tabContent) {
                log('Could not find tab containers', 'error');
                return;
            }

            // Check if already injected
            if (navTabs.querySelector('.stash-copilot-tab-nav')) {
                log('Tabs already injected');
                // Update scene ID if different
                if (sidebarTabState.sceneId !== sceneId) {
                    sidebarTabState.sceneId = sceneId;
                    sidebarTabState.contentLoaded = { analyze: false, similar: false, recs: false, gaps: false, tags: false, scripts: false };
                }
                // If analysis is running for this scene, restore loading state
                // This handles React re-renders that recreate DOM during active analysis
                if (visionState.isAnalyzing && visionState.sceneId === sceneId) {
                    restoreAnalysisLoadingState();
                }
                return;
            }

            sidebarTabState.sceneId = sceneId;
            sidebarTabState.initialized = true;
            sidebarTabState.contentLoaded = { analyze: false, similar: false, recs: false, gaps: false, tags: false, scripts: false };

            // Create tab navigation items
            const tabs = [
                { key: 'scene-copilot-analyze', label: 'Analyze', icon: '👁' },
                { key: 'scene-copilot-similar', label: 'Similar', icon: '🔍' },
                { key: 'scene-copilot-recs', label: 'Recs', icon: '⭐' },
                { key: 'scene-copilot-gaps', label: 'Gaps', icon: '🏷️' },
                { key: 'scene-copilot-tags', label: 'Tags', icon: '✨' },
                { key: 'scene-copilot-scripts', label: 'Scripts', icon: '⚡' },
            ];

            tabs.forEach(tab => {
                // Create nav item
                const navItem = document.createElement('li');
                navItem.className = 'nav-item stash-copilot-tab-nav';

                const navLink = document.createElement('a');
                navLink.className = 'nav-link';
                navLink.setAttribute('data-rb-event-key', tab.key);
                navLink.setAttribute('role', 'tab');
                navLink.setAttribute('href', '#');
                navLink.innerHTML = `<span class="stash-copilot-tab-icon">${tab.icon}</span> ${tab.label}`;

                navLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    handleSidebarTabClick(sceneTabs, tab.key, sceneId);
                });

                navItem.appendChild(navLink);
                navTabs.appendChild(navItem);

                // Create tab pane
                const tabPane = document.createElement('div');
                tabPane.className = 'tab-pane stash-copilot-tab-pane';
                tabPane.id = `${tab.key}-panel`;
                tabPane.setAttribute('role', 'tabpanel');
                tabContent.appendChild(tabPane);
            });

            // Add click listeners to native Stash tabs to deactivate our tabs when they're clicked
            navTabs.querySelectorAll('.nav-link:not(.stash-copilot-tab-nav .nav-link)').forEach(nativeTab => {
                nativeTab.addEventListener('click', () => {
                    // Deactivate our injected tabs
                    navTabs.querySelectorAll('.stash-copilot-tab-nav .nav-link').forEach(link => {
                        link.classList.remove('active');
                        link.setAttribute('aria-selected', 'false');
                    });
                    // Hide our injected tab panes
                    tabContent.querySelectorAll('.stash-copilot-tab-pane').forEach(pane => {
                        pane.classList.remove('active', 'show');
                    });
                    sidebarTabState.activeTab = null;
                });
            });

            log('Scene tabs injected successfully');
        } catch (error) {
            log(`Failed to inject scene tabs: ${error.message}`, 'error');
        }
    }

    /**
     * Handle tab click - manage active states and load content
     */
    function handleSidebarTabClick(container, eventKey, sceneId) {
        log(`Tab clicked: ${eventKey}`);

        // Deactivate all tabs (both native and injected)
        container.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            link.setAttribute('aria-selected', 'false');
        });
        container.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active', 'show');
        });

        // Activate clicked tab
        const navLink = container.querySelector(`a[data-rb-event-key="${eventKey}"]`);
        if (navLink) {
            navLink.classList.add('active');
            navLink.setAttribute('aria-selected', 'true');
        }

        const pane = document.getElementById(`${eventKey}-panel`);
        if (pane) {
            pane.classList.add('active', 'show');
            sidebarTabState.activeTab = eventKey;

            // Load content if not already loaded
            loadSidebarTabContent(eventKey, pane, sceneId);
        }
    }

    /**
     * Load content for a sidebar tab (lazy loading)
     */
    function loadSidebarTabContent(eventKey, container, sceneId) {
        const tabName = eventKey.replace('scene-copilot-', '');

        if (sidebarTabState.contentLoaded[tabName]) {
            log(`Content already loaded for ${tabName}`);
            return;
        }

        log(`Loading content for ${tabName} tab...`);

        switch (tabName) {
            case 'analyze':
                renderSidebarAnalyzeContent(container, sceneId);
                break;
            case 'similar':
                renderSidebarSimilarContent(container, sceneId);
                break;
            case 'recs':
                renderSidebarRecsContent(container, sceneId);
                break;
            case 'gaps':
                renderSidebarGapsContent(container, sceneId);
                break;
            case 'tags':
                renderSidebarTagsContent(container, sceneId);
                break;
            case 'scripts':
                renderSidebarScriptsContent(container, sceneId);
                break;
        }

        sidebarTabState.contentLoaded[tabName] = true;
    }

    /**
     * Render Analyze tab content for sidebar
     */
    function renderSidebarAnalyzeContent(container, sceneId) {
        container.innerHTML = `
            <div class="stash-copilot-sidebar-analyze">
                <div class="stash-copilot-sidebar-header">
                    <span class="stash-copilot-sidebar-title">AI Analysis</span>
                    <div class="stash-copilot-sidebar-actions">
                        <button class="stash-copilot-sidebar-btn stash-copilot-sidebar-details" title="Analysis Details">i</button>
                        <button class="stash-copilot-sidebar-btn stash-copilot-sidebar-settings" title="Edit Prompts">⚙</button>
                        <button class="stash-copilot-sidebar-btn stash-copilot-sidebar-reanalyze" title="Re-analyze">↻</button>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-intro">
                    <div class="stash-copilot-sidebar-intro-icon">👁</div>
                    <h3 class="stash-copilot-sidebar-intro-title">Vision Analysis</h3>
                    <p class="stash-copilot-sidebar-intro-description">
                        Analyze this scene using AI vision to generate a detailed description
                        and suggested tags. The analysis extracts frames from the video and
                        uses a vision language model to understand the content.
                    </p>
                    <button class="stash-copilot-sidebar-analyze-btn">
                        Analyze Scene
                    </button>
                </div>
                <div class="stash-copilot-sidebar-loading" style="display: none;">
                    <div class="stash-copilot-spinner"></div>
                    <span class="stash-copilot-sidebar-status">Initializing...</span>
                    <div class="stash-copilot-sidebar-progress-container">
                        <div class="stash-copilot-sidebar-progress-bar" style="width: 0%"></div>
                    </div>
                    <span class="stash-copilot-sidebar-progress-text"></span>
                </div>
                <div class="stash-copilot-sidebar-analysis"></div>
                <div class="stash-copilot-sidebar-tags-section" style="display: none;">
                    <div class="stash-copilot-sidebar-section-header" data-section="tags">
                        <span>Suggested Tags</span>
                        <button class="stash-copilot-sidebar-collapse-btn" title="Toggle">▼</button>
                    </div>
                    <div class="stash-copilot-sidebar-tags"></div>
                </div>
                <div class="stash-copilot-sidebar-options" style="display: none;">
                    <div class="stash-copilot-sidebar-section-header">
                        <span>Analysis Options</span>
                        <button class="stash-copilot-sidebar-options-reset" title="Reset All">↻</button>
                    </div>
                    <div class="stash-copilot-sidebar-options-content">
                        <label class="stash-copilot-sidebar-option">
                            <input type="checkbox" class="stash-copilot-option-quick-mode">
                            <span>Quick mode (faster)</span>
                        </label>
                        <label class="stash-copilot-sidebar-option">
                            <input type="checkbox" class="stash-copilot-option-skip-verification">
                            <span>Skip verification</span>
                        </label>
                        <div class="stash-copilot-sidebar-option-group">
                            <label>Frames</label>
                            <select class="stash-copilot-option-frame-count">
                                <option value="auto">Auto (smart)</option>
                                <option value="16">16 frames</option>
                                <option value="32">32 frames</option>
                                <option value="64">64 frames</option>
                            </select>
                        </div>
                        <div class="stash-copilot-sidebar-prompts-toggle">
                            <span>▶ Edit prompts</span>
                        </div>
                        <div class="stash-copilot-sidebar-prompts-inner" style="display: none;">
                            <div class="stash-copilot-sidebar-prompt-field">
                                <label>System Prompt</label>
                                <textarea class="stash-copilot-sidebar-prompt-system" rows="3" placeholder="System prompt..."></textarea>
                            </div>
                            <div class="stash-copilot-sidebar-prompt-field">
                                <label>Description Prompt</label>
                                <textarea class="stash-copilot-sidebar-prompt-description" rows="6" placeholder="Description prompt..."></textarea>
                            </div>
                            <div class="stash-copilot-sidebar-prompt-actions">
                                <button class="stash-copilot-sidebar-prompt-save">Save</button>
                                <button class="stash-copilot-sidebar-prompt-reset">Reset</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-debug" style="display: none;">
                    <div class="stash-copilot-sidebar-section-header">
                        <span>Debug Info</span>
                        <button class="stash-copilot-sidebar-debug-toggle">▼</button>
                    </div>
                    <div class="stash-copilot-sidebar-debug-content"></div>
                </div>
                <div class="stash-copilot-sidebar-chat">
                    <div class="stash-copilot-sidebar-messages"></div>
                    <div class="stash-copilot-sidebar-input-container">
                        <input type="text" class="stash-copilot-sidebar-input"
                               placeholder="Ask about this scene..." maxlength="500">
                        <button class="stash-copilot-sidebar-send" title="Send">→</button>
                    </div>
                </div>
            </div>
        `;

        // Set up event listeners
        setupSidebarAnalyzeListeners(container, sceneId);

        // If analysis is currently running for this scene, show loading state
        // This handles DOM recreation during active analysis
        if (visionState.isAnalyzing && visionState.sceneId === sceneId) {
            restoreAnalysisLoadingState();
            return;  // Skip cached check since we're analyzing
        }

        // Check for cached analysis (don't auto-run)
        checkForCachedAnalysis(sceneId);
    }

    /**
     * Render Similar tab content for sidebar
     */
    function renderSidebarSimilarContent(container, sceneId) {
        const sliderValue = Math.round(similarState.visualWeight * 100);

        container.innerHTML = `
            <div class="stash-copilot-sidebar-similar">
                <button class="stash-copilot-frame-search-btn" title="Search for similar scenes using the current video frame">
                    🎯 Search by Current Frame
                </button>
                <div class="stash-copilot-sidebar-subtabs">
                    <button class="stash-copilot-sidebar-subtab active" data-filter="all">All</button>
                    <button class="stash-copilot-sidebar-subtab" data-filter="different-performers">Diff. Performers</button>
                    <button class="stash-copilot-sidebar-filter-toggle" title="Filters">
                        <span>⚙</span>
                        <span class="stash-copilot-sidebar-filter-badge" style="display: none;">0</span>
                    </button>
                </div>
                <div class="stash-copilot-sidebar-slider">
                    <span class="stash-copilot-sidebar-slider-label">Meta</span>
                    <input type="range" class="stash-copilot-sidebar-slider-input"
                           min="0" max="100" value="${sliderValue}">
                    <span class="stash-copilot-sidebar-slider-label">Visual</span>
                    <span class="stash-copilot-sidebar-slider-value">${sliderValue}%</span>
                </div>
                <div class="stash-copilot-sidebar-model-info" style="display: none;">
                    <span class="stash-copilot-model-badge"></span>
                </div>
                <div class="stash-copilot-sidebar-filters" style="display: none;">
                    <div class="stash-copilot-sidebar-filter-group">
                        <label>Exclude Performers</label>
                        <div class="stash-copilot-sidebar-filter-input-wrap">
                            <input type="text" class="stash-copilot-sidebar-filter-performer" placeholder="Type to search...">
                            <div class="stash-copilot-sidebar-autocomplete"></div>
                        </div>
                        <div class="stash-copilot-sidebar-filter-tags stash-copilot-sidebar-excluded-performers"></div>
                    </div>
                    <div class="stash-copilot-sidebar-filter-group">
                        <label>Exclude Tags</label>
                        <div class="stash-copilot-sidebar-filter-input-wrap">
                            <input type="text" class="stash-copilot-sidebar-filter-tag" placeholder="Type to search...">
                            <div class="stash-copilot-sidebar-autocomplete"></div>
                        </div>
                        <div class="stash-copilot-sidebar-filter-tags stash-copilot-sidebar-excluded-tags"></div>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-results">
                    <div class="stash-copilot-sidebar-loading">
                        <div class="stash-copilot-spinner"></div>
                        <span>Finding similar scenes...</span>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-pagination" style="display: none;">
                    <button class="stash-copilot-sidebar-page-btn prev" disabled>&lt;</button>
                    <span class="stash-copilot-sidebar-page-info">1 / 1</span>
                    <button class="stash-copilot-sidebar-page-btn next" disabled>&gt;</button>
                </div>
            </div>
        `;

        // Set up event listeners
        setupSidebarSimilarListeners(container, sceneId);

        // Frame search button handler
        const frameSearchBtn = container.querySelector('.stash-copilot-frame-search-btn');
        if (frameSearchBtn) {
            frameSearchBtn.addEventListener('click', () => {
                startFrameSearch(sceneId, container);
            });
        }

        // Start search
        startSidebarSimilarSearch(sceneId);
    }

    // ===== Sidebar Recs State =====
    // State for sidebar recommendations (shared between discover and rewatch modes)
    const savedSeedWeight = localStorage.getItem('stash-copilot-seed-weight');
    const savedEngagementWeight = localStorage.getItem('stash-copilot-engagement-weight');
    const savedTimeDecayDays = localStorage.getItem('stash-copilot-sidebar-rec-decay');

    const sceneRecsState = {
        sceneId: null,
        mode: 'discover_new',  // 'discover_new' | 'rewatch'
        results: [],
        allResults: [],  // Unfiltered results from backend
        profile: null,
        requestId: null,
        currentPage: 1,
        resultsPerPage: 12,
        seedWeight: savedSeedWeight !== null ? parseFloat(savedSeedWeight) : 0.3,
        engagementWeight: savedEngagementWeight !== null ? parseFloat(savedEngagementWeight) : 0.6,
        timeDecayDays: parseInt(savedTimeDecayDays || '0'),
        excludePerformers: [],
        excludeTags: []
    };

    // Tag suggestions state
    const tagSuggestionState = {
        sceneId: null,
        suggestions: [],
        loading: false,
        error: null,
        currentPage: 0,
        suggestionsPerPage: 5,
    };

    /**
     * Render Recs tab content for sidebar
     */
    function renderSidebarRecsContent(container, sceneId) {
        const isDiscoverMode = sceneRecsState.mode === 'discover_new';
        const sliderValue = isDiscoverMode
            ? Math.round(sceneRecsState.seedWeight * 100)
            : Math.round((1 - sceneRecsState.engagementWeight) * 100);
        const leftLabel = isDiscoverMode ? 'Profile' : 'Engagement';
        const rightLabel = isDiscoverMode ? 'Scene' : 'Similarity';

        container.innerHTML = `
            <div class="stash-copilot-sidebar-recs">
                <div class="stash-copilot-sidebar-recency-control">
                    <label class="stash-copilot-sidebar-recency-label">Recency</label>
                    <select class="stash-copilot-sidebar-recency-select">
                        <option value="0"${sceneRecsState.timeDecayDays === 0 ? ' selected' : ''}>All time</option>
                        <option value="3"${sceneRecsState.timeDecayDays === 3 ? ' selected' : ''}>3 days</option>
                        <option value="7"${sceneRecsState.timeDecayDays === 7 ? ' selected' : ''}>7 days</option>
                        <option value="14"${sceneRecsState.timeDecayDays === 14 ? ' selected' : ''}>14 days</option>
                        <option value="28"${sceneRecsState.timeDecayDays === 28 ? ' selected' : ''}>28 days</option>
                        <option value="60"${sceneRecsState.timeDecayDays === 60 ? ' selected' : ''}>60 days</option>
                        <option value="90"${sceneRecsState.timeDecayDays === 90 ? ' selected' : ''}>90 days</option>
                        <option value="180"${sceneRecsState.timeDecayDays === 180 ? ' selected' : ''}>180 days</option>
                        <option value="365"${sceneRecsState.timeDecayDays === 365 ? ' selected' : ''}>1 year</option>
                    </select>
                </div>
                <div class="stash-copilot-sidebar-subtabs">
                    <button class="stash-copilot-sidebar-subtab ${sceneRecsState.mode === 'discover_new' ? 'active' : ''}" data-mode="discover_new" data-tooltip="Find unwatched scenes matching your taste. Slider blends profile preferences with similarity to this scene.">Discover</button>
                    <button class="stash-copilot-sidebar-subtab ${sceneRecsState.mode === 'rewatch' ? 'active' : ''}" data-mode="rewatch" data-tooltip="Rank watched scenes by engagement + similarity. Slider balances favorites vs. profile match.">Re-watch</button>
                    <button class="stash-copilot-sidebar-filter-toggle" title="Filters">
                        <span>⚙</span>
                        <span class="stash-copilot-sidebar-filter-badge" style="display: none;">0</span>
                    </button>
                </div>
                <div class="stash-copilot-sidebar-slider">
                    <span class="stash-copilot-sidebar-slider-label" data-slider-label="left">${leftLabel}</span>
                    <input type="range" class="stash-copilot-sidebar-slider-input"
                           min="0" max="100" value="${sliderValue}">
                    <span class="stash-copilot-sidebar-slider-label" data-slider-label="right">${rightLabel}</span>
                    <span class="stash-copilot-sidebar-slider-value">${sliderValue}%</span>
                </div>
                <div class="stash-copilot-sidebar-filters" style="display: none;">
                    <div class="stash-copilot-sidebar-filter-group">
                        <label>Exclude Performers</label>
                        <div class="stash-copilot-sidebar-filter-input-wrap">
                            <input type="text" class="stash-copilot-sidebar-filter-performer" placeholder="Type to search...">
                            <div class="stash-copilot-sidebar-autocomplete"></div>
                        </div>
                        <div class="stash-copilot-sidebar-filter-tags stash-copilot-sidebar-excluded-performers"></div>
                    </div>
                    <div class="stash-copilot-sidebar-filter-group">
                        <label>Exclude Tags</label>
                        <div class="stash-copilot-sidebar-filter-input-wrap">
                            <input type="text" class="stash-copilot-sidebar-filter-tag" placeholder="Type to search...">
                            <div class="stash-copilot-sidebar-autocomplete"></div>
                        </div>
                        <div class="stash-copilot-sidebar-filter-tags stash-copilot-sidebar-excluded-tags"></div>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-profile-info" style="display: none;"></div>
                <div class="stash-copilot-sidebar-results">
                    <div class="stash-copilot-sidebar-loading">
                        <div class="stash-copilot-spinner"></div>
                        <span>Getting recommendations...</span>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-pagination" style="display: none;">
                    <button class="stash-copilot-sidebar-page-btn prev" disabled>&lt;</button>
                    <span class="stash-copilot-sidebar-page-info">1 / 1</span>
                    <button class="stash-copilot-sidebar-page-btn next" disabled>&gt;</button>
                </div>
            </div>
        `;

        // Set up event listeners
        setupSidebarRecsListeners(container, sceneId);

        // Start recommendations
        startSidebarRecsSearch(sceneId);
    }

    /**
     * Render the Gaps sidebar tab skeleton with loading state.
     */
    function renderSidebarGapsContent(container, sceneId) {
        container.innerHTML = `
            <div class="stash-copilot-sidebar-gaps">
                <div class="stash-copilot-sidebar-header">
                    <span class="stash-copilot-sidebar-title">Tag Gaps</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Loading coverage data...</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-content" style="display: none"></div>
                <div class="stash-copilot-sidebar-gaps-empty" style="display: none">
                    <p>No tag gap data for this scene.</p>
                    <p class="stash-copilot-sidebar-gaps-empty-hint">Run <strong>Detect Tag Gaps</strong> from AI Insights to analyze your library.</p>
                </div>
            </div>
        `;
        loadSidebarGapsData(container, sceneId);
    }

    /**
     * Trigger the backend task to fetch scene tag gap data and poll for results.
     */
    async function loadSidebarGapsData(container, sceneId) {
        const loadingEl = container.querySelector('.stash-copilot-sidebar-gaps-loading');
        const contentEl = container.querySelector('.stash-copilot-sidebar-gaps-content');
        const emptyEl = container.querySelector('.stash-copilot-sidebar-gaps-empty');

        try {
            const requestId = `${sceneId}_${Date.now()}`;
            await runPluginTask('Get Scene Tag Gaps', {
                scene_id: String(sceneId),
                request_id: requestId
            });

            const resultFile = `/plugin/stash-copilot/assets/tag_gaps_scene_${requestId}.json`;
            let attempts = 0;
            const maxAttempts = 30;

            const poll = setInterval(async () => {
                attempts++;
                if (attempts > maxAttempts) {
                    clearInterval(poll);
                    if (loadingEl) loadingEl.style.display = 'none';
                    if (emptyEl) {
                        emptyEl.style.display = '';
                        emptyEl.querySelector('p').textContent = 'Timed out loading gap data.';
                    }
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        if (loadingEl) loadingEl.style.display = 'none';
                        if (data.has_data === false) {
                            if (emptyEl) emptyEl.style.display = '';
                        } else {
                            if (contentEl) {
                                contentEl.style.display = '';
                                await renderSidebarGapsDetail(contentEl, data, sceneId);
                            }
                        }
                    }
                } catch (e) { /* not ready yet */ }
            }, 1000);
        } catch (e) {
            log(`Load sidebar gaps error: ${e.message}`, 'error');
            if (loadingEl) loadingEl.style.display = 'none';
            if (emptyEl) emptyEl.style.display = '';
        }
    }

    /**
     * Render detailed gap analysis: coverage bar, nearest tags, uncovered frames, similar scenes.
     */
    // Tooltip content for Tag Gaps explanations
    const TAG_GAPS_TOOLTIPS = {
        coverage: `<strong>Coverage</strong> measures how much of this scene's visual content is describable by your existing tags.

<strong>How it works:</strong>
• Each frame is compared against ALL tags in your library
• The best-matching tag for each frame is recorded
• If that best match exceeds the threshold (5th percentile), the frame is "covered"

<strong>Coverage %</strong> = frames with a good tag match / total frames

A frame can be "covered" by different tags — POV for some frames, blowjob for others, etc. The coverage isn't attributed to any single tag.`,

        nearestTags: `<strong>Nearest Tags (not matching)</strong> shows which existing tags came closest to matching the uncovered frames — but still fell below the threshold.

<strong>The columns:</strong>
• <strong>Tag name</strong>: The closest-matching tag from your library
• <strong>Similarity %</strong>: How visually similar the frames are to this tag (higher = closer match)
• <strong>Frame count</strong>: How many uncovered frames had this as their best match

These are candidates for tags you might want to add to this scene, or hints that you need a new tag to describe this content.`,

        similarity: `<strong>Similarity %</strong> measures visual similarity between a frame and a tag using AI embeddings.

• <strong>~30%+</strong>: Strong match (above threshold, "covered")
• <strong>~25-30%</strong>: Close but not quite matching
• <strong>~20-25%</strong>: Weak match
• <strong>&lt;20%</strong>: Poor match

The threshold is adaptive — it's the 5th percentile of all similarity scores in your library.`,

        threshold: `<strong>Threshold</strong> is the minimum similarity required for a frame to be considered "covered".

<strong>How it's calculated:</strong>
• All frames in your library are compared to their best-matching tag
• The similarity scores are sorted
• The 5th percentile value becomes the threshold

<strong>What this means:</strong>
• Frames with similarity ≥ threshold → "covered"
• Frames with similarity &lt; threshold → "uncovered"
• ~5% of frames are always uncovered by design

The threshold adapts to your tag vocabulary — adding more descriptive tags raises the overall similarity scores.`,

        sceneTagCoverage: `<strong>Scene Tags Only</strong> shows coverage using ONLY the tags assigned to this scene, ignoring all other library tags.

<strong>Comparison:</strong>
• <strong>Coverage (above)</strong>: Uses ALL 500+ tags in your library
• <strong>Scene Tags Only</strong>: Uses only the tags on THIS scene

<strong>Why it matters:</strong>
• Low scene-tag coverage = the scene's tags don't describe all its content
• You may need to add more tags to this scene
• Or the existing tags are too generic to match well

<strong>Example:</strong>
A scene might have 80% library coverage but only 40% scene-tag coverage — meaning 40% of the content isn't described by the tags you've assigned.`,

        suggestedTags: `<strong>Suggested Tags</strong> are existing library tags that would improve this scene's coverage.

<strong>How they're found:</strong>
• Finds frames covered by library tags but NOT by scene tags
• Shows which tags best match those frames
• Click <strong>+</strong> to add the tag to this scene

<strong>Why they appear:</strong>
• Your library has tags that match this scene's content
• But you haven't assigned them to this scene yet
• Adding them will improve "Scene Tags Only" coverage

<strong>The columns:</strong>
• <strong>+</strong>: Click to add tag to scene
• <strong>Tag name</strong>: An existing tag in your library
• <strong>Sim</strong>: Average similarity to matching frames
• <strong>Frames</strong>: Number of frames this tag would help cover`
    };

    async function renderSidebarGapsDetail(container, data, sceneId) {
        const covPct = Math.round(data.coverage_ratio * 100);
        const barColor = covPct > 75 ? '#10b981' : covPct > 50 ? '#f59e0b' : '#ef4444';

        let html = `
            <div class="stash-copilot-sidebar-gaps-coverage">
                <div class="stash-copilot-sidebar-gaps-coverage-header">
                    <span>Coverage</span>
                    <button class="stash-copilot-sidebar-gaps-info-btn" data-tooltip="coverage" title="What is coverage?">?</button>
                    <span style="color: ${barColor}; font-weight: 600; margin-left: auto">${covPct}%</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-coverage-bar-wrap">
                    <div class="stash-copilot-sidebar-gaps-coverage-fill" style="width: ${covPct}%; background: ${barColor}"></div>
                </div>
                <div class="stash-copilot-sidebar-gaps-coverage-detail">
                    ${data.covered_frames} covered / ${data.uncovered_frames} uncovered of ${data.total_frames} frames
                </div>
                <div class="stash-copilot-sidebar-gaps-threshold">
                    Threshold: ${data.threshold ? (data.threshold * 100).toFixed(1) : '?'}% similarity
                    <button class="stash-copilot-sidebar-gaps-info-btn stash-copilot-sidebar-gaps-info-btn-small" data-tooltip="threshold" title="What is threshold?">?</button>
                </div>
            </div>
        `;

        // Scene-tag-specific coverage bar
        const stc = data.scene_tag_coverage;
        if (stc && stc.tag_count > 0) {
            const sceneTagPct = Math.round(stc.coverage_ratio * 100);
            const sceneTagColor = sceneTagPct > 75 ? '#10b981' : sceneTagPct > 50 ? '#f59e0b' : '#ef4444';
            html += `
                <div class="stash-copilot-sidebar-gaps-scene-coverage">
                    <div class="stash-copilot-sidebar-gaps-coverage-header">
                        <span>Scene Tags Only</span>
                        <button class="stash-copilot-sidebar-gaps-info-btn" data-tooltip="sceneTagCoverage" title="What is scene tag coverage?">?</button>
                        <span style="color: ${sceneTagColor}; font-weight: 600; margin-left: auto">${sceneTagPct}%</span>
                    </div>
                    <div class="stash-copilot-sidebar-gaps-coverage-bar-wrap">
                        <div class="stash-copilot-sidebar-gaps-coverage-fill" style="width: ${sceneTagPct}%; background: ${sceneTagColor}"></div>
                    </div>
                    <div class="stash-copilot-sidebar-gaps-coverage-detail">
                        ${stc.covered_frames} frames covered by ${stc.tag_count} scene tag${stc.tag_count !== 1 ? 's' : ''}
                    </div>
                </div>
            `;
        } else if (stc && stc.tag_count === 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-scene-coverage stash-copilot-sidebar-gaps-scene-coverage-empty">
                    <span>No tags assigned to this scene</span>
                </div>
            `;
        }

        // Suggested tags (would improve scene coverage)
        if (data.suggested_tags && data.suggested_tags.length > 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-suggested">
                    <div class="stash-copilot-sidebar-gaps-section-title">
                        Suggested Tags
                        <button class="stash-copilot-sidebar-gaps-info-btn" data-tooltip="suggestedTags" title="What are suggested tags?">?</button>
                    </div>
                    <div class="stash-copilot-sidebar-gaps-tag-list">
                        <div class="stash-copilot-sidebar-gaps-tag-header stash-copilot-sidebar-gaps-tag-header-4col">
                            <span></span>
                            <span>Tag</span>
                            <span>Sim</span>
                            <span>Frames</span>
                        </div>
                        ${data.suggested_tags.slice(0, 8).map(t => `
                            <div class="stash-copilot-sidebar-gaps-tag-row stash-copilot-sidebar-gaps-tag-row-4col">
                                <button class="stash-copilot-sidebar-gaps-add-tag-btn" data-tag="${escapeHtml(t.tag)}" data-scene-id="${sceneId}" title="Add '${escapeHtml(t.tag)}' to scene">+</button>
                                <span class="stash-copilot-sidebar-gaps-tag-name">${escapeHtml(t.tag)}</span>
                                <span class="stash-copilot-sidebar-gaps-tag-sim">${(t.avg_similarity * 100).toFixed(0)}%</span>
                                <span class="stash-copilot-sidebar-gaps-tag-count">${t.frame_count}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Nearest tags for uncovered content
        if (data.nearest_tags && data.nearest_tags.length > 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-nearest">
                    <div class="stash-copilot-sidebar-gaps-section-title">
                        Nearest Tags (not matching)
                        <button class="stash-copilot-sidebar-gaps-info-btn" data-tooltip="nearestTags" title="What are nearest tags?">?</button>
                    </div>
                    <div class="stash-copilot-sidebar-gaps-tag-list">
                        <div class="stash-copilot-sidebar-gaps-tag-header stash-copilot-sidebar-gaps-tag-header-4col">
                            <span></span>
                            <span>Tag</span>
                            <span>Sim <button class="stash-copilot-sidebar-gaps-info-btn stash-copilot-sidebar-gaps-info-btn-small" data-tooltip="similarity" title="What is similarity?">?</button></span>
                            <span>Frames</span>
                        </div>
                        ${data.nearest_tags.slice(0, 8).map(t => `
                            <div class="stash-copilot-sidebar-gaps-tag-row stash-copilot-sidebar-gaps-tag-row-4col">
                                <button class="stash-copilot-sidebar-gaps-add-tag-btn" data-tag="${escapeHtml(t.tag)}" data-scene-id="${sceneId}" title="Add '${escapeHtml(t.tag)}' to scene">+</button>
                                <span class="stash-copilot-sidebar-gaps-tag-name">${escapeHtml(t.tag)}</span>
                                <span class="stash-copilot-sidebar-gaps-tag-sim">${(t.avg_similarity * 100).toFixed(0)}%</span>
                                <span class="stash-copilot-sidebar-gaps-tag-count">${t.frame_count}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Uncovered frames strip
        if (data.uncovered_frame_list && data.uncovered_frame_list.length > 0) {
            const frameDir = `/plugin/stash-copilot/assets/embedded_frames/scene_${sceneId}`;
            html += `
                <div class="stash-copilot-sidebar-gaps-frames">
                    <div class="stash-copilot-sidebar-gaps-section-title">Uncovered Frames</div>
                    <div class="stash-copilot-sidebar-gaps-frame-strip">
                        ${data.uncovered_frame_list.slice(0, 20).map(f => {
                            const frameNum = String(f.frame_index + 1).padStart(4, '0');
                            const src = `${frameDir}/frame_${frameNum}.jpg`;
                            const mins = Math.floor(f.timestamp / 60);
                            const secs = Math.floor(f.timestamp % 60);
                            const ts = `${mins}:${String(secs).padStart(2, '0')}`;
                            return `
                                <div class="stash-copilot-sidebar-gaps-frame" title="${ts} — nearest: ${f.best_tag} (${(f.best_similarity * 100).toFixed(0)}%)">
                                    <img src="${src}" loading="lazy" alt="Frame at ${ts}" />
                                    <span class="stash-copilot-sidebar-gaps-frame-ts">${ts}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        // Similar uncovered scenes - placeholder while fetching metadata
        if (data.similar_uncovered && data.similar_uncovered.length > 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-similar">
                    <div class="stash-copilot-sidebar-gaps-section-title">Similar Uncovered Content</div>
                    <div class="stash-copilot-sidebar-gaps-similar-list">
                        <div class="stash-copilot-sidebar-gaps-loading-cards">
                            <div class="stash-copilot-spinner" style="width: 16px; height: 16px;"></div>
                            <span>Loading scenes...</span>
                        </div>
                    </div>
                </div>
            `;
        }

        // Tag preview section
        html += `
            <div class="stash-copilot-sidebar-gaps-preview">
                <div class="stash-copilot-sidebar-gaps-section-title">Test Tag Impact</div>
                <div class="stash-copilot-sidebar-gaps-preview-input">
                    <input type="text"
                           class="stash-copilot-sidebar-gaps-tag-input"
                           placeholder="Enter tag name..."
                           data-scene-id="${sceneId}" />
                    <button class="stash-copilot-sidebar-gaps-preview-btn">Preview</button>
                </div>
                <div class="stash-copilot-sidebar-gaps-preview-result" style="display: none;"></div>
            </div>
        `;

        container.innerHTML = html;

        // Setup tag preview event handlers
        setupTagPreviewHandlers(container, sceneId);

        // Setup info tooltip handlers
        setupGapsInfoTooltips(container);

        // Setup add tag button handlers
        setupAddTagHandlers(container, sceneId);

        // Fetch scene metadata and render proper cards
        if (data.similar_uncovered && data.similar_uncovered.length > 0) {
            const similarList = container.querySelector('.stash-copilot-sidebar-gaps-similar-list');
            if (similarList) {
                await renderSimilarUncoveredCards(similarList, data.similar_uncovered.slice(0, 5));
            }
        }
    }

    /**
     * Setup info button tooltips for Tag Gaps explanations.
     */
    function setupGapsInfoTooltips(container) {
        const infoButtons = container.querySelectorAll('.stash-copilot-sidebar-gaps-info-btn');

        infoButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const tooltipKey = btn.dataset.tooltip;
                const content = TAG_GAPS_TOOLTIPS[tooltipKey];
                if (!content) return;

                // Remove any existing tooltip
                const existing = document.querySelector('.stash-copilot-gaps-tooltip');
                if (existing) existing.remove();

                // Create tooltip
                const tooltip = document.createElement('div');
                tooltip.className = 'stash-copilot-gaps-tooltip';
                tooltip.innerHTML = `
                    <div class="stash-copilot-gaps-tooltip-content">${content.replace(/\n/g, '<br>')}</div>
                    <button class="stash-copilot-gaps-tooltip-close">×</button>
                `;

                // Position tooltip
                const rect = btn.getBoundingClientRect();
                tooltip.style.position = 'fixed';
                tooltip.style.top = `${rect.bottom + 8}px`;
                tooltip.style.left = `${Math.max(10, rect.left - 150)}px`;
                tooltip.style.zIndex = '10000';

                document.body.appendChild(tooltip);

                // Close button
                tooltip.querySelector('.stash-copilot-gaps-tooltip-close').addEventListener('click', () => {
                    tooltip.remove();
                });

                // Close on click outside
                const closeOnClickOutside = (e) => {
                    if (!tooltip.contains(e.target) && e.target !== btn) {
                        tooltip.remove();
                        document.removeEventListener('click', closeOnClickOutside);
                    }
                };
                setTimeout(() => document.addEventListener('click', closeOnClickOutside), 10);
            });
        });
    }

    /**
     * Setup event handlers for tag preview feature.
     */
    function setupTagPreviewHandlers(container, sceneId) {
        const input = container.querySelector('.stash-copilot-sidebar-gaps-tag-input');
        const previewBtn = container.querySelector('.stash-copilot-sidebar-gaps-preview-btn');
        const resultEl = container.querySelector('.stash-copilot-sidebar-gaps-preview-result');

        if (!input || !previewBtn || !resultEl) return;

        const runPreview = async () => {
            const tagName = input.value.trim();
            if (!tagName) return;

            previewBtn.disabled = true;
            previewBtn.textContent = '...';
            resultEl.style.display = 'block';
            resultEl.innerHTML = `
                <div class="stash-copilot-sidebar-gaps-loading-cards">
                    <div class="stash-copilot-spinner" style="width: 14px; height: 14px;"></div>
                    <span>Analyzing "${escapeHtml(tagName)}"...</span>
                </div>
            `;

            try {
                const requestId = `${sceneId}_${tagName.replace(/\W/g, '_')}_${Date.now()}`;
                await runPluginTask('Preview Tag Impact', {
                    scene_id: String(sceneId),
                    tag_name: tagName,
                    request_id: requestId
                });

                const resultFile = `/plugin/stash-copilot/assets/tag_preview_${requestId}.json`;
                let attempts = 0;
                const maxAttempts = 30;

                const poll = setInterval(async () => {
                    attempts++;
                    if (attempts > maxAttempts) {
                        clearInterval(poll);
                        resultEl.innerHTML = `<div class="stash-copilot-sidebar-gaps-preview-error">Timed out</div>`;
                        previewBtn.disabled = false;
                        previewBtn.textContent = 'Preview';
                        return;
                    }
                    try {
                        const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                        if (resp.ok) {
                            clearInterval(poll);
                            const data = await resp.json();
                            renderTagPreviewResult(resultEl, data, sceneId);
                            previewBtn.disabled = false;
                            previewBtn.textContent = 'Preview';
                        }
                    } catch (e) { /* not ready yet */ }
                }, 500);
            } catch (e) {
                resultEl.innerHTML = `<div class="stash-copilot-sidebar-gaps-preview-error">Error: ${escapeHtml(e.message)}</div>`;
                previewBtn.disabled = false;
                previewBtn.textContent = 'Preview';
            }
        };

        previewBtn.addEventListener('click', runPreview);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') runPreview();
        });
    }

    /**
     * Render the tag preview result with coverage impact visualization.
     */
    function renderTagPreviewResult(container, data, sceneId) {
        if (data.error) {
            container.innerHTML = `<div class="stash-copilot-sidebar-gaps-preview-error">${escapeHtml(data.error)}</div>`;
            return;
        }

        const currentPct = Math.round(data.current_coverage * 100);
        const newPct = Math.round(data.new_coverage * 100);
        const improvement = newPct - currentPct;
        const improvementColor = improvement > 0 ? '#10b981' : '#6b7280';

        container.innerHTML = `
            <div class="stash-copilot-sidebar-gaps-preview-tag">"${escapeHtml(data.tag_name)}"</div>
            <div class="stash-copilot-sidebar-gaps-preview-stats">
                <div class="stash-copilot-sidebar-gaps-preview-row">
                    <span>Coverage:</span>
                    <span>${currentPct}% → <strong style="color: ${improvementColor}">${newPct}%</strong></span>
                </div>
                <div class="stash-copilot-sidebar-gaps-preview-row">
                    <span>Frames covered:</span>
                    <span style="color: ${improvementColor}">+${data.frames_covered} of ${data.total_uncovered}</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-preview-row">
                    <span>Max similarity:</span>
                    <span>${(data.max_similarity * 100).toFixed(0)}%</span>
                </div>
            </div>
            <div class="stash-copilot-sidebar-gaps-preview-bar">
                <div class="stash-copilot-sidebar-gaps-preview-bar-current" style="width: ${currentPct}%"></div>
                <div class="stash-copilot-sidebar-gaps-preview-bar-new" style="width: ${newPct}%; opacity: 0.5"></div>
            </div>
            ${improvement > 0 ? `
                <button class="stash-copilot-sidebar-gaps-add-tag-btn" data-tag-name="${escapeHtml(data.tag_name)}" data-scene-id="${sceneId}">
                    Add "${escapeHtml(data.tag_name)}" to Scene
                </button>
            ` : `
                <div class="stash-copilot-sidebar-gaps-preview-hint">This tag wouldn't improve coverage (similarity below threshold ${(data.threshold * 100).toFixed(0)}%)</div>
            `}
        `;

        // Setup add tag button handler
        const addBtn = container.querySelector('.stash-copilot-sidebar-gaps-add-tag-btn');
        if (addBtn) {
            addBtn.addEventListener('click', async () => {
                const tagName = addBtn.dataset.tagName;
                const sceneIdAttr = addBtn.dataset.sceneId;
                addBtn.disabled = true;
                addBtn.textContent = 'Adding...';

                try {
                    await addTagToScene(sceneIdAttr, tagName);
                    addBtn.textContent = '✓ Added!';
                    addBtn.style.background = '#10b981';
                } catch (e) {
                    addBtn.textContent = 'Failed';
                    addBtn.style.background = '#ef4444';
                    log(`Failed to add tag: ${e.message}`, 'error');
                }
            });
        }
    }

    /**
     * Add a tag to a scene via GraphQL mutation.
     */
    async function addTagToScene(sceneId, tagName) {
        // First, find or create the tag
        let tagResult = await callGQL(`
            query FindTag($name: String!) {
                findTags(tag_filter: { name: { value: $name, modifier: EQUALS } }, filter: { per_page: 1 }) {
                    tags { id name }
                }
            }
        `, { name: tagName });

        let tagId;
        if (tagResult?.findTags?.tags?.length > 0) {
            tagId = tagResult.findTags.tags[0].id;
        } else {
            // Create the tag
            const createResult = await callGQL(`
                mutation CreateTag($input: TagCreateInput!) {
                    tagCreate(input: $input) { id name }
                }
            `, { input: { name: tagName } });
            tagId = createResult?.tagCreate?.id;
        }

        if (!tagId) throw new Error('Failed to find or create tag');

        // Get current scene tags
        const sceneResult = await callGQL(`
            query FindScene($id: ID!) {
                findScene(id: $id) {
                    tags { id }
                }
            }
        `, { id: String(sceneId) });

        const currentTagIds = sceneResult?.findScene?.tags?.map(t => t.id) || [];

        // Check if already has tag
        if (currentTagIds.includes(tagId)) {
            return; // Already has tag
        }

        // Add the new tag
        const newTagIds = [...currentTagIds, tagId];
        await callGQL(`
            mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }
        `, { input: { id: String(sceneId), tag_ids: newTagIds } });
    }

    /**
     * Setup event handlers for add tag "+" buttons in the tag lists.
     */
    function setupAddTagHandlers(container, sceneId) {
        const addBtns = container.querySelectorAll('.stash-copilot-sidebar-gaps-add-tag-btn');

        addBtns.forEach(btn => {
            // Skip if it's the preview result button (handled separately)
            if (btn.dataset.tagName) return;

            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const tagName = btn.dataset.tag;
                if (!tagName) return;

                btn.disabled = true;
                const originalText = btn.textContent;
                btn.textContent = '...';

                try {
                    await addTagToScene(sceneId, tagName);
                    btn.textContent = '✓';
                    btn.classList.add('stash-copilot-sidebar-gaps-add-tag-btn-added');

                    // Remove the row after a brief delay
                    setTimeout(() => {
                        const row = btn.closest('.stash-copilot-sidebar-gaps-tag-row');
                        if (row) {
                            row.style.opacity = '0.5';
                            row.style.textDecoration = 'line-through';
                        }
                    }, 500);
                } catch (err) {
                    btn.textContent = '!';
                    btn.title = `Failed: ${err.message}`;
                    btn.disabled = false;
                    log(`Failed to add tag '${tagName}': ${err.message}`, 'error');
                }
            });
        });
    }

    /**
     * Fetch scene metadata and render cards for similar uncovered scenes.
     */
    async function renderSimilarUncoveredCards(container, similarScenes) {
        const sceneIds = similarScenes.map(s => s.scene_id);
        let sceneMap = {};

        try {
            const gqlResult = await callGQL(`
                query FindScenes($ids: [Int!]!) {
                    findScenes(scene_filter: { id: { modifier: INCLUDES, value: $ids } }, filter: { per_page: -1 }) {
                        scenes {
                            id title date
                            files { path duration height size fingerprints { type value } }
                            performers { id name }
                            studio { id name }
                            tags { id name }
                            play_count o_counter rating100 interactive
                        }
                    }
                }`, { ids: sceneIds });
            const scenes = gqlResult?.findScenes?.scenes || [];
            scenes.forEach(s => { sceneMap[s.id] = s; });
        } catch (e) {
            log(`Gaps sidebar: failed to fetch scene details: ${e.message}`, 'error');
        }

        // Render cards with fetched metadata
        container.innerHTML = similarScenes.map((s, idx) => {
            const meta = sceneMap[String(s.scene_id)] || { id: s.scene_id };
            return buildSceneCard({
                scene: meta,
                score: s.similarity,
                cardIndex: idx,
                theme: 'tag-gaps',
                scoreLabel: 'similar'
            });
        }).join('');

        setupSceneCardEvents(container, { theme: 'tag-gaps', tooltipMode: 'cursor' });
    }

    /**
     * Check for cached analysis results and show intro or cached results
     */
    async function checkForCachedAnalysis(sceneId) {
        // Capture scene ID to validate when response arrives
        const capturedSceneId = sceneId;

        // Skip loading cached results if analysis is currently running
        // This prevents stale results from overwriting the loading state
        // when React re-renders cause DOM recreation during analysis
        if (visionState.isAnalyzing) {
            log('Analysis in progress, skipping cached check');
            return;
        }

        const resultFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;
        const panel = document.getElementById('scene-copilot-analyze-panel');
        if (!panel) return;

        try {
            const response = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'complete') {
                    // Validate we're still on the same scene before rendering
                    if (sidebarTabState.sceneId !== capturedSceneId) {
                        log(`Scene changed during cached fetch (${capturedSceneId} -> ${sidebarTabState.sceneId}), discarding results`);
                        return;
                    }
                    log('Found cached vision analysis');
                    // Hide intro, show cached results
                    const introDiv = panel.querySelector('.stash-copilot-sidebar-intro');
                    if (introDiv) introDiv.style.display = 'none';
                    renderSidebarVisionResult(data, panel, capturedSceneId);
                    return;
                }
            }
        } catch (e) {
            // No cached results - show intro (already visible)
            log(`No cached analysis: ${e.message}`);
        }
    }

    /**
     * Set up event listeners for Analyze sidebar tab
     */
    function setupSidebarAnalyzeListeners(container, sceneId) {
        // Analyze button - start analysis from intro screen
        const analyzeBtn = container.querySelector('.stash-copilot-sidebar-analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => {
                // Hide intro
                const introDiv = container.querySelector('.stash-copilot-sidebar-intro');
                if (introDiv) introDiv.style.display = 'none';
                // Start analysis
                startSidebarVisionAnalysis(sceneId);
            });
        }

        // Details button - open analysis details modal
        const detailsBtn = container.querySelector('.stash-copilot-sidebar-details');
        if (detailsBtn) {
            detailsBtn.addEventListener('click', () => {
                openVisionDetailsModal(sceneId);
            });
        }

        // Settings button toggles options panel
        const settingsBtn = container.querySelector('.stash-copilot-sidebar-settings');
        const optionsPanel = container.querySelector('.stash-copilot-sidebar-options');
        if (settingsBtn && optionsPanel) {
            settingsBtn.addEventListener('click', () => {
                const isVisible = optionsPanel.style.display !== 'none';
                optionsPanel.style.display = isVisible ? 'none' : 'block';
            });
        }

        // Prompts toggle
        const promptsToggle = container.querySelector('.stash-copilot-sidebar-prompts-toggle');
        const promptsInner = container.querySelector('.stash-copilot-sidebar-prompts-inner');
        if (promptsToggle && promptsInner) {
            promptsToggle.addEventListener('click', () => {
                const isVisible = promptsInner.style.display !== 'none';
                promptsInner.style.display = isVisible ? 'none' : 'block';
                promptsToggle.querySelector('span').textContent = isVisible ? '▶ Edit prompts' : '▼ Edit prompts';
            });
        }

        // Load custom prompts into textareas
        const customPrompts = loadCustomPrompts();
        const systemPromptEl = container.querySelector('.stash-copilot-sidebar-prompt-system');
        const descriptionPromptEl = container.querySelector('.stash-copilot-sidebar-prompt-description');

        if (systemPromptEl && customPrompts.system) {
            systemPromptEl.value = customPrompts.system;
        }
        if (descriptionPromptEl && customPrompts.description) {
            descriptionPromptEl.value = customPrompts.description;
        }

        // Save prompts button
        const savePromptsBtn = container.querySelector('.stash-copilot-sidebar-prompt-save');
        if (savePromptsBtn) {
            savePromptsBtn.addEventListener('click', () => {
                const prompts = {};
                if (systemPromptEl?.value?.trim()) {
                    prompts.system = systemPromptEl.value.trim();
                }
                if (descriptionPromptEl?.value?.trim()) {
                    prompts.description = descriptionPromptEl.value.trim();
                }
                saveCustomPrompts(prompts);
                log('Custom prompts saved');
            });
        }

        // Reset prompts button
        const resetPromptsBtn = container.querySelector('.stash-copilot-sidebar-prompt-reset');
        if (resetPromptsBtn) {
            resetPromptsBtn.addEventListener('click', () => {
                resetCustomPrompts();
                if (systemPromptEl) systemPromptEl.value = '';
                if (descriptionPromptEl) descriptionPromptEl.value = '';
                log('Custom prompts reset');
            });
        }

        // Reanalyze button
        const reanalyzeBtn = container.querySelector('.stash-copilot-sidebar-reanalyze');
        if (reanalyzeBtn) {
            reanalyzeBtn.addEventListener('click', () => {
                startSidebarVisionAnalysis(sceneId, true);
            });
        }

        // Chat send button
        const sendBtn = container.querySelector('.stash-copilot-sidebar-send');
        const input = container.querySelector('.stash-copilot-sidebar-input');
        if (sendBtn && input) {
            sendBtn.addEventListener('click', () => {
                const message = input.value.trim();
                if (message) {
                    sendSidebarVisionMessage(sceneId, message);
                    input.value = '';
                }
            });

            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const message = input.value.trim();
                    if (message) {
                        sendSidebarVisionMessage(sceneId, message);
                        input.value = '';
                    }
                }
            });
        }

        // Tags section collapse toggle
        const tagsHeader = container.querySelector('.stash-copilot-sidebar-section-header[data-section="tags"]');
        if (tagsHeader) {
            tagsHeader.addEventListener('click', () => {
                const tagsDiv = container.querySelector('.stash-copilot-sidebar-tags');
                const collapseBtn = tagsHeader.querySelector('.stash-copilot-sidebar-collapse-btn');
                if (tagsDiv && collapseBtn) {
                    const isCollapsed = tagsDiv.style.display === 'none';
                    tagsDiv.style.display = isCollapsed ? 'flex' : 'none';
                    collapseBtn.textContent = isCollapsed ? '▼' : '▶';
                }
            });
        }

        // Corrections toggle (for verification status display)
        container.addEventListener('click', (e) => {
            const toggle = e.target.closest('.stash-copilot-corrections-toggle');
            if (toggle) {
                const list = toggle.nextElementSibling;
                if (list && list.classList.contains('stash-copilot-corrections-list')) {
                    const isExpanded = toggle.dataset.expanded === 'true';
                    list.style.display = isExpanded ? 'none' : 'block';
                    toggle.dataset.expanded = !isExpanded;
                    toggle.querySelector('span').textContent = isExpanded ? 'Show corrections \u25BC' : 'Hide corrections \u25B2';
                }
            }
        });
    }

    /**
     * Set up event listeners for Similar sidebar tab
     */
    function setupSidebarSimilarListeners(container, sceneId) {
        // Sub-tab switching
        const subtabs = container.querySelectorAll('.stash-copilot-sidebar-subtab');
        subtabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const filter = tab.dataset.filter;
                if (!filter) return;

                subtabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                similarState.activeTab = filter;

                // Check if this tab has been loaded yet
                const tabState = similarState.tabs[filter];
                if (!tabState || !tabState.loaded) {
                    // Need to fetch data for this tab
                    startSidebarSimilarSearch(sceneId, false);
                } else {
                    // Already have data, just render it
                    renderSidebarSimilarResults(container);
                }
            });
        });

        // Filter toggle
        const filterToggle = container.querySelector('.stash-copilot-sidebar-filter-toggle');
        const filtersDiv = container.querySelector('.stash-copilot-sidebar-filters');
        if (filterToggle && filtersDiv) {
            filterToggle.addEventListener('click', () => {
                filtersDiv.style.display = filtersDiv.style.display === 'none' ? 'block' : 'none';
            });
        }

        // Weight slider
        const slider = container.querySelector('.stash-copilot-sidebar-slider-input');
        const sliderValue = container.querySelector('.stash-copilot-sidebar-slider-value');
        if (slider && sliderValue) {
            slider.addEventListener('input', () => {
                const value = parseInt(slider.value);
                sliderValue.textContent = `${value}%`;
                similarState.visualWeight = value / 100;
                localStorage.setItem('stash-copilot-visual-weight', similarState.visualWeight.toString());
            });

            slider.addEventListener('change', () => {
                // Refresh results with new weight
                startSidebarSimilarSearch(sceneId, true);
            });
        }

        // Pagination buttons
        const prevBtn = container.querySelector('.stash-copilot-sidebar-page-btn.prev');
        const nextBtn = container.querySelector('.stash-copilot-sidebar-page-btn.next');
        if (prevBtn && nextBtn) {
            prevBtn.addEventListener('click', () => {
                const tabData = similarState.tabs[similarState.activeTab];
                if (tabData.currentPage > 1) {
                    tabData.currentPage--;
                    renderSidebarSimilarResults(container);
                }
            });

            nextBtn.addEventListener('click', () => {
                const tabData = similarState.tabs[similarState.activeTab];
                const totalPages = Math.ceil(tabData.allResults.length / similarState.resultsPerPage);
                if (tabData.currentPage < totalPages) {
                    tabData.currentPage++;
                    renderSidebarSimilarResults(container);
                }
            });
        }

        // Set up autocomplete for filter inputs
        setupSidebarFilterAutocomplete(container, 'performer');
        setupSidebarFilterAutocomplete(container, 'tag');
    }

    /**
     * Set up event listeners for Recs sidebar tab
     */
    function setupSidebarRecsListeners(container, sceneId) {
        // Recency dropdown
        const recencySelect = container.querySelector('.stash-copilot-sidebar-recency-select');
        if (recencySelect) {
            recencySelect.addEventListener('change', () => {
                sceneRecsState.timeDecayDays = parseInt(recencySelect.value);
                localStorage.setItem('stash-copilot-sidebar-rec-decay', recencySelect.value);

                // Hide profile info to prevent stale data display
                const profileInfo = container.querySelector('.stash-copilot-sidebar-profile-info');
                if (profileInfo) profileInfo.style.display = 'none';

                startSidebarRecsSearch(sceneId, true);
            });
        }

        // Mode tab switching (Discover / Re-watch)
        const modeTabs = container.querySelectorAll('.stash-copilot-sidebar-subtab');
        modeTabs.forEach(btn => {
            btn.addEventListener('click', () => {
                const newMode = btn.dataset.mode;
                if (!newMode || newMode === sceneRecsState.mode) return;

                modeTabs.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                sceneRecsState.mode = newMode;
                sceneRecsState.currentPage = 1;

                // Update slider labels for the new mode
                updateSidebarSliderLabels(container, newMode);

                // Re-fetch with new mode
                startSidebarRecsSearch(sceneId, true);
            });
        });

        // Filter toggle
        const filterToggle = container.querySelector('.stash-copilot-sidebar-filter-toggle');
        const filtersDiv = container.querySelector('.stash-copilot-sidebar-filters');
        if (filterToggle && filtersDiv) {
            filterToggle.addEventListener('click', () => {
                filtersDiv.style.display = filtersDiv.style.display === 'none' ? 'block' : 'none';
            });
        }

        // Weight slider - mode-dependent: Discover uses seedWeight, Re-watch uses engagementWeight (inverted)
        const slider = container.querySelector('.stash-copilot-sidebar-slider-input');
        const sliderValue = container.querySelector('.stash-copilot-sidebar-slider-value');
        if (slider && sliderValue) {
            slider.addEventListener('input', () => {
                const value = parseInt(slider.value);
                sliderValue.textContent = `${value}%`;

                if (sceneRecsState.mode === 'rewatch') {
                    // Slider shows similarity%, engagement = 1 - similarity
                    const newEngWeight = 1 - (value / 100);
                    sceneRecsState.engagementWeight = newEngWeight;
                    localStorage.setItem('stash-copilot-engagement-weight', sceneRecsState.engagementWeight.toString());
                } else {
                    sceneRecsState.seedWeight = value / 100;
                    localStorage.setItem('stash-copilot-seed-weight', sceneRecsState.seedWeight.toString());
                }
            });

            slider.addEventListener('change', () => {
                // Refresh results with new weight
                startSidebarRecsSearch(sceneId, true);
            });
        }

        // Pagination buttons
        const prevBtn = container.querySelector('.stash-copilot-sidebar-page-btn.prev');
        const nextBtn = container.querySelector('.stash-copilot-sidebar-page-btn.next');
        if (prevBtn && nextBtn) {
            prevBtn.addEventListener('click', () => {
                if (sceneRecsState.currentPage > 1) {
                    sceneRecsState.currentPage--;
                    renderSidebarRecsResults(container);
                }
            });

            nextBtn.addEventListener('click', () => {
                const totalPages = Math.ceil(sceneRecsState.results.length / sceneRecsState.resultsPerPage);
                if (sceneRecsState.currentPage < totalPages) {
                    sceneRecsState.currentPage++;
                    renderSidebarRecsResults(container);
                }
            });
        }

        // Set up autocomplete for filter inputs
        setupSidebarFilterAutocomplete(container, 'performer');
        setupSidebarFilterAutocomplete(container, 'tag');
    }

    /**
     * Update sidebar slider labels based on the current mode.
     * Discover: Profile ↔ Scene; Re-watch: Engagement ↔ Similarity
     */
    function updateSidebarSliderLabels(container, mode) {
        const leftLabel = container.querySelector('.stash-copilot-sidebar-slider-label[data-slider-label="left"]');
        const rightLabel = container.querySelector('.stash-copilot-sidebar-slider-label[data-slider-label="right"]');
        const slider = container.querySelector('.stash-copilot-sidebar-slider-input');
        const sliderValueEl = container.querySelector('.stash-copilot-sidebar-slider-value');

        if (mode === 'rewatch') {
            if (leftLabel) leftLabel.textContent = 'Engagement';
            if (rightLabel) rightLabel.textContent = 'Similarity';
            // Set slider to show similarity % (inverted engagement)
            const simPct = Math.round((1 - sceneRecsState.engagementWeight) * 100);
            if (slider) slider.value = simPct;
            if (sliderValueEl) sliderValueEl.textContent = `${simPct}%`;
        } else {
            if (leftLabel) leftLabel.textContent = 'Profile';
            if (rightLabel) rightLabel.textContent = 'Scene';
            const seedPct = Math.round(sceneRecsState.seedWeight * 100);
            if (slider) slider.value = seedPct;
            if (sliderValueEl) sliderValueEl.textContent = `${seedPct}%`;
        }
    }

    /**
     * Setup autocomplete for sidebar filters
     */
    function setupSidebarFilterAutocomplete(container, type) {
        const input = container.querySelector(`.stash-copilot-sidebar-filter-${type}`);
        const autocompleteDiv = input?.parentElement?.querySelector('.stash-copilot-sidebar-autocomplete');

        if (!input || !autocompleteDiv) return;

        let debounceTimer;

        input.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            const query = input.value.trim();

            if (query.length < 2) {
                autocompleteDiv.innerHTML = '';
                autocompleteDiv.style.display = 'none';
                return;
            }

            debounceTimer = setTimeout(async () => {
                try {
                    const endpoint = type === 'performer' ? 'performers' : 'tags';
                    const response = await fetch('/graphql', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: `query { find${type === 'performer' ? 'Performers' : 'Tags'}(filter: { q: "${query}", per_page: 10 }) { ${type === 'performer' ? 'performers' : 'tags'} { id name } } }`
                        })
                    });
                    const data = await response.json();
                    const items = type === 'performer' ? data.data.findPerformers.performers : data.data.findTags.tags;

                    if (items.length > 0) {
                        autocompleteDiv.innerHTML = items.map(item =>
                            `<div class="stash-copilot-sidebar-autocomplete-item" data-id="${item.id}" data-name="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>`
                        ).join('');
                        autocompleteDiv.style.display = 'block';

                        // Add click handlers
                        autocompleteDiv.querySelectorAll('.stash-copilot-sidebar-autocomplete-item').forEach(item => {
                            item.addEventListener('click', () => {
                                const name = item.dataset.name;
                                addSidebarFilter(container, type, name);
                                input.value = '';
                                autocompleteDiv.innerHTML = '';
                                autocompleteDiv.style.display = 'none';
                            });
                        });
                    } else {
                        autocompleteDiv.innerHTML = '';
                        autocompleteDiv.style.display = 'none';
                    }
                } catch (error) {
                    log(`Autocomplete error: ${error.message}`, 'error');
                }
            }, 300);
        });

        // Hide on blur
        input.addEventListener('blur', () => {
            setTimeout(() => {
                autocompleteDiv.style.display = 'none';
            }, 200);
        });
    }

    /**
     * Add a filter tag to the sidebar
     */
    function addSidebarFilter(container, type, name) {
        const tagsContainer = container.querySelector(`.stash-copilot-sidebar-excluded-${type}s`);
        if (!tagsContainer) return;

        // Check if already added
        const existing = Array.from(tagsContainer.querySelectorAll('.stash-copilot-sidebar-filter-tag'))
            .find(tag => tag.dataset.name === name);
        if (existing) return;

        // Add to state
        if (type === 'performer') {
            if (!similarState.excludePerformers.includes(name)) {
                similarState.excludePerformers.push(name);
            }
            if (!sceneRecsState.excludePerformers.includes(name)) {
                sceneRecsState.excludePerformers.push(name);
            }
        } else {
            if (!similarState.excludeTags.includes(name)) {
                similarState.excludeTags.push(name);
            }
            if (!sceneRecsState.excludeTags.includes(name)) {
                sceneRecsState.excludeTags.push(name);
            }
        }

        // Create tag element
        const tag = document.createElement('span');
        tag.className = 'stash-copilot-sidebar-filter-tag';
        tag.dataset.name = name;
        tag.innerHTML = `${escapeHtml(name)} <button class="stash-copilot-sidebar-filter-remove">&times;</button>`;

        tag.querySelector('.stash-copilot-sidebar-filter-remove').addEventListener('click', () => {
            tag.remove();
            if (type === 'performer') {
                similarState.excludePerformers = similarState.excludePerformers.filter(p => p !== name);
                sceneRecsState.excludePerformers = sceneRecsState.excludePerformers.filter(p => p !== name);
            } else {
                similarState.excludeTags = similarState.excludeTags.filter(t => t !== name);
                sceneRecsState.excludeTags = sceneRecsState.excludeTags.filter(t => t !== name);
            }
            updateSidebarFilterBadge(container);
        });

        tagsContainer.appendChild(tag);
        updateSidebarFilterBadge(container);
    }

    /**
     * Update the filter badge count
     */
    function updateSidebarFilterBadge(container) {
        const badge = container.querySelector('.stash-copilot-sidebar-filter-badge');
        if (!badge) return;

        const count = similarState.excludePerformers.length + similarState.excludeTags.length;
        badge.textContent = count.toString();
        badge.style.display = count > 0 ? 'inline' : 'none';
    }

    // ===== Sidebar Vision Analysis Functions =====

    /**
     * Restore the loading UI state when analysis is running but DOM was recreated.
     * This handles the case where React re-renders destroy our injected DOM.
     */
    function restoreAnalysisLoadingState() {
        const panel = document.getElementById('scene-copilot-analyze-panel');
        if (!panel) return;

        log('Restoring analysis loading state after DOM recreation');

        // Hide intro
        const introDiv = panel.querySelector('.stash-copilot-sidebar-intro');
        if (introDiv) introDiv.style.display = 'none';

        // Show loading state
        const loadingDiv = panel.querySelector('.stash-copilot-sidebar-loading');
        if (loadingDiv) {
            loadingDiv.style.display = 'flex';
            // Calculate elapsed time for display
            const elapsed = visionState.analysisStartTime
                ? Math.floor((Date.now() - visionState.analysisStartTime) / 1000)
                : 0;
            const statusText = elapsed > 0 ? `Analyzing... (${elapsed}s)` : 'Analyzing...';
            loadingDiv.innerHTML = `
                <div class="stash-copilot-spinner"></div>
                <span class="stash-copilot-sidebar-status">${statusText}</span>
                <div class="stash-copilot-sidebar-progress-container">
                    <div class="stash-copilot-sidebar-progress-bar" style="width: 0%"></div>
                </div>
                <span class="stash-copilot-sidebar-progress-text"></span>
            `;
        }

        // Hide any result sections that might have been rendered
        const analysisDiv = panel.querySelector('.stash-copilot-sidebar-analysis');
        if (analysisDiv) analysisDiv.innerHTML = '';

        const tagsSection = panel.querySelector('.stash-copilot-sidebar-tags-section');
        if (tagsSection) tagsSection.style.display = 'none';

        const messagesDiv = panel.querySelector('.stash-copilot-sidebar-messages');
        if (messagesDiv) messagesDiv.innerHTML = '';
    }

    function startSidebarVisionAnalysis(sceneId, forceReanalyze = false) {
        log(`Starting sidebar vision analysis for scene ${sceneId}, forceReanalyze=${forceReanalyze}`);
        visionState.sceneId = sceneId;
        visionState.isAnalyzing = true;
        visionState.analysisStartTime = Date.now();
        // For re-analyses, we need to wait to see new analysis start before rendering progress
        // For new analyses, there's no stale data so we can render immediately
        visionState.hasSeenNewAnalysisStart = !forceReanalyze;
        // Reset resultRendered flag when starting new analysis
        visionState.resultRendered = false;

        const panel = document.getElementById('scene-copilot-analyze-panel');
        if (!panel) {
            log('Panel not found!');
            return;
        }

        // Hide intro if visible
        const introDiv = panel.querySelector('.stash-copilot-sidebar-intro');
        if (introDiv) introDiv.style.display = 'none';

        // Show loading state
        const loadingDiv = panel.querySelector('.stash-copilot-sidebar-loading');
        const analysisDiv = panel.querySelector('.stash-copilot-sidebar-analysis');

        if (loadingDiv) loadingDiv.style.display = 'flex';

        // When re-analyzing, reset UI to loading state
        if (forceReanalyze) {
            // Clear analysis content
            if (analysisDiv) analysisDiv.innerHTML = '';

            // Hide tags section
            const tagsSection = panel.querySelector('.stash-copilot-sidebar-tags-section');
            if (tagsSection) tagsSection.style.display = 'none';

            // Clear chat messages
            const messagesDiv = panel.querySelector('.stash-copilot-sidebar-messages');
            if (messagesDiv) messagesDiv.innerHTML = '';

            // Reset loading div to initial state (remove any "tags loading" class and restore original content)
            if (loadingDiv) {
                loadingDiv.classList.remove('stash-copilot-sidebar-loading-tags');
                loadingDiv.innerHTML = `
                    <div class="stash-copilot-spinner"></div>
                    <span class="stash-copilot-sidebar-status">Re-analyzing scene...</span>
                    <div class="stash-copilot-sidebar-progress-container">
                        <div class="stash-copilot-sidebar-progress-bar" style="width: 0%"></div>
                    </div>
                    <span class="stash-copilot-sidebar-progress-text"></span>
                `;
            }

            // Reset visionState
            visionState.messages = [];
            visionState.description = null;
            visionState.suggestedTags = [];
            visionState.tagTimestamps = {};
            visionState.conversationId = null;
        }

        // Check for existing analysis or trigger new one
        pollSidebarVisionAnalysis(sceneId, forceReanalyze);
    }

    async function pollSidebarVisionAnalysis(sceneId, forceReanalyze = false) {
        // Capture scene ID in closure to validate throughout async operations
        const capturedSceneId = sceneId;

        const resultFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;
        const panel = document.getElementById('scene-copilot-analyze-panel');
        if (!panel) return;

        // First check if we have existing results
        if (!forceReanalyze) {
            try {
                const response = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'complete') {
                        // Validate we're still on the same scene
                        if (sidebarTabState.sceneId !== capturedSceneId) {
                            log(`Scene changed during poll (${capturedSceneId} -> ${sidebarTabState.sceneId}), discarding results`);
                            return;
                        }
                        log('Found existing vision analysis');
                        renderSidebarVisionResult(data, panel, capturedSceneId);
                        return;
                    }
                }
            } catch (e) {
                log(`No existing vision analysis: ${e.message}`);
            }
        }

        // Trigger new analysis via GraphQL
        log('Triggering new vision analysis');
        try {
            // Get analysis options from UI
            const options = getAnalysisOptions(panel);

            const args = [
                { key: 'mode', value: { str: 'scene_vision' } },
                { key: 'scene_id', value: { str: String(sceneId) } },
                { key: 'user_confirmed', value: { str: 'true' } }
            ];

            // Force fresh analysis when reanalyzing
            if (forceReanalyze) {
                args.push({ key: 'clear_history', value: { str: 'true' } });
            }

            // Add analysis options
            if (options.quick_mode) {
                args.push({ key: 'quick_mode', value: { str: 'true' } });
            }
            if (options.skip_verification) {
                args.push({ key: 'skip_verification', value: { str: 'true' } });
            }
            if (options.frame_count) {
                args.push({ key: 'frame_count', value: { str: String(options.frame_count) } });
            }
            if (options.custom_prompts && Object.keys(options.custom_prompts).length > 0) {
                args.push({ key: 'custom_prompts', value: { str: JSON.stringify(options.custom_prompts) } });
            }

            const result = await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: args
            });

            if (!result || !result.runPluginTask) {
                throw new Error('Failed to start vision task');
            }

            // Reset UI state before polling to ensure clean slate
            // (The initial reset in startSidebarVisionAnalysis may have been overwritten by React)
            if (forceReanalyze) {
                const analysisDiv = panel.querySelector('.stash-copilot-sidebar-analysis');
                const tagsSection = panel.querySelector('.stash-copilot-sidebar-tags-section');
                const messagesDiv = panel.querySelector('.stash-copilot-sidebar-messages');
                const loadingDiv = panel.querySelector('.stash-copilot-sidebar-loading');

                if (analysisDiv) analysisDiv.innerHTML = '';
                if (tagsSection) tagsSection.style.display = 'none';
                if (messagesDiv) messagesDiv.innerHTML = '';
                if (loadingDiv) {
                    loadingDiv.style.display = 'flex';
                    loadingDiv.classList.remove('stash-copilot-sidebar-loading-tags');
                    loadingDiv.innerHTML = `
                        <div class="stash-copilot-spinner"></div>
                        <span class="stash-copilot-sidebar-status">Re-analyzing scene...</span>
                        <div class="stash-copilot-sidebar-progress-container">
                            <div class="stash-copilot-sidebar-progress-bar" style="width: 0%"></div>
                        </div>
                        <span class="stash-copilot-sidebar-progress-text"></span>
                    `;
                }
            }

            // Track when re-analysis started to avoid rendering stale results
            const analysisStartTime = forceReanalyze ? Date.now() : 0;
            let hasSeenRunningStatus = false;

            // Reset the global flag when starting re-analysis
            if (forceReanalyze) {
                visionState.hasSeenNewAnalysisStart = false;
            }

            // Start polling for results (with small delay to let backend start)
            await new Promise(resolve => setTimeout(resolve, 500));

            const pollInterval = setInterval(async () => {
                try {
                    // Validate we're still on the same scene before processing
                    if (sidebarTabState.sceneId !== capturedSceneId) {
                        log(`Scene changed during polling (${capturedSceneId} -> ${sidebarTabState.sceneId}), stopping poll`);
                        clearInterval(pollInterval);
                        visionState.isAnalyzing = false;
                        return;
                    }

                    const resultResponse = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resultResponse.ok) {
                        const data = await resultResponse.json();

                        // Track if we've seen running status (new analysis started)
                        // Backend statuses: pending, extracting, describing, tagging, complete, error
                        if (data.status === 'extracting' || data.status === 'describing' || data.status === 'tagging') {
                            hasSeenRunningStatus = true;
                            visionState.hasSeenNewAnalysisStart = true;
                        }

                        // Skip updating progress for stale data during re-analysis
                        // (only update when we've confirmed the new analysis has started)
                        if (forceReanalyze && !hasSeenRunningStatus && data.status === 'complete') {
                            log('Ignoring stale complete status during re-analysis');
                            return;
                        }

                        // Update progress (with scene ID validation)
                        updateSidebarVisionProgress(data, panel, capturedSceneId);

                        if (data.status === 'complete') {
                            clearInterval(pollInterval);
                            visionState.isAnalyzing = false;
                            visionState.conversationId = data.conversation_id;
                            renderSidebarVisionResult(data, panel, capturedSceneId);
                        } else if (data.status === 'error') {
                            clearInterval(pollInterval);
                            visionState.isAnalyzing = false;
                            showSidebarError(panel, data.error || 'Analysis failed');
                        }
                    }
                } catch (e) {
                    log(`Vision poll error: ${e.message}`);
                }
            }, 500);

            // Timeout after 5 minutes
            setTimeout(() => {
                clearInterval(pollInterval);
                if (visionState.isAnalyzing) {
                    visionState.isAnalyzing = false;
                    showSidebarError(panel, 'Analysis timed out');
                }
            }, 300000);

        } catch (e) {
            log(`Vision task error: ${e.message}`, 'error');
            showSidebarError(panel, e.message);
        }
    }

    /**
     * Render verification status badge and corrections from multi-stage analysis
     */
    function renderVerificationStatus(history) {
        if (!history.verification_status || history.verification_status === 'pending') {
            return '';
        }

        const statusConfig = {
            'verified': { icon: '\u2713', label: 'Verified', class: 'verified' },
            'corrections': {
                icon: '\u26A0',
                label: `${history.corrections?.length || 0} corrections`,
                class: 'corrections'
            },
            'skipped': { icon: '\u26A1', label: 'Unverified', class: 'skipped' },
            'failed': { icon: '\u26A1', label: 'Unverified', class: 'skipped' },
        };

        const config = statusConfig[history.verification_status] || statusConfig.skipped;

        let html = `
            <div class="stash-copilot-verification-status ${config.class}">
                <span class="stash-copilot-verification-icon">${config.icon}</span>
                <span class="stash-copilot-verification-label">${config.label}</span>
            </div>
        `;

        // Add expandable corrections if present
        if (history.verification_status === 'corrections' && history.corrections?.length > 0) {
            html += `
                <div class="stash-copilot-corrections-toggle" data-expanded="false">
                    <span>Show corrections \u25BC</span>
                </div>
                <div class="stash-copilot-corrections-list" style="display: none;">
                    ${history.corrections.map(c => `
                        <div class="stash-copilot-correction">
                            <span class="stash-copilot-correction-claim">\u274C "${escapeHtml(c.claim)}"</span>
                            <span class="stash-copilot-correction-fix">\u2192 ${escapeHtml(c.correction)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        return html;
    }

    function updateSidebarVisionProgress(data, panel, expectedSceneId = null) {
        // Skip progress updates after result has been rendered
        if (visionState.resultRendered) {
            log('Result already rendered, skipping progress update');
            return;
        }

        // Validate scene ID if provided
        if (expectedSceneId !== null && sidebarTabState.sceneId !== expectedSceneId) {
            log(`Scene mismatch in updateSidebarVisionProgress: expected ${expectedSceneId}, current ${sidebarTabState.sceneId}`);
            return;
        }

        const statusEl = panel.querySelector('.stash-copilot-sidebar-status');
        const progressBar = panel.querySelector('.stash-copilot-sidebar-progress-bar');
        const progressText = panel.querySelector('.stash-copilot-sidebar-progress-text');
        const loadingDiv = panel.querySelector('.stash-copilot-sidebar-loading');
        const analysisDiv = panel.querySelector('.stash-copilot-sidebar-analysis');
        const tagsSection = panel.querySelector('.stash-copilot-sidebar-tags-section');
        const tagsDiv = panel.querySelector('.stash-copilot-sidebar-tags');

        if (statusEl) {
            statusEl.textContent = data.status_message || getAnalysisStatusMessage(data.status);
        }

        if (progressBar && typeof data.progress === 'number') {
            progressBar.style.width = `${data.progress}%`;
        }

        if (progressText) {
            if (data.stage === 'extracting' && data.total_frames) {
                const currentFrame = Math.round((data.progress - 10) / 60 * data.total_frames);
                progressText.textContent = `Frame ${currentFrame}/${data.total_frames}`;
            } else if (data.stage === 'describing') {
                progressText.textContent = `${data.total_frames || 0} frames`;
            } else if (data.stage === 'tagging') {
                progressText.textContent = 'Suggesting tags...';
            }
        }

        // Show description progressively when ready
        // Skip rendering if we're re-analyzing and haven't seen the new analysis start yet
        // (prevents stale description_complete=true from rendering old content)
        const isStaleProgressData = visionState.isAnalyzing && !visionState.hasSeenNewAnalysisStart;
        if (isStaleProgressData && (data.description_complete || data.tags_complete)) {
            log('Skipping stale progress render during re-analysis');
            return;
        }

        if (data.description_complete && data.description && analysisDiv) {
            // Render classification badges if available
            const badgesHtml = renderClassificationBadges(data.classification);

            // Show description with badges
            analysisDiv.innerHTML = `
                ${badgesHtml}
                <div class="stash-copilot-sidebar-description">
                    ${renderMarkdown(data.description)}
                </div>
            `;

            // If tags aren't ready yet, show a secondary loading indicator for tags
            // Only update innerHTML once to avoid restarting the spinner animation
            if (!data.tags_complete && loadingDiv && !loadingDiv.classList.contains('stash-copilot-sidebar-loading-tags')) {
                loadingDiv.innerHTML = `
                    <div class="stash-copilot-spinner stash-copilot-spinner-small"></div>
                    <span class="stash-copilot-sidebar-status">Generating tag suggestions...</span>
                `;
                loadingDiv.classList.add('stash-copilot-sidebar-loading-tags');
            } else if (data.tags_complete && loadingDiv) {
                // Both complete - hide loading
                loadingDiv.style.display = 'none';
            }
        }

        // Show tags progressively when ready
        if (data.tags_complete && data.suggested_tags && tagsSection && tagsDiv) {
            if (loadingDiv) loadingDiv.style.display = 'none';
            tagsSection.style.display = 'block';
            renderSidebarTags(data.suggested_tags, data.tag_confidences || {}, data.tag_timestamps || {}, tagsDiv, panel);
        }
    }

    function renderSidebarVisionResult(data, panel, expectedSceneId = null) {
        log(`renderSidebarVisionResult called, isAnalyzing=${visionState.isAnalyzing}, status=${data?.status}, expectedScene=${expectedSceneId}`);

        // Validate scene ID if provided - prevent rendering results for wrong scene
        if (expectedSceneId !== null && sidebarTabState.sceneId !== expectedSceneId) {
            log(`Scene mismatch in renderSidebarVisionResult: expected ${expectedSceneId}, current ${sidebarTabState.sceneId}`);
            return;
        }

        // Mark result as rendered to prevent further progress updates
        visionState.resultRendered = true;

        const loadingDiv = panel.querySelector('.stash-copilot-sidebar-loading');
        const analysisDiv = panel.querySelector('.stash-copilot-sidebar-analysis');
        const tagsSection = panel.querySelector('.stash-copilot-sidebar-tags-section');
        const tagsDiv = panel.querySelector('.stash-copilot-sidebar-tags');
        const messagesDiv = panel.querySelector('.stash-copilot-sidebar-messages');
        const chatInput = panel.querySelector('.stash-copilot-sidebar-input');
        const inputContainer = panel.querySelector('.stash-copilot-sidebar-input-container');

        // Hide loading
        if (loadingDiv) loadingDiv.style.display = 'none';

        // Render classification badges if available
        const classificationBadgesHtml = renderClassificationBadges(data.classification);

        // Render description
        if (data.description && analysisDiv) {
            analysisDiv.innerHTML = `
                ${classificationBadgesHtml}
                <div class="stash-copilot-sidebar-description">
                    ${renderMarkdown(data.description)}
                </div>
            `;
        }

        // Render tags with styled + buttons
        if (data.suggested_tags && data.suggested_tags.length > 0 && tagsSection && tagsDiv) {
            tagsSection.style.display = 'block';
            renderSidebarTags(data.suggested_tags, data.tag_confidences || {}, data.tag_timestamps || {}, tagsDiv, panel);
        }

        // Render follow-up messages
        if (data.messages && data.messages.length > 2 && messagesDiv) {
            const followUpMessages = data.messages.slice(2);
            messagesDiv.innerHTML = followUpMessages.map(msg => `
                <div class="stash-copilot-sidebar-message ${msg.role}">
                    ${renderMarkdown(msg.content || '')}
                </div>
            `).join('');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Display suggested question in chat input
        if (data.suggested_question && chatInput && inputContainer) {
            chatInput.value = data.suggested_question;
            chatInput.classList.add('has-suggestion');
            inputContainer.classList.add('has-suggestion');

            // Add sparkle indicator if not already present
            if (!inputContainer.querySelector('.stash-copilot-suggestion-indicator')) {
                const indicator = document.createElement('span');
                indicator.className = 'stash-copilot-suggestion-indicator';
                indicator.innerHTML = '✨';
                indicator.title = 'AI-suggested question';
                inputContainer.insertBefore(indicator, chatInput);
            }

            // Clear suggestion styling when user starts typing
            const clearSuggestion = () => {
                chatInput.classList.remove('has-suggestion');
                inputContainer.classList.remove('has-suggestion');
                const indicator = inputContainer.querySelector('.stash-copilot-suggestion-indicator');
                if (indicator) indicator.remove();
            };

            chatInput.addEventListener('input', clearSuggestion, { once: true });
            chatInput.addEventListener('focus', () => {
                chatInput.select();
            }, { once: true });
        }

        visionState.description = data.description;
        visionState.suggestedTags = data.suggested_tags || [];
        visionState.conversationId = data.conversation_id;
    }

    /**
     * Render suggested tags with styled + buttons for adding to scene
     */
    function renderSidebarTags(tags, confidences, timestamps, tagsDiv, panel) {
        if (!tags || tags.length === 0) {
            tagsDiv.innerHTML = '<span class="stash-copilot-sidebar-no-tags">No tag suggestions</span>';
            return;
        }

        tagsDiv.innerHTML = tags.map((tag, idx) => {
            const confidence = confidences[tag] || null;
            const timestamp = timestamps[tag] || null;
            const tooltipParts = [];
            if (timestamp) tooltipParts.push(`@${formatTimestamp(timestamp)}`);
            if (confidence) tooltipParts.push(`${confidence}% confidence`);
            const tooltip = tooltipParts.join(' • ') || 'Click + to add tag';

            return `
                <div class="stash-copilot-sidebar-tag-item" data-tag="${escapeHtml(tag)}" title="${tooltip}" style="--tag-index: ${idx}">
                    <span class="stash-copilot-sidebar-tag-name">${escapeHtml(tag)}</span>
                    ${confidence ? `<span class="stash-copilot-sidebar-tag-confidence">${confidence}%</span>` : ''}
                    <button class="stash-copilot-sidebar-tag-add" title="Add tag to scene">+</button>
                </div>
            `;
        }).join('');

        // Add click handlers for the + buttons
        tagsDiv.querySelectorAll('.stash-copilot-sidebar-tag-add').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const tagItem = btn.closest('.stash-copilot-sidebar-tag-item');
                const tagName = tagItem.dataset.tag;

                // Show loading state
                btn.disabled = true;
                btn.innerHTML = '...';

                try {
                    await applyTag(tagName);
                    // Success - animate removal
                    tagItem.classList.add('stash-copilot-sidebar-tag-added');
                    setTimeout(() => tagItem.remove(), 300);
                } catch (err) {
                    // Error - reset button
                    btn.disabled = false;
                    btn.innerHTML = '+';
                    log(`Failed to add tag: ${err.message}`, 'error');
                }
            });
        });
    }

    async function sendSidebarVisionMessage(sceneId, message) {
        log(`Sending sidebar vision message: ${message}`);
        const panel = document.getElementById('scene-copilot-analyze-panel');
        if (!panel) return;

        const messagesDiv = panel.querySelector('.stash-copilot-sidebar-messages');
        const input = panel.querySelector('.stash-copilot-sidebar-input');

        // Add user message to UI
        if (messagesDiv) {
            messagesDiv.innerHTML += `
                <div class="stash-copilot-sidebar-message user">${escapeHtml(message)}</div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Disable input while processing
        if (input) input.disabled = true;

        try {
            const args = [
                { key: 'mode', value: { str: 'scene_vision' } },
                { key: 'scene_id', value: { str: String(sceneId) } },
                { key: 'message', value: { str: message } },
                { key: 'conversation_id', value: { str: visionState.conversationId || '' } }
            ];

            const result = await callGQL(`
                mutation RunPluginTask($plugin_id: ID!, $task_name: String!, $args: [PluginArgInput!]) {
                    runPluginTask(plugin_id: $plugin_id, task_name: $task_name, args: $args)
                }
            `, {
                plugin_id: PLUGIN_ID,
                task_name: 'Scene Vision Analysis',
                args: args
            });

            if (!result || !result.runPluginTask) {
                throw new Error('Failed to send message');
            }

            // Poll for response
            const resultFile = `${SCENE_VISION_PATH}/vision_history_${sceneId}.json`;
            const pollInterval = setInterval(async () => {
                try {
                    const resultResponse = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resultResponse.ok) {
                        const data = await resultResponse.json();
                        if (data.messages && data.messages.length > visionState.messages.length) {
                            clearInterval(pollInterval);
                            const newMessages = data.messages.slice(2);
                            if (messagesDiv) {
                                messagesDiv.innerHTML = newMessages.map(msg => `
                                    <div class="stash-copilot-sidebar-message ${msg.role}">
                                        ${renderMarkdown(msg.content || '')}
                                    </div>
                                `).join('');
                                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                            }
                            visionState.messages = data.messages;
                            if (input) input.disabled = false;
                        }
                    }
                } catch (e) {
                    log(`Follow-up poll error: ${e.message}`);
                }
            }, 500);

            setTimeout(() => clearInterval(pollInterval), 60000);

        } catch (e) {
            log(`Follow-up error: ${e.message}`, 'error');
            if (messagesDiv) {
                messagesDiv.innerHTML += `
                    <div class="stash-copilot-sidebar-message assistant stash-copilot-sidebar-error">
                        Error: ${escapeHtml(e.message)}
                    </div>
                `;
            }
            if (input) input.disabled = false;
        }
    }

    // ===== Sidebar Similar Scenes Functions =====

    async function startSidebarSimilarSearch(sceneId, refresh = false) {
        log(`Starting sidebar similar search for scene ${sceneId}`);
        similarState.sceneId = sceneId;

        const panel = document.getElementById('scene-copilot-similar-panel');
        if (!panel) return;

        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');

        // Show loading - recreate if it was removed by previous render
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="stash-copilot-sidebar-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Finding similar scenes...</span>
                </div>
            `;
        }
        if (paginationDiv) paginationDiv.style.display = 'none';

        // Reset tab state if refresh
        if (refresh) {
            similarState.tabs['all'] = { allResults: [], allSceneDetails: [], currentPage: 1, backendOffset: 0, hasMoreBackend: true, isSearching: false, loaded: false };
            similarState.tabs['different-performers'] = { allResults: [], allSceneDetails: [], currentPage: 1, backendOffset: 0, hasMoreBackend: true, isSearching: false, loaded: false };
        }

        // Trigger backend search via GraphQL
        try {
            const excludePerformers = similarState.activeTab === 'different-performers';
            const taskArgs = {
                scene_id: String(sceneId),
                limit: String(similarState.fetchBatchSize || 100),
                offset: '0',
                exclude_common_performers: excludePerformers ? 'true' : 'false',
                visual_weight: String(similarState.visualWeight)
            };

            await runPluginTask('Find Similar Scenes', taskArgs);

            // Poll for results
            pollSidebarSimilarResults(sceneId, panel);

        } catch (e) {
            log(`Similar search error: ${e.message}`, 'error');
            showSidebarError(panel, e.message);
        }
    }

    async function pollSidebarSimilarResults(sceneId, panel) {
        const resultFile = `/plugin/stash-copilot/assets/similar_results_${sceneId}.json`;

        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();

                    if (data.status === 'complete' || data.results) {
                        clearInterval(pollInterval);

                        // Store results - scene details are embedded in results from backend
                        const tabState = similarState.tabs[similarState.activeTab];
                        tabState.allResults = data.results || [];
                        tabState.allSceneDetails = tabState.allResults.map(r => r.scene || null);
                        tabState.hasMoreBackend = data.has_more || false;
                        tabState.loaded = true;

                        // Store model_key for display
                        similarState.modelKey = data.model_key || 'unknown';

                        // Render results directly (no separate fetch needed)
                        renderSidebarSimilarResultsUI(tabState, panel);
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        showSidebarError(panel, data.error || 'Search failed');
                    }
                }
            } catch (e) {
                log(`Similar poll error: ${e.message}`);
            }
        }, 300);

        setTimeout(() => clearInterval(pollInterval), 60000);
    }

    async function fetchSidebarSceneDetails(tabState, panel) {
        const sceneIds = tabState.allResults.map(r => parseInt(r.scene_id, 10)).filter(id => !isNaN(id));
        if (sceneIds.length === 0) {
            renderSidebarSimilarResultsUI(tabState, panel);
            return;
        }

        try {
            const response = await fetch('/graphql', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: `query FindScenes($ids: [Int!]!) {
                        findScenes(scene_filter: { id: { modifier: INCLUDES, value: $ids } }, filter: { per_page: -1 }) {
                            scenes {
                                id title date paths { screenshot } performers { name } studio { name } tags { name } play_count o_counter
                            }
                        }
                    }`,
                    variables: { ids: sceneIds }
                })
            });

            const data = await response.json();
            const scenes = data.data?.findScenes?.scenes || [];

            // Map scenes by ID
            const sceneMap = {};
            scenes.forEach(s => { sceneMap[s.id] = s; });

            // Order by results order
            tabState.allSceneDetails = tabState.allResults.map(r => sceneMap[r.scene_id]).filter(Boolean);

            renderSidebarSimilarResultsUI(tabState, panel);
        } catch (e) {
            log(`Scene details fetch error: ${e.message}`, 'error');
            showSidebarError(panel, 'Failed to load scene details');
        }
    }

    function renderSidebarSimilarResultsUI(tabState, panel) {
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');
        const modelInfoDiv = panel.querySelector('.stash-copilot-sidebar-model-info');

        if (!resultsDiv) return;

        // Hide loading
        const loadingDiv = resultsDiv.querySelector('.stash-copilot-sidebar-loading');
        if (loadingDiv) loadingDiv.style.display = 'none';

        // Show model info badge
        if (modelInfoDiv && similarState.modelKey) {
            const badge = modelInfoDiv.querySelector('.stash-copilot-model-badge');
            if (badge) {
                badge.textContent = `Model: ${similarState.modelKey}`;
                badge.title = `Embeddings generated with ${similarState.modelKey}`;
            }
            modelInfoDiv.style.display = 'flex';
        }

        // Apply filters
        let filteredResults = tabState.allResults;
        let filteredDetails = tabState.allSceneDetails;

        if (similarState.excludePerformers.length > 0 || similarState.excludeTags.length > 0) {
            const filtered = [];
            const filteredScenes = [];
            for (let i = 0; i < filteredResults.length; i++) {
                const scene = filteredDetails[i];
                if (!scene) continue;

                const performerNames = scene.performers?.map(p => p.name) || [];
                const tagNames = scene.tags?.map(t => t.name) || [];

                const excludeByPerformer = similarState.excludePerformers.some(p => performerNames.includes(p));
                const excludeByTag = similarState.excludeTags.some(t => tagNames.includes(t));

                if (!excludeByPerformer && !excludeByTag) {
                    filtered.push(filteredResults[i]);
                    filteredScenes.push(scene);
                }
            }
            filteredResults = filtered;
            filteredDetails = filteredScenes;
        }

        if (filteredResults.length === 0) {
            resultsDiv.innerHTML = `
                <div class="stash-copilot-sidebar-empty">
                    No similar scenes found
                </div>
            `;
            if (paginationDiv) paginationDiv.style.display = 'none';
            return;
        }

        // Paginate
        const start = (tabState.currentPage - 1) * similarState.resultsPerPage;
        const end = start + similarState.resultsPerPage;
        const pageResults = filteredResults.slice(start, end);
        const pageDetails = filteredDetails.slice(start, end);
        const totalPages = Math.ceil(filteredResults.length / similarState.resultsPerPage);

        // Render cards using unified card system
        resultsDiv.innerHTML = pageResults.map((result, idx) => {
            const scene = {
                ...(pageDetails[idx] || {}),
                id: pageDetails[idx]?.id || result.scene_id
            };
            return buildSceneCard({
                scene: scene,
                score: result.similarity,
                cardIndex: idx,
                theme: 'similar',
                scoreLabel: 'match'
            });
        }).join('');

        // Setup event handlers using unified system (cursor tooltip for sidebar)
        setupSceneCardEvents(resultsDiv, { theme: 'similar', tooltipMode: 'cursor' });

        // Update pagination
        if (paginationDiv && totalPages > 1) {
            paginationDiv.style.display = 'flex';
            const pageInfo = paginationDiv.querySelector('.stash-copilot-sidebar-page-info');
            const prevBtn = paginationDiv.querySelector('.stash-copilot-sidebar-page-btn.prev');
            const nextBtn = paginationDiv.querySelector('.stash-copilot-sidebar-page-btn.next');

            if (pageInfo) pageInfo.textContent = `${tabState.currentPage} / ${totalPages}`;
            if (prevBtn) prevBtn.disabled = tabState.currentPage <= 1;
            if (nextBtn) nextBtn.disabled = tabState.currentPage >= totalPages;
        } else if (paginationDiv) {
            paginationDiv.style.display = 'none';
        }
    }

    async function startFrameSearch(sceneId, container) {
        // Guard against concurrent searches
        if (frameSearchState.active) return;

        // Select the main Stash player video specifically (vjs-tech class),
        // not card preview <video> elements which have no src and currentTime=0
        const video = document.querySelector('video.vjs-tech') || document.querySelector('video[src]');
        if (!video) {
            showSidebarError(container, 'No video player found');
            return;
        }
        if (isNaN(video.currentTime) || (!video.currentTime && video.readyState < 1)) {
            showSidebarError(container, 'Play the video first to capture a frame');
            return;
        }

        const timestamp = video.currentTime;
        const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        frameSearchState.active = true;
        frameSearchState.requestId = requestId;
        frameSearchState.queryTimestamp = timestamp;

        // Disable button immediately to prevent double-clicks
        const panel = document.getElementById('scene-copilot-similar-panel');
        if (!panel) return;
        const frameSearchBtn = panel.querySelector('.stash-copilot-frame-search-btn');
        if (frameSearchBtn) frameSearchBtn.disabled = true;

        // Format timestamp for display
        const mins = Math.floor(timestamp / 60);
        const secs = Math.floor(timestamp % 60);
        const timeStr = `${mins}:${secs.toString().padStart(2, '0')}`;

        // Show loading state — replace results area
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');
        const subtabsDiv = panel.querySelector('.stash-copilot-sidebar-subtabs');
        const sliderDiv = panel.querySelector('.stash-copilot-sidebar-slider');
        const filtersDiv = panel.querySelector('.stash-copilot-sidebar-filters');

        // Hide normal similar controls
        if (subtabsDiv) subtabsDiv.style.display = 'none';
        if (sliderDiv) sliderDiv.style.display = 'none';
        if (filtersDiv) filtersDiv.style.display = 'none';
        if (paginationDiv) paginationDiv.style.display = 'none';

        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <button class="stash-copilot-back-to-similar">
                    ← Back to Similar
                </button>
                <div class="stash-copilot-sidebar-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Searching by frame at ${timeStr}...</span>
                </div>
            `;

            // Wire up back button
            const backBtn = resultsDiv.querySelector('.stash-copilot-back-to-similar');
            if (backBtn) {
                backBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    exitFrameSearch(sceneId, panel);
                });
            }
        }

        // Trigger backend task
        try {
            await runPluginTask('Find Similar by Frame', {
                mode: 'find_similar_by_frame',
                scene_id: String(sceneId),
                timestamp: String(timestamp),
                limit: '20',
                request_id: requestId
            });

            pollFrameSearchResults(requestId, sceneId, panel);
        } catch (e) {
            log(`Frame search error: ${e.message}`, 'error');
            showSidebarError(panel, e.message);
            if (frameSearchBtn) frameSearchBtn.disabled = false;
        }
    }

    function pollFrameSearchResults(requestId, sceneId, panel) {
        const resultFile = `/plugin/stash-copilot/assets/frame_search_${requestId}.json`;
        let attempts = 0;
        const maxAttempts = 400; // 60s timeout (first call loads OpenCLIP model ~10-15s)

        frameSearchState.pollInterval = setInterval(async () => {
            attempts++;
            if (attempts > maxAttempts) {
                clearInterval(frameSearchState.pollInterval);
                frameSearchState.pollInterval = null;
                frameSearchState.active = false;
                const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
                if (resultsDiv) {
                    const loading = resultsDiv.querySelector('.stash-copilot-sidebar-loading');
                    if (loading) loading.innerHTML = `
                        <span>Search timed out. The embedding model may still be loading. Try again.</span>
                    `;
                }
                const btn = panel.querySelector('.stash-copilot-frame-search-btn');
                if (btn) btn.disabled = false;
                return;
            }

            try {
                const response = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();

                    // Validate request_id to avoid stale results
                    if (data.request_id !== requestId) return;

                    if (data.status === 'complete' || data.results) {
                        clearInterval(frameSearchState.pollInterval);
                        frameSearchState.pollInterval = null;
                        frameSearchState.results = data.results || [];
                        renderFrameSearchResults(data, panel);
                    } else if (data.status === 'error') {
                        clearInterval(frameSearchState.pollInterval);
                        frameSearchState.pollInterval = null;
                        frameSearchState.active = false;
                        showSidebarError(panel, data.error || 'Frame search failed');
                        const btn = panel.querySelector('.stash-copilot-frame-search-btn');
                        if (btn) btn.disabled = false;
                    }
                }
            } catch (e) {
                log(`Frame search poll error: ${e.message}`);
            }
        }, 150);
        // Timeout is handled by the attempt counter above (200 * 150ms = 30s)
    }

    function renderFrameSearchResults(data, panel) {
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        if (!resultsDiv) return;

        const results = data.results || [];
        const btn = panel.querySelector('.stash-copilot-frame-search-btn');
        if (btn) btn.disabled = false;

        if (results.length === 0) {
            resultsDiv.innerHTML = `
                <button class="stash-copilot-back-to-similar">← Back to Similar</button>
                <div class="stash-copilot-sidebar-empty">No similar frames found</div>
            `;
            wireBackButton(resultsDiv, data.query_scene_id, panel);
            return;
        }

        // Format query timestamp for header
        const qts = data.query_timestamp || 0;
        const qMins = Math.floor(qts / 60);
        const qSecs = Math.floor(qts % 60);
        const qTimeStr = `${qMins}:${qSecs.toString().padStart(2, '0')}`;

        // Build cards using unified card system — show the matched frame as thumbnail
        const cardsHtml = results.map((result, idx) => {
            const scene = result.scene || {};
            const frameThumbnail = result.frame_path
                ? `/plugin/stash-copilot/assets/${result.frame_path}`
                : null;
            return buildSceneCard({
                scene: scene,
                score: result.similarity,
                cardIndex: idx,
                theme: 'frame-search',
                scoreLabel: 'match',
                matchTimestamp: result.matched_timestamp,
                overrideThumbnail: frameThumbnail
            });
        }).join('');

        resultsDiv.innerHTML = `
            <button class="stash-copilot-back-to-similar">← Back to Similar</button>
            <div class="stash-copilot-sidebar-frame-search-header">
                Frame at ${qTimeStr} · ${results.length} match${results.length !== 1 ? 'es' : ''}
            </div>
            ${cardsHtml}
        `;

        wireBackButton(resultsDiv, data.query_scene_id, panel);

        // Setup card events
        setupSceneCardEvents(resultsDiv, { theme: 'frame-search', tooltipMode: 'cursor' });
    }

    function wireBackButton(container, sceneId, panel) {
        const backBtn = container.querySelector('.stash-copilot-back-to-similar');
        if (backBtn) {
            backBtn.addEventListener('click', (e) => {
                e.preventDefault();
                exitFrameSearch(sceneId, panel);
            });
        }
    }

    function exitFrameSearch(sceneId, panel) {
        // Clear any running poll interval
        if (frameSearchState.pollInterval) {
            clearInterval(frameSearchState.pollInterval);
            frameSearchState.pollInterval = null;
        }
        frameSearchState.active = false;
        frameSearchState.results = [];

        // Re-enable button
        const frameSearchBtn = panel.querySelector('.stash-copilot-frame-search-btn');
        if (frameSearchBtn) frameSearchBtn.disabled = false;

        // Re-show normal similar controls
        const subtabsDiv = panel.querySelector('.stash-copilot-sidebar-subtabs');
        const sliderDiv = panel.querySelector('.stash-copilot-sidebar-slider');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');
        const filtersDiv = panel.querySelector('.stash-copilot-sidebar-filters');

        if (subtabsDiv) subtabsDiv.style.display = '';
        if (sliderDiv) sliderDiv.style.display = '';
        if (filtersDiv) filtersDiv.style.display = '';
        // paginationDiv visibility is managed by renderSidebarSimilarResultsUI, just reset to allow it
        if (paginationDiv) paginationDiv.style.display = '';

        // Re-render cached similar results if available
        const tabState = similarState.tabs[similarState.activeTab];
        if (tabState && tabState.loaded) {
            renderSidebarSimilarResultsUI(tabState, panel);
        } else {
            // Re-trigger search if no cached results
            startSidebarSimilarSearch(sceneId);
        }
    }

    /**
     * Wrapper function for pagination - finds the panel and calls the UI renderer
     */
    function renderSidebarSimilarResults(container) {
        // Find the panel element (container may be the panel itself or a child)
        const panel = container.closest('.stash-copilot-sidebar-similar') ||
                      document.getElementById('scene-copilot-similar-panel')?.querySelector('.stash-copilot-sidebar-similar');
        if (!panel) {
            log('renderSidebarSimilarResults: panel not found');
            return;
        }

        const tabState = similarState.tabs[similarState.activeTab];
        if (!tabState) {
            log('renderSidebarSimilarResults: tab state not found');
            return;
        }

        renderSidebarSimilarResultsUI(tabState, panel);
    }

    // ===== Sidebar Recommendations Functions =====

    async function startSidebarRecsSearch(sceneId, refresh = false) {
        log(`Starting sidebar recs search for scene ${sceneId} (mode: ${sceneRecsState.mode})`);
        sceneRecsState.sceneId = sceneId;

        const panel = document.getElementById('scene-copilot-recs-panel');
        if (!panel) return;

        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');

        // Show loading - recreate if it was removed by previous render
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="stash-copilot-sidebar-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Getting recommendations...</span>
                </div>
            `;
        }
        if (paginationDiv) paginationDiv.style.display = 'none';

        // Hide profile info during refresh
        const profileInfo = panel.querySelector('.stash-copilot-sidebar-profile-info');
        if (profileInfo) profileInfo.style.display = 'none';

        // Reset state if refresh
        if (refresh) {
            sceneRecsState.results = [];
            sceneRecsState.allResults = [];
            sceneRecsState.currentPage = 1;
        }

        // Generate a single request ID
        const requestId = `rec_${Date.now()}`;
        sceneRecsState.requestId = requestId;

        // Build task args
        const useTimeDecay = sceneRecsState.timeDecayDays > 0;
        const taskArgs = {
            request_id: requestId,
            scoring_method: useTimeDecay ? 'time_decayed' : 'base_weighted',
            half_life_days: String(sceneRecsState.timeDecayDays || 30),
            seed_scene_id: String(sceneId),
            limit: '120',
        };

        // Select task and add mode-specific args
        const taskName = sceneRecsState.mode === 'rewatch'
            ? 'Get Recommendations (Re-watch)'
            : 'Get Recommendations (Discover)';

        if (sceneRecsState.mode === 'rewatch') {
            taskArgs.engagement_weight = String(sceneRecsState.engagementWeight);
        } else {
            taskArgs.seed_weight = String(sceneRecsState.seedWeight);
        }

        try {
            const weightInfo = sceneRecsState.mode === 'rewatch'
                ? `engagement_weight ${sceneRecsState.engagementWeight}`
                : `seed_weight ${sceneRecsState.seedWeight}`;
            log(`Firing ${taskName} task (${requestId}, ${weightInfo})`);

            await runPluginTask(taskName, taskArgs);

            // Poll for single result file
            pollSidebarRecsResults(panel, requestId);

        } catch (e) {
            log(`Recs search error: ${e.message}`, 'error');
            showSidebarError(panel, e.message);
        }
    }

    /**
     * Poll for a single sidebar recommendation result file.
     */
    function pollSidebarRecsResults(panel, requestId) {
        const resultFile = `/plugin/stash-copilot/assets/recommendations_${requestId}.json`;

        const pollInterval = setInterval(async () => {
            // Bail if request ID has changed (new search started)
            if (sceneRecsState.requestId !== requestId) {
                clearInterval(pollInterval);
                return;
            }

            try {
                const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        clearInterval(pollInterval);
                        sceneRecsState.allResults = data.results || [];
                        sceneRecsState.results = [...sceneRecsState.allResults];
                        sceneRecsState.profile = data.profile;
                        log(`Sidebar recs results received: ${sceneRecsState.results.length} results`);
                        renderSidebarRecsResultsUI(panel);
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        const errorMsg = data.error || 'Recommendation task failed';
                        log(`Sidebar recs task error: ${errorMsg}`, 'error');
                        showSidebarError(panel, errorMsg);
                    }
                }
            } catch (e) {
                // File not ready yet, continue polling
            }
        }, 500);

        // Timeout after 120s
        setTimeout(() => {
            clearInterval(pollInterval);
            if (sceneRecsState.results.length === 0) {
                showSidebarError(panel, 'Request timed out. Please try again.');
            }
        }, 120000);
    }

    function renderSidebarRecsResultsUI(panel) {
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');

        if (!resultsDiv) return;

        // Hide loading
        const loadingDiv = resultsDiv.querySelector('.stash-copilot-sidebar-loading');
        if (loadingDiv) loadingDiv.style.display = 'none';

        // Apply filters
        let filteredResults = sceneRecsState.allResults;

        if (sceneRecsState.excludePerformers.length > 0 || sceneRecsState.excludeTags.length > 0) {
            filteredResults = filteredResults.filter(result => {
                const scene = result.scene;
                if (!scene) return true;

                const performerNames = scene.performers?.map(p => p.name) || [];
                const tagNames = scene.tags?.map(t => t.name) || [];

                const excludeByPerformer = sceneRecsState.excludePerformers.some(p => performerNames.includes(p));
                const excludeByTag = sceneRecsState.excludeTags.some(t => tagNames.includes(t));

                return !excludeByPerformer && !excludeByTag;
            });
        }

        sceneRecsState.results = filteredResults;

        if (filteredResults.length === 0) {
            resultsDiv.innerHTML = `
                <div class="stash-copilot-sidebar-empty">
                    No recommendations found
                </div>
            `;
            if (paginationDiv) paginationDiv.style.display = 'none';
            return;
        }

        // Paginate
        const start = (sceneRecsState.currentPage - 1) * sceneRecsState.resultsPerPage;
        const end = start + sceneRecsState.resultsPerPage;
        const pageResults = filteredResults.slice(start, end);
        const totalPages = Math.ceil(filteredResults.length / sceneRecsState.resultsPerPage);

        // Render cards using unified card system
        const cardTheme = 'recs';
        resultsDiv.innerHTML = pageResults.map((result, idx) => {
            const scene = result.scene;
            const score = result.combined_score || result.similarity_score || 0;
            return buildSceneCard({
                scene: scene,
                score: score,
                cardIndex: idx,
                theme: cardTheme,
                scoreLabel: 'match'
            });
        }).join('');

        // Setup event handlers using unified system (cursor tooltip for sidebar)
        setupSceneCardEvents(resultsDiv, { theme: cardTheme, tooltipMode: 'cursor' });

        // Update pagination
        if (paginationDiv && totalPages > 1) {
            paginationDiv.style.display = 'flex';
            const pageInfo = paginationDiv.querySelector('.stash-copilot-sidebar-page-info');
            const prevBtn = paginationDiv.querySelector('.stash-copilot-sidebar-page-btn.prev');
            const nextBtn = paginationDiv.querySelector('.stash-copilot-sidebar-page-btn.next');

            if (pageInfo) pageInfo.textContent = `${sceneRecsState.currentPage} / ${totalPages}`;
            if (prevBtn) prevBtn.disabled = sceneRecsState.currentPage <= 1;
            if (nextBtn) nextBtn.disabled = sceneRecsState.currentPage >= totalPages;
        } else if (paginationDiv) {
            paginationDiv.style.display = 'none';
        }

        // Render profile info section
        renderSidebarRecsProfileInfo(panel);
    }

    /**
     * Render profile info section showing what contributed to recommendations
     */
    function renderSidebarRecsProfileInfo(panel) {
        const profileSection = panel.querySelector('.stash-copilot-sidebar-profile-info');
        if (!profileSection || !sceneRecsState.profile) {
            if (profileSection) profileSection.style.display = 'none';
            return;
        }

        const profile = sceneRecsState.profile;
        const sceneCount = profile.scene_count || profile.contributing_scenes?.length || 0;

        // Don't show if no scenes contributed
        if (sceneCount === 0) {
            profileSection.style.display = 'none';
            return;
        }

        const totalResults = sceneRecsState.results?.length || 0;

        profileSection.innerHTML = `
            <div class="stash-copilot-profile-summary">
                <span class="stash-copilot-profile-icon">📊</span>
                <span>${totalResults} recommendations from ${sceneCount} profile scenes</span>
            </div>
        `;

        profileSection.style.display = 'block';
    }

    /**
     * Wrapper function for recs pagination - finds the panel and calls the UI renderer
     */
    function renderSidebarRecsResults(container) {
        // Find the panel element (container may be the panel itself or a child)
        const panel = container.closest('.stash-copilot-sidebar-recs') ||
                      document.getElementById('scene-copilot-recs-panel')?.querySelector('.stash-copilot-sidebar-recs');
        if (!panel) {
            log('renderSidebarRecsResults: panel not found');
            return;
        }

        renderSidebarRecsResultsUI(panel);
    }

    function showSidebarError(panel, message) {
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        if (resultsDiv) {
            const loadingDiv = resultsDiv.querySelector('.stash-copilot-sidebar-loading');
            if (loadingDiv) loadingDiv.style.display = 'none';

            resultsDiv.innerHTML = `
                <div class="stash-copilot-sidebar-error">
                    ${escapeHtml(message)}
                </div>
            `;
        }
    }

    function formatTimestamp(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // ===== End Scene Sidebar Tabs =====

    // ===== Session Scene Tracking =====
    // Uses localStorage with timestamp-based expiry for cross-tab sharing
    // Session expires after 4 hours of inactivity

    const SESSION_STORAGE_KEY = 'stash-copilot-session-scenes';
    const SESSION_EXPIRY_HOURS = 4;

    // Track a scene visit in shared session storage
    function trackSessionScene(sceneId) {
        if (!sceneId) return;

        try {
            const sceneIdNum = parseInt(sceneId, 10);
            if (isNaN(sceneIdNum)) return;

            const now = Date.now();
            const data = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || '{"scenes":[],"lastActivity":0}');

            // Check if session has expired (4 hours of inactivity)
            const expiryMs = SESSION_EXPIRY_HOURS * 60 * 60 * 1000;
            if (data.lastActivity && (now - data.lastActivity) > expiryMs) {
                // Session expired, start fresh
                data.scenes = [];
                log('Session expired, starting new session');
            }

            // Add to beginning if not already present (most recent first)
            if (!data.scenes.includes(sceneIdNum)) {
                data.scenes.unshift(sceneIdNum);
                // Keep only last 100 scenes per session
                if (data.scenes.length > 100) {
                    data.scenes.pop();
                }
            }

            // Update last activity timestamp
            data.lastActivity = now;
            localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data));
            log(`Tracked session scene: ${sceneIdNum} (${data.scenes.length} total)`);
        } catch (e) {
            log(`Failed to track session scene: ${e.message}`, 'warn');
        }
    }

    // Get scenes viewed this session (shared across tabs)
    function getSessionScenes() {
        try {
            const now = Date.now();
            const data = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || '{"scenes":[],"lastActivity":0}');

            // Check if session has expired
            const expiryMs = SESSION_EXPIRY_HOURS * 60 * 60 * 1000;
            if (data.lastActivity && (now - data.lastActivity) > expiryMs) {
                // Session expired, return empty
                return [];
            }

            return data.scenes || [];
        } catch (e) {
            return [];
        }
    }

    // Clear the current session
    function clearSessionScenes() {
        try {
            localStorage.removeItem(SESSION_STORAGE_KEY);
            log('Session scenes cleared');
        } catch (e) {
            log(`Failed to clear session scenes: ${e.message}`, 'warn');
        }
    }

    // ===== End Session Scene Tracking =====

    // Add custom UI to scene page
    function enhanceScenePage() {
        log('Enhancing scene page...');

        const sceneId = window.location.pathname.split('/').pop();

        // Track this scene visit for session-based recommendations
        trackSessionScene(sceneId);

        // Inject AI tabs into the native Stash sidebar
        injectSceneTabs(sceneId)
            .then(async () => {
                log('Scene page enhanced with sidebar tabs');

                // Check for auto-analyze
                const autoAnalyze = await shouldAutoAnalyze();
                if (autoAnalyze) {
                    log('Auto-analyze enabled, opening analyze tab...');
                    // Trigger the analyze tab to open
                    const analyzeTab = document.querySelector('a[data-rb-event-key="scene-copilot-analyze"]');
                    if (analyzeTab) {
                        analyzeTab.click();
                    }
                }
            })
            .catch(error => {
                log(`Failed to enhance scene page: ${error.message}`, 'error');
            });
    }

    // Add custom UI to scenes list
    function enhanceScenesList() {
        log('Enhancing scenes list...');
        // Native Stash sort "Average Stroke Range" (sortby=interactive_stroke_range)
        // handles this feature server-side. No plugin-side sort needed.
    }

    // =========================================================================
    // Performer Page UI Integration
    // =========================================================================

    // State for performer page
    const performerTabState = {
        performerId: null,
        initialized: false,
        activeTab: null,
        contentLoaded: { similar: false, description: false }
    };

    // Enhance performer page with AI tabs
    function enhancePerformerPage() {
        log('Enhancing performer page...');

        // Extract performer ID from /performers/{id} or /performers/{id}/scenes etc.
        const performerId = window.location.pathname.split('/')[2];

        injectPerformerTabs(performerId)
            .then(() => {
                log('Performer page enhanced with AI tabs');
            })
            .catch(error => {
                log(`Failed to enhance performer page: ${error.message}`, 'error');
            });
    }

    // Inject AI tabs into performer page sidebar
    async function injectPerformerTabs(performerId) {
        // Wait for the tab bar to be ready
        const tabNav = await waitForElement('.nav.nav-tabs');
        if (!tabNav) {
            log('Performer tab nav not found', 'warn');
            return;
        }

        // Check if already injected
        if (document.querySelector('.stash-copilot-performer-tab-nav')) {
            log('Performer tabs already injected');
            // Update state if performer changed
            if (performerTabState.performerId !== performerId) {
                performerTabState.performerId = performerId;
                performerTabState.contentLoaded = { similar: false, description: false };
            }
            return;
        }

        performerTabState.performerId = performerId;
        performerTabState.initialized = false;
        performerTabState.activeTab = null;
        performerTabState.contentLoaded = { similar: false, description: false };

        // Find tab content container
        const tabContent = document.querySelector('.tab-content');
        if (!tabContent) {
            log('Tab content container not found', 'warn');
            return;
        }

        // Create "Similar" tab nav item
        const similarTabNav = document.createElement('li');
        similarTabNav.className = 'nav-item stash-copilot-performer-tab-nav';
        similarTabNav.innerHTML = `
            <a class="nav-link" data-rb-event-key="performer-copilot-similar" role="tab">
                <span style="background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                    ✨ Similar
                </span>
            </a>
        `;

        // Create "AI Profile" tab nav item
        const profileTabNav = document.createElement('li');
        profileTabNav.className = 'nav-item stash-copilot-performer-tab-nav';
        profileTabNav.innerHTML = `
            <a class="nav-link" data-rb-event-key="performer-copilot-profile" role="tab">
                <span style="background: linear-gradient(135deg, #8b5cf6, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                    🤖 AI Profile
                </span>
            </a>
        `;

        // Add tab nav items
        tabNav.appendChild(similarTabNav);
        tabNav.appendChild(profileTabNav);

        // Create tab panes
        const similarPane = document.createElement('div');
        similarPane.id = 'performer-copilot-similar-panel';
        similarPane.className = 'tab-pane stash-copilot-performer-tab-pane';
        similarPane.role = 'tabpanel';

        const profilePane = document.createElement('div');
        profilePane.id = 'performer-copilot-profile-panel';
        profilePane.className = 'tab-pane stash-copilot-performer-tab-pane';
        profilePane.role = 'tabpanel';

        tabContent.appendChild(similarPane);
        tabContent.appendChild(profilePane);

        // Add click handlers
        similarTabNav.querySelector('a').addEventListener('click', (e) => {
            e.preventDefault();
            handlePerformerTabClick('similar', performerId);
        });

        profileTabNav.querySelector('a').addEventListener('click', (e) => {
            e.preventDefault();
            handlePerformerTabClick('profile', performerId);
        });

        performerTabState.initialized = true;
        log('Performer AI tabs injected');
    }

    // Handle performer tab click
    function handlePerformerTabClick(tabKey, performerId) {
        log(`Performer tab clicked: ${tabKey}`);

        // Remove active class from all tabs
        document.querySelectorAll('.stash-copilot-performer-tab-nav a').forEach(a => {
            a.classList.remove('active');
        });

        // Hide all custom tab panes
        document.querySelectorAll('.stash-copilot-performer-tab-pane').forEach(pane => {
            pane.classList.remove('active', 'show');
        });

        // Also deactivate Stash's native tabs
        document.querySelectorAll('.nav-tabs .nav-link:not([data-rb-event-key^="performer-copilot"])').forEach(a => {
            a.classList.remove('active');
        });
        document.querySelectorAll('.tab-pane:not(.stash-copilot-performer-tab-pane)').forEach(pane => {
            pane.classList.remove('active', 'show');
        });

        // Activate clicked tab
        const tabNav = document.querySelector(`a[data-rb-event-key="performer-copilot-${tabKey}"]`);
        if (tabNav) {
            tabNav.classList.add('active');
        }

        // Show corresponding pane
        const pane = document.getElementById(`performer-copilot-${tabKey}-panel`);
        if (pane) {
            pane.classList.add('active', 'show');
        }

        performerTabState.activeTab = tabKey;

        // Load content if not already loaded
        if (!performerTabState.contentLoaded[tabKey]) {
            loadPerformerTabContent(tabKey, performerId);
        }
    }

    // Load performer tab content
    async function loadPerformerTabContent(tabKey, performerId) {
        const pane = document.getElementById(`performer-copilot-${tabKey}-panel`);
        if (!pane) return;

        if (tabKey === 'similar') {
            await renderSimilarPerformersContent(pane, performerId);
        } else if (tabKey === 'profile') {
            await renderPerformerProfileContent(pane, performerId);
        }

        performerTabState.contentLoaded[tabKey] = true;
    }

    // Poll for results file from backend task
    async function pollForResults(filename, timeoutMs = 60000) {
        const startTime = Date.now();
        const pollInterval = 200; // Poll every 200ms
        const resultFile = `/plugin/stash-copilot/assets/${filename}.json`;

        while (Date.now() - startTime < timeoutMs) {
            try {
                const cacheBuster = `?t=${Date.now()}`;
                const response = await fetch(resultFile + cacheBuster, { cache: 'no-store' });

                if (response.ok) {
                    const data = await response.json();

                    // Check if result is complete or has an error
                    if (data.status === 'complete' || data.status === 'error' || data.results) {
                        return data;
                    }
                }
            } catch (e) {
                // File not ready yet, continue polling
            }

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }

        // Timeout
        throw new Error('Request timed out');
    }

    // Render similar performers content
    async function renderSimilarPerformersContent(container, performerId) {
        container.innerHTML = `
            <div class="stash-copilot-performer-loading">
                <div class="loading-spinner"></div>
                <p>Finding similar performers...</p>
            </div>
        `;

        try {
            // Generate a unique request ID
            const requestId = `${performerId}_${Date.now()}`;

            // Trigger the find similar performers task
            await runPluginTask('Find Similar Performers', {
                performer_id: performerId,
                limit: '20',
                request_id: requestId
            });

            // Poll for results
            const results = await pollForResults(`similar_performers_${requestId}`, 60000);

            if (results.status === 'error') {
                throw new Error(results.error || 'Failed to find similar performers');
            }

            if (!results.results || results.results.length === 0) {
                container.innerHTML = `
                    <div class="stash-copilot-performer-empty">
                        <p>No similar performers found.</p>
                        <p class="text-muted">Make sure to run "Embed All Performers" task first.</p>
                    </div>
                `;
                return;
            }

            // Render results
            const html = `
                <div class="stash-copilot-similar-performers">
                    <div class="similar-performers-header">
                        <h4>Similar Performers</h4>
                        <span class="badge">${results.results.length} found</span>
                    </div>
                    <div class="similar-performers-grid">
                        ${results.results.map(p => buildPerformerCard(p)).join('')}
                    </div>
                </div>
            `;

            container.innerHTML = html;
            setupPerformerCardEvents(container);

        } catch (error) {
            log(`Error loading similar performers: ${error.message}`, 'error');
            container.innerHTML = `
                <div class="stash-copilot-performer-error">
                    <p>Failed to find similar performers: ${error.message}</p>
                    <button class="btn btn-secondary" onclick="location.reload()">Retry</button>
                </div>
            `;
        }
    }

    // Render performer AI profile content
    async function renderPerformerProfileContent(container, performerId) {
        container.innerHTML = `
            <div class="stash-copilot-performer-loading">
                <div class="loading-spinner"></div>
                <p>Loading AI profile...</p>
            </div>
        `;

        try {
            // Check if performer has embedding and description
            const results = await pollForResults(`similar_performers_${performerId}`, 5000).catch(() => null);

            // For now, show a placeholder with instructions
            // Full profile will include: AI-generated description, top tags, stats
            container.innerHTML = `
                <div class="stash-copilot-performer-profile">
                    <div class="profile-section">
                        <h4>🤖 AI-Generated Profile</h4>
                        <p class="text-muted">AI-generated performer descriptions are coming soon!</p>
                        <p class="text-muted">To generate a description, run the "Describe Performer" task from the Stash Task Manager.</p>
                    </div>
                    <div class="profile-actions">
                        <button class="btn btn-primary" onclick="stashCopilot.runDescribePerformer(${performerId})">
                            Generate AI Description
                        </button>
                    </div>
                </div>
            `;

        } catch (error) {
            log(`Error loading performer profile: ${error.message}`, 'error');
            container.innerHTML = `
                <div class="stash-copilot-performer-error">
                    <p>Failed to load profile: ${error.message}</p>
                </div>
            `;
        }
    }

    // Build performer card HTML
    function buildPerformerCard(performer) {
        const similarity = Math.round((performer.similarity || 0) * 100);
        const sceneCount = performer.scene_count || 0;
        const name = performer.name || 'Unknown';
        const gender = performer.gender || '';
        const country = performer.country || '';

        return `
            <div class="stash-copilot-performer-card" data-performer-id="${performer.performer_id}">
                <div class="performer-card-image">
                    <a href="/performers/${performer.performer_id}">
                        <img src="/performer/${performer.performer_id}/image" alt="${name}" onerror="this.src='/assets/default_performer.png'"/>
                    </a>
                </div>
                <div class="performer-card-info">
                    <a href="/performers/${performer.performer_id}" class="performer-name">${name}</a>
                    <div class="performer-details">
                        ${gender ? `<span class="gender">${gender}</span>` : ''}
                        ${country ? `<span class="country">${country}</span>` : ''}
                    </div>
                    <div class="performer-stats">
                        <span class="scene-count">${sceneCount} scenes</span>
                        <span class="similarity-badge" style="background: linear-gradient(135deg, #10b981, #059669);">
                            ${similarity}% match
                        </span>
                    </div>
                </div>
            </div>
        `;
    }

    // Setup performer card event handlers
    function setupPerformerCardEvents(container) {
        // Hover effects are handled by CSS
        // Click navigation is handled by the anchor tags
    }

    // Run describe performer task
    async function runDescribePerformer(performerId) {
        try {
            log(`Starting describe performer task for ${performerId}`);
            await runPluginTask('Describe Performer', {
                performer_id: performerId.toString(),
                force: 'true'
            });
            alert('Performer description task started. Check the Stash Task Manager for progress.');
        } catch (error) {
            log(`Error running describe performer: ${error.message}`, 'error');
            alert(`Failed to start task: ${error.message}`);
        }
    }

    // Path-based page detection and enhancement
    function onPageChange() {
        const path = window.location.pathname;

        log(`Page changed: ${path}`);

        if (path.startsWith('/scenes/') && path.split('/').length === 3) {
            enhanceScenePage();
        } else if (path === '/scenes') {
            enhanceScenesList();
        } else if (/^\/performers\/\d+(\/|$)/.test(path)) {
            // Match /performers/{id} and /performers/{id}/scenes, /performers/{id}/images, etc.
            enhancePerformerPage();
        } else if (path === '/plugins/stash-copilot/search') {
            // Render semantic search page
            setTimeout(renderSearchPage, 100);
        } else if (path === '/plugins/stash-copilot/tag-dedup') {
            // Render tag dedup page
            setTimeout(renderTagDedupPage, 100);
        }
        // Navbar dropdown handles stats display globally now
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tag Deduplication UI
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Create nav button for tag dedup page
     */
    function createDedupNavButton() {
        if (document.getElementById('stash-copilot-dedup-nav-item')) return;

        const labelingNavItem = document.getElementById('stash-copilot-labeling-nav-item');
        const searchNavItem = document.getElementById('stash-copilot-search-nav-item');
        const insertAfter = labelingNavItem || searchNavItem;

        if (!insertAfter) {
            const tagsLink = document.querySelector('.navbar-nav a[href="/tags"]');
            if (!tagsLink) return;
            var insertPoint = tagsLink.closest('.nav-item') || tagsLink.parentElement;
        } else {
            var insertPoint = insertAfter;
        }

        const navItem = document.createElement('li');
        navItem.className = 'nav-item';
        navItem.id = 'stash-copilot-dedup-nav-item';

        const btn = document.createElement('a');
        btn.id = 'stash-copilot-dedup-nav-btn';
        btn.className = 'nav-link';
        btn.href = '/plugins/stash-copilot/tag-dedup';
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                <circle cx="9" cy="7" r="4"/>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
            <span>Tag Dedup</span>
        `;
        btn.title = 'Find and merge duplicate tags';

        navItem.appendChild(btn);
        insertPoint.parentNode.insertBefore(navItem, insertPoint.nextSibling);

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToDedupPage();
        });

        log('Tag Dedup nav button created');
    }

    function navigateToDedupPage() {
        history.pushState({ stashCopilotDedup: true }, '', '/plugins/stash-copilot/tag-dedup');
        renderTagDedupPage();
    }

    /**
     * Handle browser back/forward for dedup page
     */
    function setupDedupNavigationHandler() {
        window.addEventListener('popstate', () => {
            const path = window.location.pathname;
            if (path === '/plugins/stash-copilot/tag-dedup') {
                renderTagDedupPage();
            } else if (document.querySelector('.stash-copilot-dedup-page')) {
                window.location.reload();
            }
        });
    }

    /**
     * Render the tag dedup page (full SPA page replacement)
     */
    function renderTagDedupPage() {
        // Guard against double-render (click handler + onPageChange both fire)
        if (document.querySelector('.stash-copilot-dedup-page')) {
            log('Tag dedup page already rendered, skipping');
            return;
        }
        log('Rendering tag dedup page...');

        let mainContent = document.querySelector('.main');
        if (!mainContent) mainContent = document.querySelector('#root > div:last-child');
        if (!mainContent) mainContent = document.querySelector('.container-fluid') || document.querySelector('#root');
        if (!mainContent) {
            log('Could not find main content area', 'error');
            return;
        }

        // Reset state
        state.tagDedupCandidates = [];
        state.tagDedupCurrentIndex = 0;
        state.tagDedupMergeCount = 0;
        state.tagDedupSkipCount = 0;
        state.tagDedupScenesUpdated = 0;
        state.tagDedupProcessing = false;

        mainContent.innerHTML = `
            <div class="stash-copilot-dedup-page">
                <div class="stash-copilot-dedup-header">
                    <div class="stash-copilot-dedup-header-left">
                        <a href="/tags" class="stash-copilot-dedup-back-btn" onclick="event.preventDefault(); window.location.href='/tags';">← Back</a>
                        <h1 class="stash-copilot-dedup-title">
                            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                                <circle cx="9" cy="7" r="4"/>
                                <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                                <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                            </svg>
                            Tag Deduplication
                        </h1>
                    </div>
                </div>
                <div class="stash-copilot-dedup-body">
                    <div class="stash-copilot-dedup-loading">
                        <div class="stash-copilot-spinner"></div>
                        <span>Scanning for duplicate tags...</span>
                    </div>
                </div>
            </div>
        `;

        // Add keyboard listener
        document.removeEventListener('keydown', handleDedupKeyboard);
        document.addEventListener('keydown', handleDedupKeyboard);

        // Start the scan
        startTagDedupScan();
    }

    async function startTagDedupScan() {
        const requestId = `dedup_${Date.now()}`;
        state.tagDedupRequestId = requestId;

        try {
            await runPluginTask('Find Duplicate Tags', {
                mode: 'find_duplicate_tags',
                request_id: requestId,
            });
        } catch (e) {
            log(`Failed to start tag dedup scan: ${e.message}`, 'error');
            renderDedupError('Failed to start scan. Check plugin logs.');
            return;
        }

        // Poll for results
        const resultFile = `/plugin/stash-copilot/assets/tag_dedup_${requestId}.json`;
        let attempts = 0;
        state.tagDedupPollInterval = setInterval(async () => {
            attempts++;
            if (state.tagDedupRequestId !== requestId) {
                clearInterval(state.tagDedupPollInterval);
                return;
            }

            try {
                const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    clearInterval(state.tagDedupPollInterval);
                    state.tagDedupPollInterval = null;

                    if (data.status === 'complete' && data.candidates && data.candidates.length > 0) {
                        state.tagDedupCandidates = data.candidates;
                        renderDedupPair();
                    } else if (data.status === 'complete') {
                        renderDedupEmpty('No duplicate tags found above 75% similarity.');
                    } else if (data.status === 'no_embeddings') {
                        renderDedupEmpty('No tag embeddings found. Run "Build Tag Vocabulary" first.');
                    } else {
                        renderDedupError(data.error || 'Unknown error');
                    }
                }
            } catch (e) { /* file not ready yet */ }

            if (attempts > 150) { // 30s timeout
                clearInterval(state.tagDedupPollInterval);
                renderDedupError('Scan timed out. Check plugin logs.');
            }
        }, 200);
    }

    function renderDedupPair() {
        const body = document.querySelector('.stash-copilot-dedup-body');
        if (!body) return;

        const candidates = state.tagDedupCandidates;
        const idx = state.tagDedupCurrentIndex;

        if (idx >= candidates.length) {
            renderDedupSummary();
            return;
        }

        const candidate = candidates[idx];
        const total = candidates.length;
        const similarityPct = Math.round(candidate.similarity * 100);
        const progressPct = Math.round((idx / total) * 100);

        body.innerHTML = `
            <div class="stash-copilot-dedup-pair-enter">
                <div class="stash-copilot-dedup-pair-info">
                    Pair ${idx + 1} of ${total}
                    <span class="stash-copilot-dedup-similarity-badge">${similarityPct}% similar</span>
                </div>

                <div class="stash-copilot-dedup-versus">
                    <div class="stash-copilot-dedup-card ${candidate.suggested_keep === 'a' ? 'suggested' : ''}" data-side="a">
                        <div class="stash-copilot-dedup-tag-name">${escapeHtml(candidate.tag_a.name)}</div>
                        <div class="stash-copilot-dedup-scene-count"><strong>${candidate.tag_a.scene_count}</strong> scenes</div>
                        ${candidate.suggested_keep === 'a' ? '<div class="stash-copilot-dedup-keep-badge">suggested keep</div>' : ''}
                    </div>

                    <div class="stash-copilot-dedup-vs">VS</div>

                    <div class="stash-copilot-dedup-card ${candidate.suggested_keep === 'b' ? 'suggested' : ''}" data-side="b">
                        <div class="stash-copilot-dedup-tag-name">${escapeHtml(candidate.tag_b.name)}</div>
                        <div class="stash-copilot-dedup-scene-count"><strong>${candidate.tag_b.scene_count}</strong> scenes</div>
                        ${candidate.suggested_keep === 'b' ? '<div class="stash-copilot-dedup-keep-badge">suggested keep</div>' : ''}
                    </div>
                </div>

                <div class="stash-copilot-dedup-actions">
                    <button class="stash-copilot-dedup-btn keep-left" id="dedup-keep-left">
                        ← Keep Left
                    </button>
                    <button class="stash-copilot-dedup-btn skip-btn" id="dedup-skip">
                        Skip
                    </button>
                    <button class="stash-copilot-dedup-btn keep-right" id="dedup-keep-right">
                        Keep Right →
                    </button>
                </div>

                <div class="stash-copilot-dedup-keyboard-hint">
                    <kbd>←</kbd> Keep Left
                    <kbd>↓</kbd> Skip
                    <kbd>→</kbd> Keep Right
                </div>

                <div class="stash-copilot-dedup-merge-status" id="dedup-merge-status" style="display:none"></div>

                <div class="stash-copilot-dedup-progress">
                    <div class="stash-copilot-dedup-progress-bar">
                        <div class="stash-copilot-dedup-progress-fill" style="width: ${progressPct}%"></div>
                    </div>
                    <div class="stash-copilot-dedup-progress-text">
                        ${state.tagDedupMergeCount} merged · ${state.tagDedupSkipCount} skipped
                    </div>
                </div>
            </div>
        `;

        // Attach click handlers to cards and buttons
        body.querySelector('.stash-copilot-dedup-card[data-side="a"]').addEventListener('click', () => handleDedupKeep('a'));
        body.querySelector('.stash-copilot-dedup-card[data-side="b"]').addEventListener('click', () => handleDedupKeep('b'));
        body.querySelector('#dedup-keep-left').addEventListener('click', () => handleDedupKeep('a'));
        body.querySelector('#dedup-keep-right').addEventListener('click', () => handleDedupKeep('b'));
        body.querySelector('#dedup-skip').addEventListener('click', () => handleDedupSkip());
    }

    async function handleDedupKeep(side) {
        if (state.tagDedupProcessing) return;
        state.tagDedupProcessing = true;

        const candidate = state.tagDedupCandidates[state.tagDedupCurrentIndex];
        if (!candidate) { state.tagDedupProcessing = false; return; }

        const keepTag = side === 'a' ? candidate.tag_a : candidate.tag_b;
        const removeTag = side === 'a' ? candidate.tag_b : candidate.tag_a;
        const removeSide = side === 'a' ? 'b' : 'a';
        const keepSide = side;

        // Animate cards
        const removeCard = document.querySelector(`.stash-copilot-dedup-card[data-side="${removeSide}"]`);
        const keepCard = document.querySelector(`.stash-copilot-dedup-card[data-side="${keepSide}"]`);
        if (removeCard) removeCard.classList.add('removing');
        if (keepCard) keepCard.classList.add('keeping');

        // Disable buttons and show merge status
        document.querySelectorAll('.stash-copilot-dedup-btn').forEach(b => b.disabled = true);
        const statusEl = document.getElementById('dedup-merge-status');
        if (statusEl) {
            statusEl.style.display = 'flex';
            statusEl.className = 'stash-copilot-dedup-merge-status merging';
            statusEl.innerHTML = `<div class="status-spinner"></div> Merging "${escapeHtml(removeTag.name)}" into "${escapeHtml(keepTag.name)}"…`;
        }

        try {
            const requestId = `merge_${Date.now()}`;
            await runPluginTask('Merge Tags', {
                mode: 'merge_tags',
                keep_tag_id: String(keepTag.id),
                remove_tag_id: String(removeTag.id),
                request_id: requestId,
            });

            // Poll for merge result
            const resultFile = `/plugin/stash-copilot/assets/tag_merge_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);

                        if (data.status === 'complete') {
                            state.tagDedupMergeCount++;
                            state.tagDedupScenesUpdated += data.scenes_updated || 0;
                            removeMergedTagFromCandidates(removeTag.id);

                            // Show success status briefly
                            if (statusEl) {
                                const scenesMsg = data.scenes_updated ? `${data.scenes_updated} scene${data.scenes_updated !== 1 ? 's' : ''} updated` : 'No scenes to update';
                                statusEl.className = 'stash-copilot-dedup-merge-status success';
                                statusEl.innerHTML = `✓ Merged! ${scenesMsg}`;
                            }
                            await new Promise(r => setTimeout(r, 600));
                        } else {
                            log(`Merge failed: ${data.error}`, 'error');
                            if (statusEl) {
                                statusEl.className = 'stash-copilot-dedup-merge-status error';
                                statusEl.innerHTML = `✗ ${data.error || 'Merge failed'}`;
                            }
                            await new Promise(r => setTimeout(r, 1500));
                        }

                        state.tagDedupCurrentIndex++;
                        state.tagDedupProcessing = false;
                        renderDedupPair();
                    }
                } catch (e) { /* not ready */ }

                if (attempts > 100) {
                    clearInterval(poll);
                    log('Merge timed out', 'error');
                    if (statusEl) {
                        statusEl.className = 'stash-copilot-dedup-merge-status error';
                        statusEl.innerHTML = '✗ Merge timed out';
                    }
                    await new Promise(r => setTimeout(r, 1500));
                    state.tagDedupCurrentIndex++;
                    state.tagDedupProcessing = false;
                    renderDedupPair();
                }
            }, 200);
        } catch (e) {
            log(`Merge error: ${e.message}`, 'error');
            if (statusEl) {
                statusEl.className = 'stash-copilot-dedup-merge-status error';
                statusEl.innerHTML = `✗ ${e.message}`;
            }
            state.tagDedupProcessing = false;
            document.querySelectorAll('.stash-copilot-dedup-btn').forEach(b => b.disabled = false);
        }
    }

    function removeMergedTagFromCandidates(removedTagId) {
        const remaining = state.tagDedupCandidates.slice(state.tagDedupCurrentIndex + 1);
        const filtered = remaining.filter(
            c => c.tag_a.id !== removedTagId && c.tag_b.id !== removedTagId
        );
        state.tagDedupCandidates = [
            ...state.tagDedupCandidates.slice(0, state.tagDedupCurrentIndex + 1),
            ...filtered,
        ];
    }

    async function handleDedupSkip() {
        if (state.tagDedupProcessing) return;
        state.tagDedupProcessing = true;

        const candidate = state.tagDedupCandidates[state.tagDedupCurrentIndex];
        if (!candidate) { state.tagDedupProcessing = false; return; }

        try {
            const requestId = `dismiss_${Date.now()}`;
            await runPluginTask('Dismiss Tag Merge', {
                mode: 'dismiss_tag_merge',
                tag_a_name: candidate.tag_a.name,
                tag_b_name: candidate.tag_b.name,
                request_id: requestId,
            });
        } catch (e) {
            log(`Dismiss error: ${e.message}`, 'warn');
        }

        state.tagDedupSkipCount++;
        state.tagDedupCurrentIndex++;
        state.tagDedupProcessing = false;
        renderDedupPair();
    }

    function handleDedupKeyboard(event) {
        // Don't capture if typing in an input, or not on dedup page
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
        if (!document.querySelector('.stash-copilot-dedup-page')) return;
        if (state.tagDedupProcessing) return;

        if (event.key === 'ArrowLeft') {
            event.preventDefault();
            handleDedupKeep('a');
        } else if (event.key === 'ArrowRight') {
            event.preventDefault();
            handleDedupKeep('b');
        } else if (event.key === 'ArrowDown' || event.key === 's') {
            event.preventDefault();
            handleDedupSkip();
        }
    }

    function renderDedupSummary() {
        const body = document.querySelector('.stash-copilot-dedup-body');
        if (!body) return;

        document.removeEventListener('keydown', handleDedupKeyboard);

        body.innerHTML = `
            <div class="stash-copilot-dedup-summary">
                <h3>Deduplication Complete</h3>
                <p class="stash-copilot-dedup-summary-subtitle">All candidate pairs have been reviewed.</p>
                <div class="stash-copilot-dedup-summary-stats">
                    <div class="stash-copilot-dedup-stat">
                        <div class="stash-copilot-dedup-stat-value">${state.tagDedupMergeCount}</div>
                        <div class="stash-copilot-dedup-stat-label">Tags Merged</div>
                    </div>
                    <div class="stash-copilot-dedup-stat">
                        <div class="stash-copilot-dedup-stat-value">${state.tagDedupSkipCount}</div>
                        <div class="stash-copilot-dedup-stat-label">Skipped</div>
                    </div>
                    <div class="stash-copilot-dedup-stat">
                        <div class="stash-copilot-dedup-stat-value">${state.tagDedupScenesUpdated}</div>
                        <div class="stash-copilot-dedup-stat-label">Scenes Updated</div>
                    </div>
                </div>
            </div>
        `;
    }

    function renderDedupEmpty(message) {
        const body = document.querySelector('.stash-copilot-dedup-body');
        if (!body) return;
        body.innerHTML = `
            <div class="stash-copilot-dedup-empty">
                <div class="stash-copilot-dedup-empty-icon">✓</div>
                <p>${escapeHtml(message)}</p>
            </div>
        `;
    }

    function renderDedupError(message) {
        const body = document.querySelector('.stash-copilot-dedup-body');
        if (!body) return;
        body.innerHTML = `
            <div class="stash-copilot-dedup-empty">
                <div class="stash-copilot-dedup-empty-icon">⚠</div>
                <p>${escapeHtml(message)}</p>
            </div>
        `;
    }

    // Initialize plugin
    function init() {
        if (state.initialized) {
            log('Already initialized', 'warn');
            return;
        }

        log('Initializing...');

        // Preload markdown libraries (non-blocking)
        loadMarkdownLibraries().catch(err => {
            log(`Markdown libraries unavailable, using fallback: ${err.message}`, 'warn');
        });

        // Add navbar buttons (with retry for SPA navigation)
        function tryAddNavbarButtons() {
            const navbar = document.querySelector('.navbar-buttons.ml-auto');
            if (navbar) {
                // Add main AI Insights dropdown button
                if (!document.getElementById('stash-copilot-nav-container')) {
                    createNavbarButton();
                }
                // Add AI Search button
                if (!document.getElementById('stash-copilot-search-nav-btn')) {
                    createSearchNavButton();
                }
                // Image Labeling and Tag Dedup nav buttons are intentionally
                // not injected. The underlying pages and route handlers stay
                // wired up for direct URL access; only the navbar entry
                // points are hidden. Re-enable by uncommenting below.
                // if (!document.getElementById('stash-copilot-labeling-nav-btn')) {
                //     createLabelingNavButton();
                // }
                // if (!document.getElementById('stash-copilot-dedup-nav-btn')) {
                //     createDedupNavButton();
                // }
            } else {
                // Retry after a short delay
                setTimeout(tryAddNavbarButtons, 500);
            }
        }

        tryAddNavbarButtons();

        // Setup handler for browser back/forward navigation from search/labeling pages
        setupSearchNavigationHandler();
        setupLabelingNavigationHandler();
        setupDedupNavigationHandler();

        // Listen for page changes (for scene enhancements)
        const stash = window.stash || {};
        if (stash.addEventListener) {
            stash.addEventListener('page:scene', enhanceScenePage);
            stash.addEventListener('page:scenes', enhanceScenesList);
        } else {
            // Fallback: Use URL change detection
            let lastPath = window.location.pathname;
            setInterval(() => {
                const currentPath = window.location.pathname;
                if (currentPath !== lastPath) {
                    lastPath = currentPath;
                    onPageChange();
                }
                // Also check if navbar buttons need to be re-added (SPA navigation)
                tryAddNavbarButtons();
            }, 500);

            // Initial page load
            onPageChange();
        }

        state.initialized = true;
        log('Initialized successfully');
    }

    // ===== Preference Trainer (Train Tab) =====

    const preferenceState = {
        sessionId: null,
        isTraining: false,
        pairs: [],
        pairIndex: 0,
        convergence: null,
        phase: 'broad',
        nComparisons: 0,
        responseStartTime: null,
        seenSceneIds: new Set(),
        explorationRate: 0.2,  // 0.0 (focused) to 1.0 (diverse)
        pureRandom: false,  // Skip cluster bootstrapping, use uniform random
    };

    /**
     * Load existing model stats from the backend
     */
    async function loadExistingTrainStats(container) {
        try {
            const requestId = `train_stats_${Date.now()}`;
            await runPluginTask('Get Preference Stats', {
                request_id: requestId,
            });

            // Poll for results
            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 20) {
                    clearInterval(poll);
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        if (data.convergence) {
                            updateTrainStats(container, data);
                        }
                    }
                } catch (_) { /* Not ready yet */ }
            }, 500);
        } catch (_) { /* Stats not available yet, that's ok */ }
    }

    /**
     * Render taste profile pills into a .stash-copilot-taste-profile container.
     * @param {HTMLElement} profileEl - The .stash-copilot-taste-profile element
     * @param {Array} tasteProfile - Array of {text, score, source} entries
     */
    function renderTasteProfile(profileEl, tasteProfile) {
        if (!profileEl) return;
        if (!tasteProfile || !tasteProfile.length) {
            profileEl.style.display = 'none';
            return;
        }

        const likes = tasteProfile.filter(t => t.score > 0);
        const dislikes = tasteProfile.filter(t => t.score < 0);

        const likesContainer = profileEl.querySelector('.stash-copilot-taste-profile-likes .stash-copilot-taste-profile-pills');
        const dislikesContainer = profileEl.querySelector('.stash-copilot-taste-profile-dislikes .stash-copilot-taste-profile-pills');

        if (likesContainer) {
            likesContainer.innerHTML = likes.map(t =>
                `<span class="stash-copilot-taste-pill stash-copilot-taste-pill--like" title="${Math.round(t.score * 100)}% match">${t.text}</span>`
            ).join('');
        }
        if (dislikesContainer) {
            dislikesContainer.innerHTML = dislikes.map(t =>
                `<span class="stash-copilot-taste-pill stash-copilot-taste-pill--dislike" title="${Math.round(Math.abs(t.score) * 100)}% match">${t.text}</span>`
            ).join('');
        }

        // Show likes section only if there are likes, same for dislikes
        const likesSection = profileEl.querySelector('.stash-copilot-taste-profile-likes');
        const dislikesSection = profileEl.querySelector('.stash-copilot-taste-profile-dislikes');
        if (likesSection) likesSection.style.display = likes.length ? '' : 'none';
        if (dislikesSection) dislikesSection.style.display = dislikes.length ? '' : 'none';

        profileEl.style.display = (likes.length || dislikes.length) ? '' : 'none';
    }

    /**
     * Update stats display with model data
     */
    function updateTrainStats(container, data) {
        const compEl = container.querySelector('[data-stat="comparisons"]');
        const confEl = container.querySelector('[data-stat="confidence"]');
        const phaseEl = container.querySelector('[data-stat="phase"]');

        if (compEl && data.n_comparisons !== undefined) {
            compEl.textContent = String(data.n_comparisons);
        }
        if (confEl && data.convergence) {
            confEl.textContent = `${data.convergence.confidence_pct}%`;
        }
        if (phaseEl && data.phase) {
            const phaseLabels = { broad: 'Exploring', refine: 'Refining', boundary: 'Fine-tuning' };
            const label = phaseLabels[data.phase] || data.phase;
            const phasePct = data.convergence && data.convergence.phase_progress_pct !== undefined
                ? Math.round(data.convergence.phase_progress_pct) : null;
            phaseEl.textContent = phasePct !== null ? `${label} ${phasePct}%` : label;
        }

        // Show/hide reset button based on whether model has been trained
        const resetBtn = container.querySelector('.stash-copilot-train-reset-btn');
        if (resetBtn) {
            resetBtn.style.display = (data.n_comparisons && data.n_comparisons > 0) ? '' : 'none';
        }

        // Render taste profile on the intro screen
        if (data.taste_profile) {
            const introEl = container.querySelector('.stash-copilot-train-intro');
            if (introEl) {
                const profileEl = introEl.querySelector('.stash-copilot-taste-profile');
                renderTasteProfile(profileEl, data.taste_profile);
            }
        }
    }

    /**
     * Set up Train tab event listeners
     */
    function setupTrainListeners(container) {
        // Start button
        const startBtn = container.querySelector('.stash-copilot-train-start-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => startTrainingSession(container));
        }

        // End button
        const endBtn = container.querySelector('.stash-copilot-train-end-btn');
        if (endBtn) {
            endBtn.addEventListener('click', () => endTrainingSession(container));
        }

        // Restart button
        const restartBtn = container.querySelector('.stash-copilot-train-restart-btn');
        if (restartBtn) {
            restartBtn.addEventListener('click', () => {
                container.querySelector('.stash-copilot-train-complete').style.display = 'none';
                container.querySelector('.stash-copilot-train-intro').style.display = '';
                startTrainingSession(container);
            });
        }

        // Reset model button
        const resetBtn = container.querySelector('.stash-copilot-train-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => resetPreferenceModel(container));
        }

        // Exploration rate slider
        const explorationSlider = container.querySelector('.stash-copilot-train-exploration-input');
        const explorationValue = container.querySelector('.stash-copilot-train-exploration-value');
        if (explorationSlider && explorationValue) {
            explorationSlider.addEventListener('input', () => {
                const pct = parseInt(explorationSlider.value, 10);
                explorationValue.textContent = `${pct}%`;
                preferenceState.explorationRate = pct / 100;
            });
        }

        // Pure random checkbox
        const pureRandomCheckbox = container.querySelector('.stash-copilot-train-pure-random-checkbox');
        if (pureRandomCheckbox) {
            pureRandomCheckbox.addEventListener('change', () => {
                preferenceState.pureRandom = pureRandomCheckbox.checked;
                log(`Pure random mode: ${preferenceState.pureRandom}`);
            });
        }

        // Refresh recommendations button
        const refreshBtn = container.querySelector('.stash-copilot-train-recs-refresh');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => loadPreferenceRecs(container));
        }

        // Swipe action buttons
        container.querySelectorAll('.stash-copilot-train-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                handleTrainSwipe(container, action);
            });
        });

        // Keyboard shortcuts
        const keyHandler = (e) => {
            if (!preferenceState.isTraining) return;
            // Don't capture when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch (e.key) {
                case 'ArrowRight':
                    e.preventDefault();
                    handleTrainSwipe(container, 'like');
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    handleTrainSwipe(container, 'dislike');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    handleTrainSwipe(container, 'super_like');
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    handleTrainSwipe(container, 'skip');
                    break;
            }
        };

        document.addEventListener('keydown', keyHandler);

        // Store handler reference for cleanup
        preferenceState._keyHandler = keyHandler;

        // Clean up when modal panel loses active class or is removed from DOM
        const observer = new MutationObserver(() => {
            const panel = container.closest('.stash-copilot-insights-panel');
            if (!panel || !panel.classList.contains('active') || !document.body.contains(panel)) {
                document.removeEventListener('keydown', keyHandler);
                preferenceState._keyHandler = null;
                observer.disconnect();
            }
        });

        const panel = container.closest('.stash-copilot-insights-panel');
        if (panel && panel.parentElement) {
            observer.observe(panel.parentElement, { attributes: true, subtree: true, childList: true });
        }

        // Also clean up when modal is removed from DOM
        const cleanupOnNav = () => {
            if (!document.body.contains(container)) {
                document.removeEventListener('keydown', keyHandler);
                preferenceState._keyHandler = null;
                window.removeEventListener('popstate', cleanupOnNav);
                observer.disconnect();
            }
        };
        window.addEventListener('popstate', cleanupOnNav);

        // Touch swipe gestures
        setupTouchSwipe(container);
    }

    /**
     * Set up touch swipe gestures for the preference trainer card area.
     * Enables Tinder-style drag-to-swipe on touch devices.
     */
    function setupTouchSwipe(container) {
        const cardArea = container.querySelector('.stash-copilot-train-card-area');
        if (!cardArea) return;

        const touchState = {
            startX: 0,
            startY: 0,
            currentX: 0,
            currentY: 0,
            isDragging: false,
            moved: false,
            startTime: 0,
        };

        // Expose moved flag so click handlers can check it
        cardArea._touchState = touchState;

        const DISTANCE_THRESHOLD = 80;
        const VELOCITY_THRESHOLD = 0.5; // px/ms

        function getSwipeCard() {
            return cardArea.querySelector('.stash-copilot-train-swipe-card');
        }

        function getCardStack() {
            return cardArea.querySelector('.stash-copilot-train-card-stack');
        }

        function getOverlay(direction) {
            const cardStack = getCardStack();
            if (!cardStack) return null;
            const cls = direction === 'super_like' ? 'super-like-overlay' : `${direction}-overlay`;
            return cardStack.querySelector(`.${cls}`);
        }

        function hideAllOverlays() {
            const cardStack = getCardStack();
            if (!cardStack) return;
            cardStack.querySelectorAll('.stash-copilot-train-swipe-overlay').forEach(o => {
                o.style.opacity = '';
                o.style.transform = '';
                o.classList.remove('visible');
            });
        }

        function getDirection(deltaX, deltaY) {
            const absX = Math.abs(deltaX);
            const absY = Math.abs(deltaY);
            if (absX > absY) {
                return deltaX > 0 ? 'like' : 'dislike';
            } else {
                return deltaY < 0 ? 'super_like' : 'skip';
            }
        }

        cardArea.addEventListener('touchstart', (e) => {
            if (!preferenceState.isTraining) return;
            const swipeCard = getSwipeCard();
            if (!swipeCard) return;

            const touch = e.touches[0];
            touchState.startX = touch.clientX;
            touchState.startY = touch.clientY;
            touchState.currentX = touch.clientX;
            touchState.currentY = touch.clientY;
            touchState.isDragging = true;
            touchState.moved = false;
            touchState.startTime = Date.now();

            swipeCard.classList.add('dragging');
            swipeCard.classList.remove('snap-back');
        }, { passive: false });

        cardArea.addEventListener('touchmove', (e) => {
            if (!touchState.isDragging) return;
            const swipeCard = getSwipeCard();
            if (!swipeCard) return;

            const touch = e.touches[0];
            touchState.currentX = touch.clientX;
            touchState.currentY = touch.clientY;

            const deltaX = touchState.currentX - touchState.startX;
            const deltaY = touchState.currentY - touchState.startY;

            // Mark as moved if dragged more than 10px (suppresses click)
            if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                touchState.moved = true;
            }

            // Prevent page scroll while dragging
            e.preventDefault();

            // Move the card with finger
            const rotation = deltaX * 0.1;
            swipeCard.style.transform = `translate(${deltaX}px, ${deltaY}px) rotate(${rotation}deg)`;

            // Show directional overlay with progressive opacity
            hideAllOverlays();
            const direction = getDirection(deltaX, deltaY);
            const dominantDelta = Math.abs(deltaX) > Math.abs(deltaY) ? Math.abs(deltaX) : Math.abs(deltaY);
            const cardWidth = swipeCard.offsetWidth || 300;
            const threshold = Math.min(DISTANCE_THRESHOLD, cardWidth * 0.3);
            const progress = Math.min(1, dominantDelta / threshold);

            const overlay = getOverlay(direction);
            if (overlay && progress > 0.1) {
                overlay.style.opacity = String(progress);
                overlay.style.transform = `translate(-50%, -50%) scale(${0.5 + progress * 0.5})`;
            }
        }, { passive: false });

        cardArea.addEventListener('touchend', (e) => {
            if (!touchState.isDragging) return;
            touchState.isDragging = false;

            const swipeCard = getSwipeCard();
            if (!swipeCard) return;

            swipeCard.classList.remove('dragging');

            const deltaX = touchState.currentX - touchState.startX;
            const deltaY = touchState.currentY - touchState.startY;
            const elapsed = Date.now() - touchState.startTime;
            const absX = Math.abs(deltaX);
            const absY = Math.abs(deltaY);
            const dominantDelta = absX > absY ? absX : absY;
            const cardWidth = swipeCard.offsetWidth || 300;
            const threshold = Math.min(DISTANCE_THRESHOLD, cardWidth * 0.3);
            const velocity = elapsed > 0 ? dominantDelta / elapsed : 0;

            const meetsThreshold = dominantDelta > threshold || velocity > VELOCITY_THRESHOLD;

            if (meetsThreshold && dominantDelta > 20) {
                // Swipe accepted — trigger the action
                const direction = getDirection(deltaX, deltaY);
                // Reset card transform so handleTrainSwipe animation plays cleanly
                swipeCard.style.transform = '';
                hideAllOverlays();
                handleTrainSwipe(container, direction);
            } else {
                // Snap back to center
                hideAllOverlays();
                swipeCard.classList.add('snap-back');
                swipeCard.style.transform = '';

                // Remove snap-back class after transition
                const cleanup = () => {
                    swipeCard.classList.remove('snap-back');
                    swipeCard.removeEventListener('transitionend', cleanup);
                };
                swipeCard.addEventListener('transitionend', cleanup);
                // Fallback cleanup if transitionend doesn't fire
                setTimeout(() => swipeCard.classList.remove('snap-back'), 350);
            }
        }, { passive: true });

        // Cancel drag on touch cancel (e.g. notification, multi-touch)
        cardArea.addEventListener('touchcancel', () => {
            if (!touchState.isDragging) return;
            touchState.isDragging = false;

            const swipeCard = getSwipeCard();
            if (swipeCard) {
                swipeCard.classList.remove('dragging');
                swipeCard.classList.add('snap-back');
                swipeCard.style.transform = '';
                setTimeout(() => swipeCard.classList.remove('snap-back'), 350);
            }
            hideAllOverlays();
        }, { passive: true });

        // Suppress click after drag to prevent accidental scene navigation
        cardArea.addEventListener('click', (e) => {
            if (touchState.moved) {
                e.preventDefault();
                e.stopPropagation();
                touchState.moved = false;
            }
        }, true); // capture phase to intercept before card click handlers
    }

    /**
     * Start a preference training session
     */
    async function startTrainingSession(container) {
        log('Starting preference training session...');

        // Show session UI, hide intro
        container.querySelector('.stash-copilot-train-intro').style.display = 'none';
        container.querySelector('.stash-copilot-train-complete').style.display = 'none';
        const sessionEl = container.querySelector('.stash-copilot-train-session');
        sessionEl.style.display = '';

        // Show loading
        const cardArea = container.querySelector('.stash-copilot-train-card-area');
        cardArea.innerHTML = `
            <div class="stash-copilot-train-loading">
                <div class="stash-copilot-spinner"></div>
                <span>Starting session...</span>
            </div>
        `;
        container.querySelector('.stash-copilot-train-actions').style.display = 'none';

        try {
            const requestId = `train_${Date.now()}`;
            await runPluginTask('Start Preference Session', {
                session_mode: 'swipe',
                batch_size: '20',
                request_id: requestId,
                exploration_rate: String(preferenceState.explorationRate),
                pure_random: String(preferenceState.pureRandom),
            });

            // Poll for session start results
            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 60) {
                    clearInterval(poll);
                    cardArea.innerHTML = '<div class="stash-copilot-train-error">Session start timed out. Make sure scenes are embedded first.</div>';
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.status === 'ready' || data.status === 'complete') {
                            clearInterval(poll);
                            onSessionStarted(container, data);
                        } else if (data.status === 'error') {
                            clearInterval(poll);
                            cardArea.innerHTML = `<div class="stash-copilot-train-error">${escapeHtml(data.error || 'Session failed to start')}</div>`;
                        }
                    }
                } catch (_) { /* Not ready yet */ }
            }, 500);
        } catch (err) {
            log(`Failed to start training: ${err}`, 'error');
            cardArea.innerHTML = `<div class="stash-copilot-train-error">Failed to start session: ${escapeHtml(err.message)}</div>`;
        }
    }

    /**
     * Handle session start response
     */
    function onSessionStarted(container, data) {
        log(`Training session started: ${data.session_id}, ${data.pairs.length} pairs`);

        preferenceState.sessionId = data.session_id;
        preferenceState.isTraining = true;
        preferenceState.pairs = data.pairs || [];
        preferenceState.pairIndex = 0;
        preferenceState.convergence = data.convergence;
        preferenceState.phase = data.phase;
        preferenceState.nComparisons = data.n_comparisons || 0;

        // Track all scene_a_ids from initial batch as seen
        for (const p of preferenceState.pairs) {
            preferenceState.seenSceneIds.add(p.scene_a_id);
        }

        // Update progress
        updateTrainProgress(container);

        // Show first scene
        if (preferenceState.pairs.length > 0) {
            renderTrainCard(container, preferenceState.pairs[0]);
            container.querySelector('.stash-copilot-train-actions').style.display = '';
        } else {
            const cardArea = container.querySelector('.stash-copilot-train-card-area');
            cardArea.innerHTML = '<div class="stash-copilot-train-error">No scene pairs available. Make sure scenes are embedded first.</div>';
        }
    }

    /**
     * Render a scene card in the training area
     */
    function renderTrainCard(container, pair) {
        const cardArea = container.querySelector('.stash-copilot-train-card-area');
        if (!cardArea) return;

        // Use scene_a for swipe mode (single scene presentation)
        const scene = pair.scene_a || {};
        const sceneId = pair.scene_a_id;

        // Build the card using the shared card system if scene data is available
        if (scene && scene.id) {
            const cardHtml = buildSceneCard({
                scene: scene,
                score: pair.predicted_probability || 0.5,
                cardIndex: 0,
                theme: 'preference',
                scoreLabel: 'confidence',
            });

            cardArea.innerHTML = `
                <div class="stash-copilot-train-card-stack">
                    <div class="stash-copilot-train-swipe-card" data-scene-id="${sceneId}">
                        ${cardHtml}
                    </div>
                    <div class="stash-copilot-train-swipe-overlay like-overlay">👍</div>
                    <div class="stash-copilot-train-swipe-overlay dislike-overlay">👎</div>
                    <div class="stash-copilot-train-swipe-overlay super-like-overlay">🔥</div>
                    <div class="stash-copilot-train-swipe-overlay skip-overlay">⏭</div>
                </div>
            `;

            // Setup card interactions
            setupSceneCardEvents(cardArea, {
                theme: 'preference',
                tooltipMode: 'fixed'
            });
        } else {
            // Fallback: minimal card with thumbnail
            const thumbUrl = `/plugin/stash-copilot/assets/embedded_frames/scene_${sceneId}/frame_0001.jpg`;
            cardArea.innerHTML = `
                <div class="stash-copilot-train-card-stack">
                    <div class="stash-copilot-train-swipe-card" data-scene-id="${sceneId}">
                        <div class="stash-copilot-train-minimal-card">
                            <img src="${thumbUrl}" alt="Scene ${sceneId}" onerror="this.src='/plugin/stash-copilot/assets/placeholder.png'">
                            <div class="stash-copilot-train-minimal-info">
                                <span>Scene #${sceneId}</span>
                            </div>
                        </div>
                    </div>
                    <div class="stash-copilot-train-swipe-overlay like-overlay">👍</div>
                    <div class="stash-copilot-train-swipe-overlay dislike-overlay">👎</div>
                    <div class="stash-copilot-train-swipe-overlay super-like-overlay">🔥</div>
                    <div class="stash-copilot-train-swipe-overlay skip-overlay">⏭</div>
                </div>
            `;
        }

        // Track response time start
        preferenceState.responseStartTime = Date.now();
    }

    /**
     * Handle a swipe action (optimistic - advances immediately, records in background)
     */
    async function handleTrainSwipe(container, direction) {
        if (!preferenceState.isTraining || !preferenceState.pairs[preferenceState.pairIndex]) {
            return;
        }

        const pair = preferenceState.pairs[preferenceState.pairIndex];
        const responseTime = preferenceState.responseStartTime
            ? Date.now() - preferenceState.responseStartTime
            : null;

        // Animate swipe
        const cardStack = container.querySelector('.stash-copilot-train-card-stack');
        const swipeCard = container.querySelector('.stash-copilot-train-swipe-card');
        if (swipeCard && cardStack) {
            const overlayClass = direction === 'super_like' ? 'super-like-overlay' : `${direction}-overlay`;
            const overlay = cardStack.querySelector(`.${overlayClass}`);
            if (overlay) {
                overlay.classList.add('visible');
            }

            swipeCard.classList.add(`swiping-${direction}`);

            // Brief animation pause
            await new Promise(r => setTimeout(r, 250));
        }

        // Show surprise toast immediately using pre-computed values
        if (direction !== 'skip') {
            const surprise = (direction === 'like' || direction === 'super_like')
                ? pair.surprise_if_liked
                : pair.surprise_if_disliked;
            if (surprise != null) {
                showSurpriseToast(container, surprise);
            }
        }

        // Track this scene as seen so it won't be shown again
        preferenceState.seenSceneIds.add(pair.scene_a_id);

        // Optimistically advance to next card immediately
        preferenceState.nComparisons++;
        advanceToNextCard(container);

        // Record the swipe in the background (don't block UI)
        recordSwipeInBackground(container, pair, direction, responseTime);
    }

    /**
     * Record swipe to backend without blocking the UI
     */
    function recordSwipeInBackground(container, pair, direction, responseTime) {
        const requestId = `swipe_${Date.now()}`;
        runPluginTask('Record Preference Swipe', {
            session_id: preferenceState.sessionId,
            scene_id: String(pair.scene_a_id),
            direction: direction,
            response_time_ms: responseTime ? String(responseTime) : '',
            request_id: requestId,
        }).then(() => {
            // Poll for response to get updated convergence and any new pairs
            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 40) {
                    clearInterval(poll);
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        onSwipeRecorded(container, data);
                    }
                } catch (_) { /* Not ready */ }
            }, 300);
        }).catch(err => {
            log(`Failed to record swipe: ${err}`, 'error');
        });
    }

    /**
     * Handle swipe recorded response - update state and show surprise feedback
     */
    function onSwipeRecorded(container, data) {
        // Update convergence from authoritative backend response
        if (data.convergence) {
            preferenceState.convergence = data.convergence;
            preferenceState.nComparisons = data.n_comparisons || preferenceState.nComparisons;
            preferenceState.phase = data.phase || preferenceState.phase;
            updateTrainProgress(container);
        }

        // If we got new pairs from the model, append unseen ones to the queue
        if (data.pairs && data.pairs.length > 0) {
            const newPairs = data.pairs.filter(p => !preferenceState.seenSceneIds.has(p.scene_a_id));
            for (const p of newPairs) {
                preferenceState.seenSceneIds.add(p.scene_a_id);
                preferenceState.pairs.push(p);
            }
        }
    }

    /**
     * Show a brief toast when the model was surprised by the user's swipe.
     * surprise is 0-1: 0 = perfectly predicted, 1 = completely wrong.
     */
    function showSurpriseToast(container, surprise) {
        // Only show for notable surprises
        if (surprise < 0.7) return;

        // Remove any existing toast
        const existing = container.querySelector('.stash-copilot-surprise-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'stash-copilot-surprise-toast';

        if (surprise >= 0.85) {
            toast.setAttribute('data-level', 'high');
            const messages = [
                'Plot twist!',
                'Didn\'t see that coming',
                'Surprise!',
                'Mind changed!',
            ];
            toast.textContent = messages[Math.floor(Math.random() * messages.length)];
        } else {
            toast.setAttribute('data-level', 'medium');
            const messages = [
                'Interesting...',
                'Noted!',
                'Unexpected',
            ];
            toast.textContent = messages[Math.floor(Math.random() * messages.length)];
        }

        // Overlay above the session (absolute positioned)
        const session = container.querySelector('.stash-copilot-train-session');
        if (session) {
            session.appendChild(toast);
            setTimeout(() => toast.remove(), 2500);
        }
    }

    /**
     * Move to the next card in the queue
     */
    function advanceToNextCard(container) {
        preferenceState.pairIndex++;
        updateTrainProgress(container);

        if (preferenceState.pairIndex < preferenceState.pairs.length) {
            renderTrainCard(container, preferenceState.pairs[preferenceState.pairIndex]);
        } else {
            // Queue exhausted — fetch more pairs instead of ending session
            fetchMorePairs(container);
        }
    }

    /**
     * Fetch a fresh batch of pairs when the current queue runs out.
     * Shows a loading spinner while waiting, then continues training.
     */
    async function fetchMorePairs(container) {
        const cardArea = container.querySelector('.stash-copilot-train-card-area');
        if (cardArea) {
            cardArea.innerHTML = `
                <div class="stash-copilot-train-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Loading more scenes...</span>
                </div>
            `;
        }

        try {
            const requestId = `refill_${Date.now()}`;
            await runPluginTask('Start Preference Session', {
                session_mode: 'swipe',
                batch_size: '20',
                request_id: requestId,
                exploration_rate: String(preferenceState.explorationRate),
                pure_random: String(preferenceState.pureRandom),
            });

            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 60) {
                    clearInterval(poll);
                    if (cardArea) {
                        cardArea.innerHTML = '<div class="stash-copilot-train-error">Failed to load more scenes. Try ending and restarting the session.</div>';
                    }
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        if (data.status === 'ready' || data.status === 'complete') {
                            clearInterval(poll);
                            if (data.pairs && data.pairs.length > 0) {
                                // Filter out any scenes already seen this session
                                const freshPairs = data.pairs.filter(p => !preferenceState.seenSceneIds.has(p.scene_a_id));
                                if (freshPairs.length === 0) {
                                    // All scenes already seen — show exhausted message
                                    clearInterval(poll);
                                    if (cardArea) {
                                        cardArea.innerHTML = '<div class="stash-copilot-train-error">No more unseen scenes available. You may have seen them all!</div>';
                                    }
                                    return;
                                }
                                preferenceState.pairs = freshPairs;
                                preferenceState.pairIndex = 0;
                                preferenceState.sessionId = data.session_id || preferenceState.sessionId;
                                if (data.convergence) {
                                    preferenceState.convergence = data.convergence;
                                    preferenceState.nComparisons = data.n_comparisons || preferenceState.nComparisons;
                                    preferenceState.phase = data.phase || preferenceState.phase;
                                }
                                // Track new pairs as seen
                                for (const p of preferenceState.pairs) {
                                    preferenceState.seenSceneIds.add(p.scene_a_id);
                                }
                                updateTrainProgress(container);
                                renderTrainCard(container, preferenceState.pairs[0]);
                            } else {
                                if (cardArea) {
                                    cardArea.innerHTML = '<div class="stash-copilot-train-error">No more scenes available. You may have seen them all!</div>';
                                }
                            }
                        } else if (data.status === 'error') {
                            clearInterval(poll);
                            if (cardArea) {
                                cardArea.innerHTML = `<div class="stash-copilot-train-error">${escapeHtml(data.error || 'Failed to load more scenes')}</div>`;
                            }
                        }
                    }
                } catch (_) { /* Not ready yet */ }
            }, 500);
        } catch (err) {
            log(`Failed to fetch more pairs: ${err}`, 'error');
            if (cardArea) {
                cardArea.innerHTML = `<div class="stash-copilot-train-error">Failed to load more scenes: ${escapeHtml(err.message)}</div>`;
            }
        }
    }

    /**
     * Update progress bar and confidence meter
     */
    function updateTrainProgress(container) {
        const nComparisons = preferenceState.nComparisons || 0;
        const phaseLabels = { broad: 'Exploring', refine: 'Refining', boundary: 'Fine-tuning' };
        const phaseName = phaseLabels[preferenceState.phase] || preferenceState.phase || '';
        const phasePct = preferenceState.convergence && preferenceState.convergence.phase_progress_pct !== undefined
            ? Math.round(preferenceState.convergence.phase_progress_pct) : null;
        const phaseLabel = phasePct !== null ? `${phaseName} ${phasePct}%` : phaseName;

        const progressFill = container.querySelector('.stash-copilot-train-progress-fill');
        const progressText = container.querySelector('.stash-copilot-train-progress-text');
        // Confidence-based fill (already shown in separate meter, use comparisons for visual)
        const conf = preferenceState.convergence ? (preferenceState.convergence.confidence_pct || 0) : 0;
        if (progressFill) progressFill.style.width = `${Math.min(100, conf)}%`;
        if (progressText) progressText.textContent = `${nComparisons} swipes · ${phaseLabel}`;

        // Update confidence
        if (preferenceState.convergence) {
            const conf = preferenceState.convergence.confidence_pct || 0;
            const confFill = container.querySelector('.stash-copilot-train-confidence-fill');
            const confText = container.querySelector('.stash-copilot-train-confidence-text');
            if (confFill) confFill.style.width = `${conf}%`;
            if (confText) confText.textContent = `${conf}% confident`;
        }
    }

    /**
     * End the training session
     */
    async function endTrainingSession(container) {
        if (!preferenceState.sessionId) {
            // No session to end, just show complete
            showTrainComplete(container);
            return;
        }

        log(`Ending training session: ${preferenceState.sessionId}`);
        preferenceState.isTraining = false;

        try {
            const requestId = `end_${Date.now()}`;
            await runPluginTask('End Preference Session', {
                session_id: preferenceState.sessionId,
                request_id: requestId,
            });

            // Poll for end results
            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 30) {
                    clearInterval(poll);
                    showTrainComplete(container);
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        if (data.convergence) {
                            preferenceState.convergence = data.convergence;
                            preferenceState.nComparisons = data.n_comparisons || preferenceState.nComparisons;
                        }
                        if (data.taste_profile) {
                            preferenceState.tasteProfile = data.taste_profile;
                        }
                        showTrainComplete(container);
                    }
                } catch (_) { /* Not ready */ }
            }, 500);
        } catch (err) {
            log(`Failed to end session: ${err}`, 'error');
            showTrainComplete(container);
        }
    }

    /**
     * Load preference-based recommendations into the Train panel
     */
    /**
     * Reset the preference model (delete all learned data for current model_key)
     */
    async function resetPreferenceModel(container) {
        if (!confirm('This will delete all your trained preference data and start fresh. Continue?')) {
            return;
        }

        const resetBtn = container.querySelector('.stash-copilot-train-reset-btn');
        if (resetBtn) {
            resetBtn.disabled = true;
            resetBtn.textContent = 'Resetting...';
        }

        try {
            const requestId = `reset_${Date.now()}`;
            await runPluginTask('Reset Preference Model', {
                mode: 'preference_reset',
                request_id: requestId,
            });

            // Poll for results
            const resultFile = `/plugin/stash-copilot/assets/preference_trainer_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 30) {
                    clearInterval(poll);
                    if (resetBtn) {
                        resetBtn.disabled = false;
                        resetBtn.textContent = 'Reset Model';
                    }
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);

                        // Update stats display to reflect reset state
                        updateTrainStats(container, data);

                        // Hide recs section since model is now empty
                        const recsSection = container.querySelector('.stash-copilot-train-recs');
                        if (recsSection) {
                            const grid = recsSection.querySelector('.stash-copilot-train-recs-grid');
                            const emptyEl = recsSection.querySelector('.stash-copilot-train-recs-empty');
                            if (grid) { grid.innerHTML = ''; grid.style.display = 'none'; }
                            if (emptyEl) {
                                emptyEl.querySelector('p').textContent = 'Complete at least one training session to see recommendations here.';
                                emptyEl.style.display = '';
                            }
                        }

                        // Reset button back to normal but hidden (0 comparisons)
                        if (resetBtn) {
                            resetBtn.disabled = false;
                            resetBtn.textContent = 'Reset Model';
                            resetBtn.style.display = 'none';
                        }

                        log('Preference model reset successfully');
                    }
                } catch (_) { /* Not ready */ }
            }, 500);
        } catch (err) {
            log(`Failed to reset preference model: ${err}`, 'error');
            if (resetBtn) {
                resetBtn.disabled = false;
                resetBtn.textContent = 'Reset Model';
            }
        }
    }

    /**
     * Load preference-based recommendations into the Train panel
     */
    async function loadPreferenceRecs(container) {
        const recsSection = container.querySelector('.stash-copilot-train-recs');
        if (!recsSection) return;

        const grid = recsSection.querySelector('.stash-copilot-train-recs-grid');
        const loadingEl = recsSection.querySelector('.stash-copilot-train-recs-loading');
        const emptyEl = recsSection.querySelector('.stash-copilot-train-recs-empty');

        // Show the section with loading state
        recsSection.style.display = '';
        grid.innerHTML = '';
        grid.style.display = 'none';
        emptyEl.style.display = 'none';
        loadingEl.style.display = '';

        const requestId = `pref_recs_${Date.now()}`;

        try {
            await runPluginTask('Get Preference Recommendations', {
                mode: 'preference_recs',
                rec_mode: 'discover',
                limit: '24',
                request_id: requestId,
            });

            // Poll for results
            const resultFile = `/plugin/stash-copilot/assets/preference_recs_${requestId}.json`;
            let attempts = 0;
            const poll = setInterval(async () => {
                attempts++;
                if (attempts > 40) {
                    clearInterval(poll);
                    loadingEl.style.display = 'none';
                    emptyEl.style.display = '';
                    emptyEl.querySelector('p').textContent = 'Recommendation loading timed out. Try refreshing.';
                    return;
                }
                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        loadingEl.style.display = 'none';

                        if (data.status === 'no_model' || data.status === 'no_embeddings' || data.status === 'no_candidates') {
                            emptyEl.style.display = '';
                            if (data.status === 'no_model') {
                                emptyEl.querySelector('p').textContent = 'Complete at least one training session to see recommendations here.';
                            } else if (data.status === 'no_embeddings') {
                                emptyEl.querySelector('p').textContent = 'No scene embeddings found. Run "Embed All Scenes" first.';
                            } else {
                                emptyEl.querySelector('p').textContent = 'No unwatched scenes available for recommendations.';
                            }
                            return;
                        }

                        if (data.results && data.results.length > 0) {
                            renderPreferenceRecs(container, data);
                        } else {
                            emptyEl.style.display = '';
                        }
                    }
                } catch (_) { /* Not ready yet */ }
            }, 500);
        } catch (err) {
            log(`Failed to load preference recs: ${err}`, 'error');
            loadingEl.style.display = 'none';
            emptyEl.style.display = '';
            emptyEl.querySelector('p').textContent = 'Failed to load recommendations.';
        }
    }

    /**
     * Render preference-based recommendation cards
     */
    function renderPreferenceRecs(container, data) {
        const recsSection = container.querySelector('.stash-copilot-train-recs');
        if (!recsSection) return;

        const grid = recsSection.querySelector('.stash-copilot-train-recs-grid');
        grid.innerHTML = '';
        grid.style.display = '';

        const results = data.results || [];

        results.forEach((result, index) => {
            const cardHtml = buildSceneCard({
                scene: result.scene,
                score: result.preference_score,
                cardIndex: index,
                theme: 'preference',
                scoreLabel: 'preference',
            });
            grid.insertAdjacentHTML('beforeend', cardHtml);
        });

        setupSceneCardEvents(grid, { theme: 'preference', tooltipMode: 'fixed' });
    }

    /**
     * Show training complete screen
     */
    function showTrainComplete(container) {
        preferenceState.isTraining = false;

        container.querySelector('.stash-copilot-train-session').style.display = 'none';
        container.querySelector('.stash-copilot-train-intro').style.display = 'none';
        const completeEl = container.querySelector('.stash-copilot-train-complete');
        completeEl.style.display = '';

        // Update final stats
        const conv = preferenceState.convergence;
        const compEl = completeEl.querySelector('[data-stat="final-comparisons"]');
        const confEl = completeEl.querySelector('[data-stat="final-confidence"]');
        const phaseEl = completeEl.querySelector('[data-stat="final-phase"]');

        if (compEl) compEl.textContent = String(preferenceState.nComparisons);
        if (confEl && conv) confEl.textContent = `${conv.confidence_pct}%`;
        if (phaseEl) {
            const phaseLabels = { broad: 'Exploring', refine: 'Refining', boundary: 'Fine-tuning' };
            const label = phaseLabels[preferenceState.phase] || preferenceState.phase;
            const phasePct = conv && conv.phase_progress_pct !== undefined
                ? Math.round(conv.phase_progress_pct) : null;
            phaseEl.textContent = phasePct !== null ? `${label} ${phasePct}%` : label;
        }

        // Render taste profile on the complete screen
        if (preferenceState.tasteProfile) {
            const profileEl = completeEl.querySelector('.stash-copilot-taste-profile');
            renderTasteProfile(profileEl, preferenceState.tasteProfile);
        }

        // Also update the intro stats for next time (include taste profile)
        updateTrainStats(container, {
            n_comparisons: preferenceState.nComparisons,
            convergence: conv,
            phase: preferenceState.phase,
            taste_profile: preferenceState.tasteProfile,
        });

        // Auto-refresh preference recommendations after session ends
        loadPreferenceRecs(container);
    }

    // ===== Tag Suggestions Functions =====

    /**
     * Render Tags tab content for sidebar
     */
    function renderSidebarTagsContent(container, sceneId) {
        tagSuggestionState.sceneId = sceneId;

        container.innerHTML = `
            <div class="stash-copilot-sidebar-tags-tab">
                <div class="stash-copilot-sidebar-header">
                    <span class="stash-copilot-sidebar-title">Tag Suggestions</span>
                    <div class="stash-copilot-sidebar-actions">
                        <button class="stash-copilot-sidebar-btn stash-copilot-clear-dismissed-btn" title="Clear Dismissed Tags">↻</button>
                    </div>
                </div>
                <div class="stash-copilot-sidebar-intro stash-copilot-tags-intro">
                    <div class="stash-copilot-sidebar-intro-icon">✨</div>
                    <h3 class="stash-copilot-sidebar-intro-title">AI Tag Suggestions</h3>
                    <p class="stash-copilot-sidebar-intro-description">
                        Find tags that match this scene's visual content by comparing frames
                        against your tag vocabulary. Suggestions are based on embedding similarity.
                    </p>
                    <button class="stash-copilot-sidebar-analyze-btn stash-copilot-suggest-tags-btn">
                        Suggest Tags
                    </button>
                </div>
                <div class="stash-copilot-tags-loading" style="display: none;">
                    <div class="stash-copilot-spinner"></div>
                    <span class="stash-copilot-tags-status">Analyzing scene...</span>
                </div>
                <div class="stash-copilot-tags-content" style="display: none;"></div>
                <div class="stash-copilot-tags-error" style="display: none;"></div>
            </div>
        `;

        // Suggest Tags button handler
        const suggestBtn = container.querySelector('.stash-copilot-suggest-tags-btn');
        suggestBtn.addEventListener('click', () => {
            runTagSuggestions(container, sceneId);
        });

        // Clear Dismissed button handler
        const clearBtn = container.querySelector('.stash-copilot-clear-dismissed-btn');
        clearBtn.addEventListener('click', async () => {
            clearBtn.disabled = true;
            clearBtn.textContent = '...';
            try {
                await runPluginTask('Clear Dismissed Tags', { scene_id: String(sceneId) });
                clearBtn.textContent = '✓';
                setTimeout(() => {
                    clearBtn.textContent = '↻';
                    clearBtn.disabled = false;
                }, 1500);
            } catch (e) {
                log(`Failed to clear dismissed tags: ${e.message}`, 'error');
                clearBtn.textContent = '✗';
                setTimeout(() => {
                    clearBtn.textContent = '↻';
                    clearBtn.disabled = false;
                }, 1500);
            }
        });
    }

    /**
     * Run tag suggestion analysis for a scene
     */
    async function runTagSuggestions(container, sceneId) {
        const introEl = container.querySelector('.stash-copilot-tags-intro');
        const loadingEl = container.querySelector('.stash-copilot-tags-loading');
        const contentEl = container.querySelector('.stash-copilot-tags-content');
        const errorEl = container.querySelector('.stash-copilot-tags-error');
        const statusEl = container.querySelector('.stash-copilot-tags-status');

        // Hide intro, show loading
        if (introEl) introEl.style.display = 'none';
        if (errorEl) errorEl.style.display = 'none';
        if (contentEl) contentEl.style.display = 'none';
        if (loadingEl) loadingEl.style.display = 'flex';

        const requestId = `tags_${sceneId}_${Date.now()}`;
        tagSuggestionState.loading = true;
        tagSuggestionState.error = null;

        try {
            await runPluginTask('Get Tag Suggestions', {
                scene_id: String(sceneId),
                request_id: requestId,
            });
            pollTagSuggestions(container, sceneId, requestId, statusEl);
        } catch (error) {
            log(`Tag suggestions error: ${error.message}`, 'error');
            tagSuggestionState.loading = false;
            tagSuggestionState.error = error.message;
            if (loadingEl) loadingEl.style.display = 'none';
            if (errorEl) {
                errorEl.style.display = 'block';
                errorEl.innerHTML = `<div class="stash-copilot-error-message">Error: ${escapeHtml(error.message)}</div>`;
            }
        }
    }

    /**
     * Poll for tag suggestion results
     */
    async function pollTagSuggestions(container, sceneId, requestId, statusEl) {
        const loadingEl = container.querySelector('.stash-copilot-tags-loading');
        const contentEl = container.querySelector('.stash-copilot-tags-content');
        const errorEl = container.querySelector('.stash-copilot-tags-error');

        const maxAttempts = 60;  // 60 seconds timeout
        const pollInterval = 1000;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const response = await fetch(`/plugin/stash-copilot/assets/tag_suggestions_${requestId}.json?t=${Date.now()}`, {
                    cache: 'no-store'
                });
                if (response.ok) {
                    const data = await response.json();

                    if (data.status === 'complete') {
                        tagSuggestionState.suggestions = data.suggestions || [];
                        tagSuggestionState.currentPage = 0;
                        tagSuggestionState.loading = false;

                        if (loadingEl) loadingEl.style.display = 'none';
                        if (contentEl) {
                            contentEl.style.display = 'block';
                            renderTagSuggestions(contentEl, sceneId);
                        }
                        return;
                    } else if (data.status === 'error') {
                        tagSuggestionState.loading = false;
                        tagSuggestionState.error = data.error || 'Unknown error';
                        if (loadingEl) loadingEl.style.display = 'none';
                        if (errorEl) {
                            errorEl.style.display = 'block';
                            errorEl.innerHTML = `<div class="stash-copilot-error-message">${escapeHtml(data.error || 'Unknown error')}</div>`;
                        }
                        return;
                    } else if (data.status === 'no_embeddings') {
                        tagSuggestionState.loading = false;
                        if (loadingEl) loadingEl.style.display = 'none';
                        if (errorEl) {
                            errorEl.style.display = 'block';
                            errorEl.innerHTML = `
                                <div class="stash-copilot-error-message">
                                    <p>No embeddings found for this scene.</p>
                                    <p class="stash-copilot-error-hint">Run <strong>Embed All Scenes</strong> from AI Insights to generate embeddings first.</p>
                                </div>
                            `;
                        }
                        return;
                    } else if (data.status === 'processing' && statusEl) {
                        statusEl.textContent = data.message || 'Processing...';
                    }
                }
            } catch (e) {
                // File not ready yet, continue polling
            }
            await new Promise(r => setTimeout(r, pollInterval));
        }

        // Timeout
        tagSuggestionState.loading = false;
        tagSuggestionState.error = 'Timeout waiting for results';
        if (loadingEl) loadingEl.style.display = 'none';
        if (errorEl) {
            errorEl.style.display = 'block';
            errorEl.innerHTML = `<div class="stash-copilot-error-message">Timeout waiting for results. Please try again.</div>`;
        }
    }

    /**
     * Render tag suggestion cards
     */
    function renderTagSuggestions(contentEl, sceneId) {
        const { suggestions, currentPage, suggestionsPerPage } = tagSuggestionState;
        const totalPages = Math.ceil(suggestions.length / suggestionsPerPage);
        const start = currentPage * suggestionsPerPage;
        const pageSuggestions = suggestions.slice(start, start + suggestionsPerPage);

        if (suggestions.length === 0) {
            contentEl.innerHTML = `
                <div class="stash-copilot-tags-empty">
                    <p>No tag suggestions found for this scene.</p>
                    <p class="stash-copilot-tags-empty-hint">This could mean all matching tags are already applied, or no tags in your vocabulary match this scene's content.</p>
                </div>
            `;
            return;
        }

        let html = '<div class="stash-copilot-suggestions-list">';

        for (const suggestion of pageSuggestions) {
            const scorePercent = Math.round(suggestion.max_similarity * 100);
            const scoreClass = scorePercent >= 70 ? 'high-confidence' : scorePercent >= 50 ? 'medium-confidence' : '';

            html += `
                <div class="stash-copilot-suggestion-card" data-tag-id="${suggestion.tag_id}">
                    <div class="stash-copilot-suggestion-header">
                        <span class="stash-copilot-tag-name">${escapeHtml(suggestion.tag_name)}</span>
                        <span class="stash-copilot-tag-score ${scoreClass}">${scorePercent}%</span>
                    </div>
                    <div class="stash-copilot-evidence-frames">
                        ${(suggestion.evidence_frames || []).slice(0, 4).map(frame => `
                            <div class="stash-copilot-evidence-frame" data-timestamp="${frame.timestamp}" title="${frame.timestamp}">
                                <img src="/plugin/stash-copilot/${frame.thumbnail_path}" alt="Frame ${frame.frame_index}" onerror="this.parentElement.style.display='none'">
                                <span class="stash-copilot-frame-similarity">${Math.round(frame.similarity * 100)}%</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="stash-copilot-suggestion-meta">
                        <span>${suggestion.frame_count} frame${suggestion.frame_count !== 1 ? 's' : ''} matched</span>
                        <span class="stash-copilot-suggestion-avg">avg ${Math.round(suggestion.mean_similarity * 100)}%</span>
                    </div>
                    <div class="stash-copilot-suggestion-actions">
                        <button class="stash-copilot-btn stash-copilot-btn-apply" data-scene-id="${sceneId}" data-tag-id="${suggestion.tag_id}" data-tag-name="${escapeHtml(suggestion.tag_name)}">
                            <span class="btn-icon">✓</span> Apply
                        </button>
                        <button class="stash-copilot-btn stash-copilot-btn-dismiss" data-scene-id="${sceneId}" data-tag-id="${suggestion.tag_id}">
                            <span class="btn-icon">✕</span> Dismiss
                        </button>
                    </div>
                </div>
            `;
        }

        html += '</div>';

        // Pagination
        if (totalPages > 1) {
            html += `
                <div class="stash-copilot-tags-pagination">
                    <button class="stash-copilot-btn stash-copilot-btn-prev" ${currentPage === 0 ? 'disabled' : ''}>
                        <span>&#9664;</span>
                    </button>
                    <span class="stash-copilot-pagination-info">${currentPage + 1} / ${totalPages}</span>
                    <button class="stash-copilot-btn stash-copilot-btn-next" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>
                        <span>&#9654;</span>
                    </button>
                </div>
            `;
        }

        // Summary
        html += `
            <div class="stash-copilot-tags-summary">
                ${suggestions.length} suggestion${suggestions.length !== 1 ? 's' : ''} found
            </div>
        `;

        contentEl.innerHTML = html;
        setupTagSuggestionEvents(contentEl, sceneId);
    }

    /**
     * Setup event handlers for tag suggestion cards
     */
    function setupTagSuggestionEvents(contentEl, sceneId) {
        // Apply button handlers
        contentEl.querySelectorAll('.stash-copilot-btn-apply').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const tagId = parseInt(btn.dataset.tagId);
                const tagName = btn.dataset.tagName;
                btn.disabled = true;
                btn.innerHTML = '<span class="btn-icon">...</span>';

                try {
                    // Add the tag to the scene
                    await addTagToScene(sceneId, tagName);

                    const card = btn.closest('.stash-copilot-suggestion-card');
                    card.classList.add('stash-copilot-card-applied');
                    btn.innerHTML = '<span class="btn-icon">✓</span> Applied';
                    btn.classList.add('stash-copilot-btn-success');

                    // Remove from suggestions after animation
                    setTimeout(() => {
                        tagSuggestionState.suggestions = tagSuggestionState.suggestions.filter(s => s.tag_id !== tagId);
                        // Adjust page if needed
                        const totalPages = Math.ceil(tagSuggestionState.suggestions.length / tagSuggestionState.suggestionsPerPage);
                        if (tagSuggestionState.currentPage >= totalPages && totalPages > 0) {
                            tagSuggestionState.currentPage = totalPages - 1;
                        }
                        renderTagSuggestions(contentEl, sceneId);
                    }, 400);
                } catch (error) {
                    log(`Failed to apply tag: ${error.message}`, 'error');
                    btn.innerHTML = '<span class="btn-icon">✗</span> Failed';
                    btn.classList.add('stash-copilot-btn-error');
                    btn.disabled = false;
                    setTimeout(() => {
                        btn.innerHTML = '<span class="btn-icon">✓</span> Apply';
                        btn.classList.remove('stash-copilot-btn-error');
                    }, 2000);
                }
            });
        });

        // Dismiss button handlers
        contentEl.querySelectorAll('.stash-copilot-btn-dismiss').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const tagId = parseInt(btn.dataset.tagId);
                btn.disabled = true;
                btn.innerHTML = '<span class="btn-icon">...</span>';

                try {
                    await runPluginTask('Dismiss Suggested Tag', {
                        scene_id: String(sceneId),
                        tag_id: String(tagId)
                    });

                    const card = btn.closest('.stash-copilot-suggestion-card');
                    card.classList.add('stash-copilot-card-dismissed');

                    // Remove from suggestions after animation
                    setTimeout(() => {
                        tagSuggestionState.suggestions = tagSuggestionState.suggestions.filter(s => s.tag_id !== tagId);
                        // Adjust page if needed
                        const totalPages = Math.ceil(tagSuggestionState.suggestions.length / tagSuggestionState.suggestionsPerPage);
                        if (tagSuggestionState.currentPage >= totalPages && totalPages > 0) {
                            tagSuggestionState.currentPage = totalPages - 1;
                        }
                        renderTagSuggestions(contentEl, sceneId);
                    }, 400);
                } catch (error) {
                    log(`Failed to dismiss tag: ${error.message}`, 'error');
                    btn.innerHTML = '<span class="btn-icon">✗</span>';
                    btn.disabled = false;
                    setTimeout(() => {
                        btn.innerHTML = '<span class="btn-icon">✕</span> Dismiss';
                    }, 2000);
                }
            });
        });

        // Pagination handlers
        const prevBtn = contentEl.querySelector('.stash-copilot-btn-prev');
        const nextBtn = contentEl.querySelector('.stash-copilot-btn-next');

        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                if (tagSuggestionState.currentPage > 0) {
                    tagSuggestionState.currentPage--;
                    renderTagSuggestions(contentEl, sceneId);
                }
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                const totalPages = Math.ceil(tagSuggestionState.suggestions.length / tagSuggestionState.suggestionsPerPage);
                if (tagSuggestionState.currentPage < totalPages - 1) {
                    tagSuggestionState.currentPage++;
                    renderTagSuggestions(contentEl, sceneId);
                }
            });
        }

        // Evidence frame click -> seek video
        contentEl.querySelectorAll('.stash-copilot-evidence-frame').forEach(frame => {
            frame.addEventListener('click', () => {
                const timestamp = frame.dataset.timestamp;
                if (timestamp) {
                    seekVideoToTimestamp(timestamp);
                }
            });
        });
    }

    /**
     * Seek video to a timestamp (format: "M:SS" or "MM:SS")
     */
    function seekVideoToTimestamp(timestamp) {
        const parts = timestamp.split(':').map(Number);
        let totalSeconds = 0;
        if (parts.length === 2) {
            totalSeconds = parts[0] * 60 + parts[1];
        } else if (parts.length === 3) {
            totalSeconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
        }
        const video = document.querySelector('video');
        if (video) {
            video.currentTime = totalSeconds;
            video.play().catch(() => {});  // Ignore autoplay restrictions
        }
    }

    // ===== End Tag Suggestions Functions =====

    // ===== EroScripts: Sidebar Tab + Modal =====
    // 5th sidebar tab triggers a modal picker that searches
    // discuss.eroscripts.com via the backend tasks. All Stash plugin task IO
    // uses the runPluginTask + JSON-poll pattern established by the rest of
    // this file. We avoid innerHTML for any user-controlled string —
    // eroscripts post titles, usernames, etc. flow through the DOM API
    // (createElement + textContent) to make XSS impossible by construction.

    const EROS_POLL_INTERVAL_MS = 250;
    const EROS_POLL_TIMEOUT_MS = 30000;
    const EROS_ASSET_BASE = '/plugin/stash-copilot/assets/eroscripts';

    const eroState = {
        modal: null,
        sceneId: null,
        pollAbort: null,
    };

    function eroNewRequestId() {
        return 'r' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    }

    /**
     * Run a backend task and poll its asset JSON until status is final.
     */
    async function eroRunAndPoll(taskName, extraArgs, resultPrefix, signal) {
        const requestId = eroNewRequestId();
        const args = Object.assign({ request_id: requestId }, extraArgs || {});
        await runPluginTask(taskName, args);

        const url = EROS_ASSET_BASE + '/' + resultPrefix + '_' + requestId + '.json';
        const start = Date.now();
        while (Date.now() - start < EROS_POLL_TIMEOUT_MS) {
            if (signal && signal.aborted) {
                throw new DOMException('Aborted', 'AbortError');
            }
            try {
                const resp = await fetch(url + '?t=' + Date.now(), { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data && (data.status === 'complete' || data.status === 'error')) {
                        return data;
                    }
                }
            } catch (e) { /* still pending — keep polling */ }
            await new Promise(r => setTimeout(r, EROS_POLL_INTERVAL_MS));
        }
        throw new Error('EroScripts ' + resultPrefix + ' timed out');
    }

    // -- DOM helpers (textContent-based, immune to XSS by construction) ----

    function eroEl(tag, opts) {
        const el = document.createElement(tag);
        if (!opts) return el;
        if (opts.cls) el.className = opts.cls;
        if (opts.text !== undefined) el.textContent = opts.text;
        if (opts.attrs) {
            for (const k of Object.keys(opts.attrs)) {
                el.setAttribute(k, opts.attrs[k]);
            }
        }
        if (opts.title) el.title = opts.title;
        if (opts.children) {
            for (const c of opts.children) if (c) el.appendChild(c);
        }
        return el;
    }

    /**
     * Render content of the Scripts sidebar tab.
     *
     * Async because we first probe `EroScripts Status` to decide which of
     * four states to render: matched (rich card), orphan_local (funscript
     * but no sidecar), orphan_metadata (sidecar but no file), or empty.
     */
    async function renderSidebarScriptsContent(container, sceneId) {
        container.textContent = '';
        const loading = eroEl('div', { cls: 'stash-copilot-sidebar-scripts-loading' });
        loading.appendChild(eroEl('div', { cls: 'stash-copilot-spinner' }));
        loading.appendChild(eroEl('span', { text: 'Checking funscript status…' }));
        container.appendChild(loading);

        let status = null;
        try {
            status = await eroRunAndPoll(
                'EroScripts Status', { scene_id: String(sceneId) }, 'status'
            );
        } catch (e) {
            log(`EroScripts status probe failed: ${e.message}`, 'warn');
        }
        container.textContent = '';

        const state = (status && status.state) || 'empty';
        switch (state) {
            case 'matched':
                eroRenderMatchedState(container, sceneId, status);
                break;
            case 'orphan_local':
                eroRenderOrphanLocalState(container, sceneId, status);
                break;
            case 'orphan_metadata':
                eroRenderOrphanMetadataState(container, sceneId, status);
                break;
            default:
                eroRenderEmptyState(container, sceneId);
        }
    }

    /**
     * Force the Scripts tab to re-fetch status next time it's shown, and
     * if it's currently active, re-render now.
     *
     * Called after a successful download so the user sees the matched
     * state without having to click off and back onto the tab.
     */
    function eroInvalidateScriptsTab(sceneId) {
        if (sidebarTabState && sidebarTabState.contentLoaded) {
            sidebarTabState.contentLoaded.scripts = false;
        }
        const pane = document.getElementById('scene-copilot-scripts-panel');
        if (pane && pane.classList.contains('active')) {
            renderSidebarScriptsContent(pane, sceneId);
        }
    }

    function eroRenderEmptyState(container, sceneId) {
        const wrap = eroEl('div', { cls: 'stash-copilot-sidebar-scripts' });
        const header = eroEl('div', { cls: 'stash-copilot-sidebar-header' });
        header.appendChild(eroEl('span', { cls: 'stash-copilot-sidebar-title', text: 'Funscript' }));
        wrap.appendChild(header);

        const intro = eroEl('div', { cls: 'stash-copilot-sidebar-intro' });
        intro.appendChild(eroEl('div', { cls: 'stash-copilot-sidebar-intro-icon', text: '⚡' }));
        intro.appendChild(eroEl('h3', { cls: 'stash-copilot-sidebar-intro-title', text: 'EroScripts' }));
        intro.appendChild(eroEl('p', {
            cls: 'stash-copilot-sidebar-intro-description',
            text: 'Search discuss.eroscripts.com for a funscript matching this scene. Top results from the Free Scripts category appear first; pick one to download.'
        }));
        const btn = eroEl('button', { cls: 'stash-copilot-sidebar-eros-btn', text: 'Find on EroScripts' });
        btn.addEventListener('click', () => openEroScriptsModal(sceneId));
        intro.appendChild(btn);
        wrap.appendChild(intro);
        container.appendChild(wrap);
    }

    /**
     * Matched state: sidecar JSON + funscript on disk. Show the rich card.
     */
    function eroRenderMatchedState(container, sceneId, status) {
        const sc = status.sidecar || {};
        const wrap = eroEl('div', { cls: 'stash-copilot-sidebar-scripts matched' });

        const header = eroEl('div', { cls: 'stash-copilot-sidebar-header' });
        header.appendChild(eroEl('span', { cls: 'stash-copilot-sidebar-title', text: 'Funscript' }));
        wrap.appendChild(header);

        const card = eroEl('div', { cls: 'stash-copilot-eros-matched-card' });

        // Title row.
        if (sc.eroscripts_thread_title) {
            card.appendChild(eroEl('div', {
                cls: 'stash-copilot-eros-matched-title',
                text: sc.eroscripts_thread_title
            }));
        }

        // Creator + likes line.
        const meta = eroEl('div', { cls: 'stash-copilot-eros-matched-meta' });
        if (sc.eroscripts_creator_avatar_url) {
            const av = eroEl('img', { cls: 'stash-copilot-eros-card-avatar',
                                       attrs: { loading: 'lazy', alt: '' } });
            av.src = sc.eroscripts_creator_avatar_url;
            meta.appendChild(av);
        }
        if (sc.eroscripts_creator_username) {
            meta.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-creator',
                                              text: '@' + sc.eroscripts_creator_username }));
        }
        if (typeof sc.eroscripts_like_count === 'number' && sc.eroscripts_like_count > 0) {
            meta.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-likes',
                                              text: '❤ ' + sc.eroscripts_like_count }));
        }
        if (meta.childNodes.length) card.appendChild(meta);

        // Tag chips (up to 6).
        const tags = Array.isArray(sc.eroscripts_tags) ? sc.eroscripts_tags : [];
        if (tags.length) {
            const tagWrap = eroEl('div', { cls: 'stash-copilot-eros-card-tags' });
            for (const t of tags.slice(0, 6)) {
                tagWrap.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-tag', text: t }));
            }
            card.appendChild(tagWrap);
        }

        // File-detail row.
        const fileRow = eroEl('div', { cls: 'stash-copilot-eros-matched-file' });
        if (sc.funscript_filename) {
            fileRow.appendChild(eroEl('span', {
                cls: 'stash-copilot-eros-matched-filename',
                text: sc.funscript_filename
            }));
        } else if (status.funscript_filename) {
            fileRow.appendChild(eroEl('span', {
                cls: 'stash-copilot-eros-matched-filename',
                text: status.funscript_filename
            }));
        }
        if (sc.downloaded_at) {
            fileRow.appendChild(eroEl('span', {
                cls: 'stash-copilot-eros-matched-date',
                text: 'Downloaded ' + eroFormatRelativeDate(sc.downloaded_at)
            }));
        }
        if (fileRow.childNodes.length) card.appendChild(fileRow);

        // Actions: View thread + Re-search.
        const actions = eroEl('div', { cls: 'stash-copilot-eros-matched-actions' });
        if (sc.eroscripts_thread_url) {
            const link = eroEl('a', {
                cls: 'stash-copilot-eros-matched-link',
                text: '↗ View on EroScripts',
                attrs: { target: '_blank', rel: 'noopener' }
            });
            link.href = sc.eroscripts_thread_url;
            actions.appendChild(link);
        }
        const research = eroEl('button', {
            cls: 'stash-copilot-eros-matched-research',
            text: 'Re-search', attrs: { type: 'button' }
        });
        research.addEventListener('click', () => openEroScriptsModal(sceneId));
        actions.appendChild(research);
        card.appendChild(actions);

        wrap.appendChild(card);
        container.appendChild(wrap);
    }

    /**
     * Funscript exists locally but we have no sidecar — likely the user got
     * it from somewhere other than this plugin. Nudge to attach metadata.
     */
    function eroRenderOrphanLocalState(container, sceneId, status) {
        const wrap = eroEl('div', { cls: 'stash-copilot-sidebar-scripts orphan-local' });
        const header = eroEl('div', { cls: 'stash-copilot-sidebar-header' });
        header.appendChild(eroEl('span', { cls: 'stash-copilot-sidebar-title', text: 'Funscript' }));
        wrap.appendChild(header);

        const intro = eroEl('div', { cls: 'stash-copilot-sidebar-intro' });
        intro.appendChild(eroEl('div', { cls: 'stash-copilot-sidebar-intro-icon', text: '⚡' }));
        intro.appendChild(eroEl('h3', { cls: 'stash-copilot-sidebar-intro-title', text: 'Local funscript present' }));
        const fnameLine = eroEl('p', { cls: 'stash-copilot-sidebar-intro-description' });
        fnameLine.appendChild(document.createTextNode('A funscript is already saved next to this scene'));
        if (status.funscript_filename) {
            fnameLine.appendChild(document.createTextNode(': '));
            fnameLine.appendChild(eroEl('code', { text: status.funscript_filename }));
        }
        fnameLine.appendChild(document.createTextNode('. You can search EroScripts to attach metadata or replace it.'));
        intro.appendChild(fnameLine);

        const btn = eroEl('button', { cls: 'stash-copilot-sidebar-eros-btn', text: 'Search EroScripts' });
        btn.addEventListener('click', () => openEroScriptsModal(sceneId));
        intro.appendChild(btn);
        wrap.appendChild(intro);
        container.appendChild(wrap);
    }

    /**
     * Sidecar exists but the funscript is gone — user deleted it. Offer to
     * re-download by re-opening the modal pre-targeted at the same thread.
     */
    function eroRenderOrphanMetadataState(container, sceneId, status) {
        const sc = status.sidecar || {};
        const wrap = eroEl('div', { cls: 'stash-copilot-sidebar-scripts orphan-metadata' });
        const header = eroEl('div', { cls: 'stash-copilot-sidebar-header' });
        header.appendChild(eroEl('span', { cls: 'stash-copilot-sidebar-title', text: 'Funscript' }));
        wrap.appendChild(header);

        const intro = eroEl('div', { cls: 'stash-copilot-sidebar-intro' });
        intro.appendChild(eroEl('div', { cls: 'stash-copilot-sidebar-intro-icon', text: '⚠' }));
        intro.appendChild(eroEl('h3', { cls: 'stash-copilot-sidebar-intro-title', text: 'Funscript file is missing' }));
        const desc = eroEl('p', { cls: 'stash-copilot-sidebar-intro-description' });
        desc.appendChild(document.createTextNode('We have eroscripts metadata for this scene'));
        if (sc.eroscripts_thread_title) {
            desc.appendChild(document.createTextNode(' (thread: '));
            desc.appendChild(eroEl('em', { text: sc.eroscripts_thread_title }));
            desc.appendChild(document.createTextNode(')'));
        }
        desc.appendChild(document.createTextNode(', but the .funscript file is no longer next to the video.'));
        intro.appendChild(desc);

        const btn = eroEl('button', { cls: 'stash-copilot-sidebar-eros-btn', text: 'Re-download or search' });
        btn.addEventListener('click', () => openEroScriptsModal(sceneId));
        intro.appendChild(btn);
        wrap.appendChild(intro);
        container.appendChild(wrap);
    }

    /**
     * Format an ISO 8601 timestamp as a coarse-grained relative description
     * suitable for "Downloaded X" text. Falls back to the raw date if the
     * value is unparseable.
     */
    function eroFormatRelativeDate(iso) {
        const t = Date.parse(iso);
        if (isNaN(t)) return iso;
        const diff = Date.now() - t;
        const sec = Math.floor(diff / 1000);
        if (sec < 60) return 'just now';
        const min = Math.floor(sec / 60);
        if (min < 60) return min + ' min ago';
        const hr = Math.floor(min / 60);
        if (hr < 24) return hr + ' hour' + (hr === 1 ? '' : 's') + ' ago';
        const day = Math.floor(hr / 24);
        if (day < 30) return day + ' day' + (day === 1 ? '' : 's') + ' ago';
        const mo = Math.floor(day / 30);
        if (mo < 12) return mo + ' month' + (mo === 1 ? '' : 's') + ' ago';
        const yr = Math.floor(mo / 12);
        return yr + ' year' + (yr === 1 ? '' : 's') + ' ago';
    }

    async function openEroScriptsModal(sceneId) {
        eroState.sceneId = sceneId;
        if (eroState.modal) eroDestroyModal();
        eroState.modal = eroBuildModal();
        document.body.appendChild(eroState.modal);

        eroShowLoading(eroState.modal, 'Checking eroscripts auth…');
        try {
            const auth = await eroRunAndPoll('EroScripts Validate Auth', { action: 'check' }, 'auth');
            if (auth.valid && auth.username) {
                eroShowSearch(eroState.modal, sceneId, auth.username);
            } else {
                eroShowAuthSetup(eroState.modal, auth.error || null);
            }
        } catch (e) {
            eroShowError(eroState.modal, 'Could not check auth state: ' + e.message);
        }
    }

    function eroBuildModal() {
        const overlay = eroEl('div', { cls: 'stash-copilot-eros-overlay' });
        const modal = eroEl('div', { cls: 'stash-copilot-eros-modal',
                                      attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Find Funscript on EroScripts' } });
        const header = eroEl('div', { cls: 'stash-copilot-eros-header' });
        header.appendChild(eroEl('span', { cls: 'stash-copilot-eros-title', text: '⚡ Find Funscript on EroScripts' }));
        const headerActions = eroEl('div', { cls: 'stash-copilot-eros-header-actions' });
        const configBtn = eroEl('button', { cls: 'stash-copilot-eros-config-btn',
                                            text: '⚙', title: 'Re-authenticate' });
        configBtn.addEventListener('click', () => eroShowAuthSetup(overlay, null));
        const closeBtn = eroEl('button', { cls: 'stash-copilot-eros-close-btn',
                                           text: '✕', title: 'Close (Esc)',
                                           attrs: { 'aria-label': 'Close' } });
        closeBtn.addEventListener('click', eroDestroyModal);
        headerActions.appendChild(configBtn);
        headerActions.appendChild(closeBtn);
        header.appendChild(headerActions);

        const body = eroEl('div', { cls: 'stash-copilot-eros-body' });
        const footer = eroEl('div', { cls: 'stash-copilot-eros-footer' });
        footer.style.display = 'none';

        modal.appendChild(header);
        modal.appendChild(body);
        modal.appendChild(footer);
        overlay.appendChild(modal);

        // Click outside the modal closes it.
        overlay.addEventListener('click', e => {
            if (e.target === overlay) eroDestroyModal();
        });
        // Esc closes it.
        const escHandler = e => { if (e.key === 'Escape') eroDestroyModal(); };
        document.addEventListener('keydown', escHandler);
        overlay._escHandler = escHandler;

        return overlay;
    }

    function eroDestroyModal() {
        if (!eroState.modal) return;
        if (eroState.pollAbort) {
            eroState.pollAbort.abort();
            eroState.pollAbort = null;
        }
        if (eroState.modal._escHandler) {
            document.removeEventListener('keydown', eroState.modal._escHandler);
        }
        eroState.modal.remove();
        eroState.modal = null;
    }

    function eroBody(modal) { return modal.querySelector('.stash-copilot-eros-body'); }

    function eroShowLoading(modal, label) {
        const body = eroBody(modal);
        body.textContent = '';
        const wrap = eroEl('div', { cls: 'stash-copilot-eros-loading' });
        wrap.appendChild(eroEl('div', { cls: 'stash-copilot-spinner' }));
        wrap.appendChild(eroEl('span', { cls: 'stash-copilot-eros-status', text: label }));
        body.appendChild(wrap);
    }

    function eroShowError(modal, message) {
        const body = eroBody(modal);
        body.textContent = '';
        const wrap = eroEl('div', { cls: 'stash-copilot-eros-error' });
        wrap.appendChild(eroEl('p', { text: message }));
        body.appendChild(wrap);
    }

    function eroShowAuthSetup(modal, priorError) {
        const body = eroBody(modal);
        body.textContent = '';
        const wrap = eroEl('div', { cls: 'stash-copilot-eros-auth' });
        wrap.appendChild(eroEl('h3', { text: 'Connect to EroScripts' }));

        const ol = eroEl('ol', { cls: 'stash-copilot-eros-auth-steps' });
        const li1 = eroEl('li');
        li1.appendChild(document.createTextNode('Open '));
        const link = eroEl('a', { text: 'discuss.eroscripts.com',
                                  attrs: { href: 'https://discuss.eroscripts.com',
                                           target: '_blank', rel: 'noopener' } });
        li1.appendChild(link);
        li1.appendChild(document.createTextNode(' and sign in.'));
        ol.appendChild(li1);
        ol.appendChild(eroEl('li', { text: 'Open browser DevTools → Application → Cookies → discuss.eroscripts.com.' }));
        const li3 = eroEl('li');
        li3.appendChild(document.createTextNode('Copy the value of the cookie named '));
        li3.appendChild(eroEl('code', { text: '_t' }));
        li3.appendChild(document.createTextNode(' and paste below.'));
        ol.appendChild(li3);
        wrap.appendChild(ol);

        if (priorError) {
            wrap.appendChild(eroEl('div', { cls: 'stash-copilot-eros-auth-error', text: priorError }));
        }

        const ta = eroEl('textarea', {
            cls: 'stash-copilot-eros-auth-cookie',
            attrs: { rows: '4', placeholder: 'Paste your _t cookie value here...',
                     spellcheck: 'false', autocomplete: 'off' }
        });
        wrap.appendChild(ta);

        const actions = eroEl('div', { cls: 'stash-copilot-eros-auth-actions' });
        const disconnectBtn = eroEl('button', { cls: 'stash-copilot-eros-auth-disconnect',
                                                text: 'Disconnect',
                                                attrs: { type: 'button' } });
        const saveBtn = eroEl('button', { cls: 'stash-copilot-eros-auth-save',
                                          text: 'Save & Validate',
                                          attrs: { type: 'button' } });
        actions.appendChild(disconnectBtn);
        actions.appendChild(saveBtn);
        wrap.appendChild(actions);

        const statusEl = eroEl('div', { cls: 'stash-copilot-eros-auth-status' });
        wrap.appendChild(statusEl);
        body.appendChild(wrap);

        async function doSave() {
            const cookie = ta.value.trim();
            if (!cookie) {
                statusEl.textContent = 'Paste a cookie first.';
                statusEl.className = 'stash-copilot-eros-auth-status error';
                return;
            }
            saveBtn.disabled = true;
            disconnectBtn.disabled = true;
            statusEl.textContent = 'Validating with eroscripts.com…';
            statusEl.className = 'stash-copilot-eros-auth-status loading';
            try {
                const result = await eroRunAndPoll(
                    'EroScripts Validate Auth',
                    { action: 'validate', cookie: cookie },
                    'auth'
                );
                if (result.valid && result.username) {
                    statusEl.textContent = 'Connected as @' + result.username + '. Loading search…';
                    statusEl.className = 'stash-copilot-eros-auth-status success';
                    setTimeout(() => eroShowSearch(modal, eroState.sceneId, result.username), 400);
                } else {
                    statusEl.textContent = result.error || 'Validation failed.';
                    statusEl.className = 'stash-copilot-eros-auth-status error';
                    saveBtn.disabled = false;
                    disconnectBtn.disabled = false;
                }
            } catch (e) {
                statusEl.textContent = 'Error: ' + e.message;
                statusEl.className = 'stash-copilot-eros-auth-status error';
                saveBtn.disabled = false;
                disconnectBtn.disabled = false;
            }
        }

        async function doDisconnect() {
            disconnectBtn.disabled = true;
            saveBtn.disabled = true;
            statusEl.textContent = 'Clearing stored cookie…';
            try {
                await eroRunAndPoll('EroScripts Validate Auth', { action: 'clear' }, 'auth');
                statusEl.textContent = 'Disconnected.';
                statusEl.className = 'stash-copilot-eros-auth-status';
                ta.value = '';
            } catch (e) {
                statusEl.textContent = 'Error: ' + e.message;
                statusEl.className = 'stash-copilot-eros-auth-status error';
            } finally {
                saveBtn.disabled = false;
                disconnectBtn.disabled = false;
            }
        }

        saveBtn.addEventListener('click', doSave);
        disconnectBtn.addEventListener('click', doDisconnect);
        ta.addEventListener('keydown', e => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') doSave();
        });
        setTimeout(() => ta.focus(), 50);
    }

    function eroShowSearch(modal, sceneId, username) {
        const body = eroBody(modal);
        body.textContent = '';
        const wrap = eroEl('div', { cls: 'stash-copilot-eros-search' });

        const bar = eroEl('div', { cls: 'stash-copilot-eros-search-bar' });
        const input = eroEl('input', {
            cls: 'stash-copilot-eros-search-input',
            attrs: { type: 'text', placeholder: 'Search eroscripts (press Enter)…',
                     spellcheck: 'false', autocomplete: 'off' }
        });
        const searchBtn = eroEl('button', { cls: 'stash-copilot-eros-search-btn',
                                            text: 'Search', attrs: { type: 'button' } });
        bar.appendChild(input);
        bar.appendChild(searchBtn);
        wrap.appendChild(bar);

        const meta = eroEl('div', { cls: 'stash-copilot-eros-search-meta' });
        meta.appendChild(eroEl('span', { cls: 'stash-copilot-eros-username',
                                          text: 'Connected as @' + username }));
        const statusText = eroEl('span', { cls: 'stash-copilot-eros-status-text' });
        meta.appendChild(statusText);
        wrap.appendChild(meta);

        const resultsEl = eroEl('div', { cls: 'stash-copilot-eros-results' });
        wrap.appendChild(resultsEl);
        body.appendChild(wrap);

        async function doSearch() {
            const query = input.value.trim();
            searchBtn.disabled = true;
            statusText.textContent = query
                ? 'Searching for "' + query + '"…'
                : 'Searching for this scene…';
            statusText.className = 'stash-copilot-eros-status-text loading';
            resultsEl.textContent = '';

            try {
                const data = await eroRunAndPoll(
                    'EroScripts Search',
                    { scene_id: String(sceneId), query: query },
                    'search'
                );
                if (data.auth_required) {
                    eroShowAuthSetup(modal, data.error || 'Session expired.');
                    return;
                }
                if (data.rate_limited) {
                    statusText.textContent = data.error || 'Rate limited.';
                    statusText.className = 'stash-copilot-eros-status-text error';
                    return;
                }
                if (data.status === 'error') {
                    statusText.textContent = data.error || 'Search failed.';
                    statusText.className = 'stash-copilot-eros-status-text error';
                    return;
                }
                if (data.suggested_query && !input.value) {
                    input.value = data.suggested_query;
                }
                eroRenderResults(resultsEl, data.results || [], statusText, data.suggested_query || '');
            } catch (e) {
                statusText.textContent = 'Error: ' + e.message;
                statusText.className = 'stash-copilot-eros-status-text error';
            } finally {
                searchBtn.disabled = false;
            }
        }

        searchBtn.addEventListener('click', doSearch);
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                doSearch();
            }
        });

        // Initial scene-derived search.
        doSearch();
    }

    function eroRenderResults(container, results, statusText, fallbackQuery) {
        container.textContent = '';
        if (!results.length) {
            const empty = eroEl('div', { cls: 'stash-copilot-eros-empty' });
            const p1 = eroEl('p');
            p1.appendChild(document.createTextNode('No matches found'));
            if (fallbackQuery) {
                p1.appendChild(document.createTextNode(' for "'));
                p1.appendChild(eroEl('em', { text: fallbackQuery }));
                p1.appendChild(document.createTextNode('"'));
            }
            p1.appendChild(document.createTextNode('.'));
            empty.appendChild(p1);
            empty.appendChild(eroEl('p', {
                text: 'Try simpler terms — performer name + a unique word usually works best.'
            }));
            container.appendChild(empty);
            statusText.textContent = '';
            return;
        }
        statusText.textContent = results.length + ' result' + (results.length === 1 ? '' : 's');
        statusText.className = 'stash-copilot-eros-status-text';

        for (const r of results) {
            container.appendChild(eroBuildResultCard(r));
        }
    }

    /**
     * Two-phase download click flow:
     *   1. Call EroScripts Download with topic_id only — backend returns attachment list
     *   2. If 0 → surface external links / "no direct download"
     *      If 1 → call Download again with attachment_url, save the script
     *      If 2+ → show inline picker, on user pick → call Download with attachment_url
     */
    async function eroHandleDownloadClick(btn, resultMeta) {
        const sceneId = eroState.sceneId;
        if (!sceneId) return;

        eroSetCardBusy(btn, 'Fetching topic…');
        try {
            const list = await eroRunAndPoll(
                'EroScripts Download',
                { scene_id: String(sceneId), topic_id: String(resultMeta.topic_id) },
                'download'
            );
            if (list.auth_required) {
                eroShowAuthSetup(eroState.modal, list.error || 'Session expired.');
                return;
            }
            if (list.status === 'error') {
                eroSetCardError(btn, list.error || 'Failed to fetch topic.');
                return;
            }
            const atts = list.attachments || [];
            if (atts.length === 0) {
                const ext = list.external_links || [];
                const msg = ext.length
                    ? `No direct funscript on this thread — links to external host(s). Open the thread to download.`
                    : `No funscript attachment on this thread.`;
                eroSetCardError(btn, msg);
                return;
            }
            if (atts.length === 1) {
                await eroDownloadOne(btn, resultMeta, atts[0]);
                return;
            }
            eroShowAttachmentPicker(btn, resultMeta, atts);
        } catch (e) {
            eroSetCardError(btn, 'Error: ' + e.message);
        }
    }

    async function eroDownloadOne(btn, resultMeta, attachment) {
        eroSetCardBusy(btn, 'Downloading ' + attachment.filename + '…');
        eroShowFooter('Downloading ' + attachment.filename + '…');
        try {
            const data = await eroRunAndPoll(
                'EroScripts Download',
                {
                    scene_id: String(eroState.sceneId),
                    topic_id: String(resultMeta.topic_id),
                    attachment_url: attachment.url,
                    hint_creator_username: resultMeta.creator_username || '',
                    hint_creator_avatar_url: resultMeta.avatar_url || '',
                    hint_like_count: String(resultMeta.like_count || 0),
                    hint_tags: JSON.stringify(resultMeta.tags || []),
                    hint_created_at: resultMeta.created_at || '',
                },
                'download'
            );
            if (data.auth_required) {
                eroShowAuthSetup(eroState.modal, data.error || 'Session expired.');
                return;
            }
            if (data.status === 'error' || data.error) {
                eroSetCardError(btn, data.error || 'Download failed.');
                eroShowFooter(data.error || 'Download failed.', true);
                return;
            }
            if (data.was_duplicate) {
                eroSetCardSuccess(btn, 'Already have this');
                eroShowFooter('You already have this funscript — no changes.');
            } else if (data.suffix_applied) {
                eroSetCardSuccess(btn, 'Saved as ' + data.saved_filename);
                eroShowFooter('Saved as ' + data.saved_filename + ' (existing primary funscript kept).');
            } else {
                eroSetCardSuccess(btn, 'Saved ✓');
                eroShowFooter('Saved as ' + data.saved_filename + ' ✓');
            }
            // Refresh the Scripts sidebar tab so it transitions from empty
            // → matched state without the user having to click off/back on.
            eroInvalidateScriptsTab(eroState.sceneId);
        } catch (e) {
            eroSetCardError(btn, 'Error: ' + e.message);
            eroShowFooter('Error: ' + e.message, true);
        }
    }

    function eroShowAttachmentPicker(btn, resultMeta, attachments) {
        // Replace the button with a small dropdown of attachment names.
        const card = btn.closest('.stash-copilot-eros-card');
        if (!card) return;
        const actions = card.querySelector('.stash-copilot-eros-card-actions');
        if (!actions) return;
        const picker = eroEl('div', { cls: 'stash-copilot-eros-card-picker' });
        picker.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-picker-label',
                                            text: 'Pick attachment:' }));
        for (const a of attachments) {
            const item = eroEl('button', {
                cls: 'stash-copilot-eros-card-picker-item',
                attrs: { type: 'button' },
                title: a.filename,  // full name on hover; the visible label gets ellipsized
            });
            // Build label: filename node + optional " · 12 KB" size suffix.
            // We use child nodes (not innerHTML/title interpolation) so the
            // user-controlled filename never touches HTML parsing.
            const nameSpan = eroEl('span', { cls: 'stash-copilot-eros-picker-name',
                                              text: a.filename });
            item.appendChild(nameSpan);
            if (typeof a.size_bytes === 'number' && a.size_bytes > 0) {
                item.appendChild(eroEl('span', {
                    cls: 'stash-copilot-eros-picker-size',
                    text: ' · ' + eroFormatBytes(a.size_bytes),
                }));
            }
            item.addEventListener('click', () => {
                picker.remove();
                eroResetCard(btn);
                eroDownloadOne(btn, resultMeta, a);
            });
            picker.appendChild(item);
        }
        // Hide the original Download button while picker is shown.
        btn.style.display = 'none';
        actions.appendChild(picker);
    }

    /**
     * Format a byte count as a short human-readable string.
     * Funscripts are typically 5 KB to a few MB; we use binary thresholds
     * since that's what most file managers display.
     */
    function eroFormatBytes(n) {
        if (n < 1024) return n + ' B';
        if (n < 1024 * 1024) return (n / 1024).toFixed(n < 10 * 1024 ? 1 : 0) + ' KB';
        return (n / (1024 * 1024)).toFixed(n < 10 * 1024 * 1024 ? 1 : 0) + ' MB';
    }

    function eroSetCardBusy(btn, label) {
        btn.disabled = true;
        btn.dataset.origText = btn.dataset.origText || btn.textContent;
        btn.textContent = label;
        btn.classList.remove('error', 'success');
        btn.classList.add('busy');
    }
    function eroSetCardError(btn, label) {
        btn.disabled = false;
        btn.textContent = label;
        btn.classList.remove('busy', 'success');
        btn.classList.add('error');
    }
    function eroSetCardSuccess(btn, label) {
        btn.disabled = true;
        btn.textContent = label;
        btn.classList.remove('busy', 'error');
        btn.classList.add('success');
    }
    function eroResetCard(btn) {
        btn.disabled = false;
        if (btn.dataset.origText) btn.textContent = btn.dataset.origText;
        btn.classList.remove('busy', 'error', 'success');
        btn.style.display = '';
    }

    function eroShowFooter(msg, isError) {
        if (!eroState.modal) return;
        const footer = eroState.modal.querySelector('.stash-copilot-eros-footer');
        if (!footer) return;
        footer.textContent = msg;
        footer.classList.toggle('error', !!isError);
        footer.style.display = '';
    }

    function eroBuildResultCard(r) {
        const card = eroEl('div', { cls: 'stash-copilot-eros-card',
                                    attrs: { 'data-topic-id': String(r.topic_id) } });
        if (r.thumbnail_url) {
            const img = eroEl('img', { cls: 'stash-copilot-eros-card-thumb',
                                       attrs: { loading: 'lazy', alt: '' } });
            img.src = r.thumbnail_url;
            card.appendChild(img);
        } else {
            card.appendChild(eroEl('div', { cls: 'stash-copilot-eros-card-thumb-placeholder', text: '⚡' }));
        }

        const body = eroEl('div', { cls: 'stash-copilot-eros-card-body' });
        body.appendChild(eroEl('div', { cls: 'stash-copilot-eros-card-title',
                                        text: r.title || '(untitled)' }));

        const meta = eroEl('div', { cls: 'stash-copilot-eros-card-meta' });
        if (r.avatar_url) {
            const av = eroEl('img', { cls: 'stash-copilot-eros-card-avatar',
                                      attrs: { loading: 'lazy', alt: '' } });
            av.src = r.avatar_url;
            meta.appendChild(av);
        }
        if (r.creator_username) {
            meta.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-creator',
                                             text: '@' + r.creator_username }));
        }
        meta.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-likes',
                                         text: '❤ ' + (Number(r.like_count) || 0) }));
        body.appendChild(meta);

        const tagWrap = eroEl('div', { cls: 'stash-copilot-eros-card-tags' });
        for (const t of (r.tags || []).slice(0, 5)) {
            tagWrap.appendChild(eroEl('span', { cls: 'stash-copilot-eros-card-tag', text: t }));
        }
        body.appendChild(tagWrap);

        body.appendChild(eroEl('div', { cls: 'stash-copilot-eros-card-excerpt',
                                        text: (r.excerpt || '').slice(0, 220) }));

        const actions = eroEl('div', { cls: 'stash-copilot-eros-card-actions' });
        const threadLink = eroEl('a', { cls: 'stash-copilot-eros-card-thread-link',
                                        text: '↗ View thread',
                                        attrs: { target: '_blank', rel: 'noopener' } });
        threadLink.href = r.url;
        actions.appendChild(threadLink);
        const dl = eroEl('button', { cls: 'stash-copilot-eros-card-download',
                                     text: 'Download',
                                     attrs: { type: 'button',
                                              'data-topic-id': String(r.topic_id) } });
        dl.addEventListener('click', () => eroHandleDownloadClick(dl, r));
        actions.appendChild(dl);
        body.appendChild(actions);

        card.appendChild(body);
        return card;
    }

    // ===== End EroScripts =====

    // Expose public API
    window.stashCopilot = {
        runDescribePerformer: runDescribePerformer,
        // Add other public methods as needed
    };

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
