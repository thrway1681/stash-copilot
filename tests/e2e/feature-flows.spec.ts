import { test, expect, type Page } from '@playwright/test';

/**
 * P1 feature flows (issue #5 testing infra): drive a feature end-to-end through
 * the real dispatchTask path — UI trigger -> Stash runs the stub backend -> the
 * stub writes a fixture result file -> the UI polls it -> the panel resolves.
 *
 * The stub (scripts/ci/stub-run-plugin.sh) stands in for the ML/LLM backend that
 * can't run in the Alpine Stash container; fixtures live in scripts/ci/fixtures/.
 * These prove the dispatchTask keying paths the commit-3 migrations touch.
 * Per-feature coverage grows as each feature is migrated (a flow test lands in
 * the same PR); this file seeds the representative cases.
 *
 * NOTE: asserts the panel leaves its loading state (the plumbing resolved), not
 * exact card markup — keeps the guard robust against render-detail churn.
 */

async function collectPageErrors(page: Page): Promise<Error[]> {
  const errors: Error[] = [];
  page.on('pageerror', (err) => errors.push(err));
  return errors;
}

/** Dismiss Stash's one-time post-setup version/changelog dialog if present. */
async function dismissStashDialogs(page: Page): Promise<void> {
  const closeBtn = page.locator('[role="dialog"].modal.show button:has-text("Close")');
  for (let i = 0; i < 3; i++) {
    if (!(await closeBtn.first().isVisible().catch(() => false))) return;
    await closeBtn.first().click().catch(() => undefined);
    await page.waitForTimeout(300);
  }
}

/** Open a scene page and the named AI sidebar tab. */
async function openSceneTab(page: Page, sceneId: number, tabKey: string): Promise<void> {
  await page.goto(`/scenes/${sceneId}`, { waitUntil: 'domcontentloaded' });
  await page.locator('.scene-tabs').first().waitFor({ timeout: 30000 });
  await page.locator('.stash-copilot-tab-nav').first().waitFor({ timeout: 20000 });
  await dismissStashDialogs(page);
  await page.locator(`a[data-rb-event-key="${tabKey}"]`).click();
}

test.describe('feature flows (stub backend)', () => {
  test('Similar tab: scene_id-keyed find_similar resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // Similar auto-triggers find_similar on open (keyed by scene_id ->
    // similar_results_<sceneId>.json, which the stub writes).
    await openSceneTab(page, 2, 'scene-copilot-similar');

    const panel = page.locator('#scene-copilot-similar-panel');
    await expect(panel).toBeVisible({ timeout: 10000 });

    // The "Finding similar scenes..." loading state must clear once the poll
    // resolves against the stub's fixture (empty results -> a terminal state).
    await expect(panel.locator('.stash-copilot-sidebar-loading')).toBeHidden({ timeout: 20000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Gaps tab: get_scene_tag_gaps renders coverage + preview_tag_impact resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The Gaps tab auto-triggers get_scene_tag_gaps on open (request_id-keyed ->
    // tag_gaps_scene_<rid>.json). The fixture sets has_data:true, so the panel
    // renders the coverage detail — which includes the Test Tag Impact UI.
    await openSceneTab(page, 1, 'scene-copilot-gaps');

    const panel = page.locator('#scene-copilot-gaps-panel');
    await expect(panel).toBeVisible({ timeout: 10000 });

    await expect(panel.locator('.stash-copilot-sidebar-gaps-loading')).toBeHidden({ timeout: 20000 });
    await expect(panel.locator('.stash-copilot-sidebar-gaps-coverage')).toBeVisible({ timeout: 5000 });

    // Exercise preview_tag_impact (request_id-keyed -> tag_preview_<rid>.json):
    // type a tag and click Preview; the fixture renders the coverage impact.
    await panel.locator('.stash-copilot-sidebar-gaps-tag-input').fill('outdoor');
    await panel.locator('.stash-copilot-sidebar-gaps-preview-btn').click();

    const previewResult = panel.locator('.stash-copilot-sidebar-gaps-preview-result');
    await expect(previewResult).toBeVisible({ timeout: 15000 });
    await expect(previewResult.locator('.stash-copilot-sidebar-gaps-preview-tag')).toContainText('outdoor');

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Tags tab: request_id-keyed get_tag_suggestions resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The Tags tab shows an intro with a "Suggest Tags" button (no auto-load);
    // clicking it runs get_tag_suggestions (request_id-keyed). The fixture
    // returns status:complete with empty suggestions -> content empty state.
    await openSceneTab(page, 1, 'scene-copilot-tags');

    const panel = page.locator('#scene-copilot-tags-panel');
    await expect(panel).toBeVisible({ timeout: 10000 });

    await panel.locator('.stash-copilot-suggest-tags-btn').click();

    await expect(panel.locator('.stash-copilot-tags-loading')).toBeHidden({ timeout: 20000 });
    await expect(panel.locator('.stash-copilot-tags-content')).toBeVisible({ timeout: 5000 });

    // Exercise a fire-and-forget dispatchTask call: the "Clear Dismissed" (↻)
    // button runs clearDismissedTags (resultKey null, no poll) and flips to ✓.
    const clearBtn = panel.locator('.stash-copilot-clear-dismissed-btn');
    await clearBtn.click();
    await expect(clearBtn).toHaveText('✓', { timeout: 10000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Performer Similar tab: request_id-keyed find_similar_performers resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // Seeded performer id 1 exists; the plugin injects a Similar tab on the
    // performer page. Clicking it runs find_similar_performers (request_id-keyed).
    // The fixture returns empty results -> the "No similar performers" state.
    await page.goto('/performers/1', { waitUntil: 'domcontentloaded' });
    await page.locator('.stash-copilot-performer-tab-nav').first().waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);

    await page.locator('a[data-rb-event-key="performer-copilot-similar"]').click();

    await expect(page.locator('.stash-copilot-performer-loading')).toBeHidden({ timeout: 20000 });
    await expect(page.locator('.stash-copilot-performer-empty')).toBeVisible({ timeout: 5000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('AI Insights modal: request_id-keyed detect_tag_gaps resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The plugin injects an "AI Insights" navbar button that opens a modal.
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.locator('#stash-copilot-nav-btn').waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);
    await page.locator('#stash-copilot-nav-btn').click();

    // Switch to the Tag Gaps tab and run detection (request_id-keyed -> tag_gaps_*).
    await page.locator('.stash-copilot-insights-tab[data-tab="tag_gaps"]').click();
    const detectBtn = page.locator('.stash-copilot-tag-gaps-detect-btn');
    await detectBtn.click();

    // The fixture (status:complete, empty scenes) resolves to the summary + the
    // button flipping to "Re-detect".
    await expect(detectBtn).toHaveText(/Re-detect/, { timeout: 20000 });
    await expect(page.locator('.stash-copilot-tag-gaps-summary')).toBeVisible({ timeout: 5000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('AI Insights modal: request_id-keyed build_taste_map resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.locator('#stash-copilot-nav-btn').waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);
    await page.locator('#stash-copilot-nav-btn').click();

    await page.locator('.stash-copilot-insights-tab[data-tab="taste_map"]').click();
    const buildBtn = page.locator('.stash-copilot-taste-map-build-btn');
    await buildBtn.click();

    // The fixture (1 cluster, 1 scene) resolves -> renderTasteMap flips the
    // button to "Rebuild" and renders the Plotly chart + cluster sidebar.
    await expect(buildBtn).toHaveText(/Rebuild/, { timeout: 20000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Similar tab "Search by Frame": request_id-keyed find_similar_by_frame resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    await openSceneTab(page, 1, 'scene-copilot-similar');
    const panel = page.locator('#scene-copilot-similar-panel');

    // The Similar tab auto-runs find_similar on open; wait for it to settle so it
    // doesn't race/overwrite the frame-search render.
    await expect(panel.locator('.stash-copilot-sidebar-loading')).toBeHidden({ timeout: 20000 });

    // startFrameSearch reads the player's currentTime and requires it > 0 (or
    // readyState >= 1). Headless doesn't auto-load the clip into a seekable
    // video.vjs-tech, so inject a fake one — the video is only the timestamp
    // source; dispatchTask is what's under test.
    await page.evaluate(() => {
      let v = document.querySelector('video.vjs-tech');
      if (!v) {
        v = document.createElement('video');
        v.className = 'vjs-tech';
        document.body.appendChild(v);
      }
      Object.defineProperty(v, 'currentTime', { value: 1.5, configurable: true });
      Object.defineProperty(v, 'readyState', { value: 4, configurable: true });
    });

    await panel.locator('.stash-copilot-frame-search-btn').click();

    // Frame search resolves to its own render (empty fixture -> "No similar frames
    // found" + a Back button, which is unique to the frame-search view).
    await expect(panel.locator('.stash-copilot-back-to-similar')).toBeVisible({ timeout: 20000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Similar tab "Search by Frame": exiting mid-search drops the late result (cancellation guard)', async ({ page }) => {
    // Regression for #95: the migrated frame search awaits dispatchTask (an
    // uncancellable poll loop, unlike the old clearInterval-able setInterval).
    // Clicking "Back to Similar" before the result lands must NOT let the late
    // result re-render the frame-search view over the restored Similar view.
    const pageErrors = await collectPageErrors(page);

    await openSceneTab(page, 1, 'scene-copilot-similar');
    const panel = page.locator('#scene-copilot-similar-panel');

    // Let the auto find_similar settle so it isn't what we observe later.
    await expect(panel.locator('.stash-copilot-sidebar-loading')).toBeHidden({ timeout: 20000 });

    // Inject a seekable fake player (headless has none) — only a timestamp source.
    await page.evaluate(() => {
      let v = document.querySelector('video.vjs-tech');
      if (!v) {
        v = document.createElement('video');
        v.className = 'vjs-tech';
        document.body.appendChild(v);
      }
      Object.defineProperty(v, 'currentTime', { value: 1.5, configurable: true });
      Object.defineProperty(v, 'readyState', { value: 4, configurable: true });
    });

    // startFrameSearch renders the loading-state "Back to Similar" synchronously,
    // BEFORE its await — so it's clickable immediately. The stub delays
    // find_similar_by_frame (sleep 2), so this back-click reliably lands first.
    await panel.locator('.stash-copilot-frame-search-btn').click();
    await panel.locator('.stash-copilot-back-to-similar').click();

    // Exit restores the Similar controls right away.
    await expect(panel.locator('.stash-copilot-sidebar-subtabs')).toBeVisible({ timeout: 5000 });

    // Wait past the stub's delay so the now-stale dispatchTask result resolves.
    // The token guard must drop it: the frame-search empty state must NOT appear
    // and the Similar view must remain.
    await page.waitForTimeout(3500);

    await expect(
      panel.locator('.stash-copilot-sidebar-empty', { hasText: 'No similar frames' })
    ).toHaveCount(0);
    await expect(panel.locator('.stash-copilot-sidebar-subtabs')).toBeVisible();

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Search page: request_id-keyed get_embedding_models + search_by_text resolve via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The plugin renders its own search page at this route (onPageChange ->
    // renderSearchPage). On render it auto-runs get_embedding_models to populate
    // the model dropdown (request_id-keyed -> embedding_models_<rid>.json).
    await page.goto('/plugins/stash-copilot/search', { waitUntil: 'domcontentloaded' });
    await page.locator('.stash-copilot-search-page').waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);

    // get_embedding_models resolved via the stub: the dropdown leaves its
    // "Loading models..." placeholder and lists the fixture's model.
    await expect(page.locator('#stash-copilot-model-select')).toContainText('openclip:ViT-B-32', {
      timeout: 15000,
    });

    // Run a search (request_id-keyed -> search_results_<rid>.json). The fixture
    // returns empty results, so the panel resolves to its empty state.
    await page.locator('.stash-copilot-search-input').fill('test query');
    await page.locator('.stash-copilot-search-btn').click();

    await expect(page.locator('.stash-copilot-search-loading')).toBeHidden({ timeout: 20000 });
    await expect(page.locator('.stash-copilot-search-empty')).toBeVisible({ timeout: 5000 });

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('AI Insights modal: fixed-file generate_summary resolves via the stub', async ({ page }) => {
    // generate_summary is the FIXED-file keying path (last_summary.json, no
    // request_id in the name). The migration also fixes a real bug: the old call
    // site invoked a non-existent 'Generate Summary' task + an undefined
    // pollForSummary. The stub stamps a fresh generated_at (__NOW__) so the
    // freshness poll resolves.
    const pageErrors = await collectPageErrors(page);

    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.locator('#stash-copilot-nav-btn').waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);
    await page.locator('#stash-copilot-nav-btn').click();

    // Summary is the default tab; click it explicitly for robustness.
    await page.locator('.stash-copilot-insights-tab[data-tab="summary"]').click();

    const generateBtn = page.locator('.stash-copilot-generate-btn');
    await generateBtn.click();

    // The fresh summary renders and the button returns to its idle, enabled state.
    await expect(page.locator('.stash-copilot-summary-text')).toContainText('CI Library Summary', {
      timeout: 20000,
    });
    await expect(generateBtn).toHaveText('Generate Summary', { timeout: 5000 });
    await expect(generateBtn).toBeEnabled();

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('Tag dedup page: request_id-keyed find_duplicate_tags + merge_tags resolve via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The plugin renders its own tag-dedup page (onPageChange -> renderTagDedupPage)
    // and auto-runs find_duplicate_tags (request_id-keyed -> tag_dedup_<rid>.json).
    await page.goto('/plugins/stash-copilot/tag-dedup', { waitUntil: 'domcontentloaded' });
    await page.locator('.stash-copilot-dedup-body').waitFor({ timeout: 30000 });
    await dismissStashDialogs(page);

    // The stub's single candidate pair renders.
    await expect(page.locator('.stash-copilot-dedup-versus')).toBeVisible({ timeout: 20000 });
    await expect(page.locator('.stash-copilot-dedup-versus')).toContainText('ci-tag');

    // Keep the left tag -> merge_tags (request_id-keyed -> tag_merge_<rid>.json).
    // With one candidate, completing the merge advances to the summary screen.
    await page.locator('#dedup-keep-left').click();

    await expect(page.locator('.stash-copilot-dedup-summary')).toBeVisible({ timeout: 20000 });
    await expect(page.locator('.stash-copilot-dedup-summary')).toContainText('Deduplication Complete');

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });
});
