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

  test('Gaps tab: request_id-keyed get_scene_tag_gaps resolves via the stub', async ({ page }) => {
    const pageErrors = await collectPageErrors(page);

    // The Gaps tab auto-triggers get_scene_tag_gaps on open (keyed by request_id
    // -> tag_gaps_scene_<requestId>.json). The fixture sets has_data:false, so
    // the panel resolves to its empty state.
    await openSceneTab(page, 1, 'scene-copilot-gaps');

    const panel = page.locator('#scene-copilot-gaps-panel');
    await expect(panel).toBeVisible({ timeout: 10000 });

    await expect(panel.locator('.stash-copilot-sidebar-gaps-loading')).toBeHidden({ timeout: 20000 });
    await expect(panel.locator('.stash-copilot-sidebar-gaps-empty')).toBeVisible({ timeout: 5000 });

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

    expect(pageErrors, `uncaught JS errors:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });
});
