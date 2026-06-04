import { test, expect, type Page } from '@playwright/test';

/**
 * P0 smoke (issue #5 testing infra): prove the plugin UI loads in a real Stash
 * and injects its AI sidebar tabs with no uncaught JavaScript errors.
 *
 * This is the cheap, high-value regression guard for editing the ~14k-line
 * stash-copilot.js: a syntax/runtime error that breaks the plugin surfaces here
 * as a `pageerror` and/or missing tabs. It needs no LLM and no plugin backend —
 * just the UI installed + a scene to view (the CI `ui` job seeds both).
 *
 * The plugin injects six nav items (Analyze / Similar / Recs / Gaps / Tags /
 * Scripts) as <li class="nav-item stash-copilot-tab-nav"> into Stash's native
 * `.scene-tabs > .nav-tabs` (see injectSceneTabs in stash-copilot.js).
 */

const EXPECTED_TAB_COUNT = 6;

/** Attach an uncaught-exception collector to a page. */
function collectPageErrors(page: Page): Error[] {
  const errors: Error[] = [];
  page.on('pageerror', (err) => errors.push(err));
  return errors;
}

/**
 * After first-run setup Stash shows a one-time version/changelog dialog that
 * overlays the page (`.ModalComponent.modal.show`) and intercepts clicks. Each
 * Playwright test gets a fresh context, so it reappears per test — dismiss it
 * before interacting. No-op if absent.
 */
async function dismissStashDialogs(page: Page): Promise<void> {
  const closeBtn = page.locator('[role="dialog"].modal.show button:has-text("Close")');
  for (let i = 0; i < 3; i++) {
    if (!(await closeBtn.first().isVisible().catch(() => false))) return;
    await closeBtn.first().click().catch(() => undefined);
    await page.waitForTimeout(300);
  }
}

test.describe('plugin UI smoke', () => {
  test('library/home page loads without plugin errors', async ({ page }) => {
    const pageErrors = collectPageErrors(page);

    await page.goto('/', { waitUntil: 'domcontentloaded' });
    // Stash is a React SPA; wait for its app shell (the top navbar) to mount,
    // which means the injected plugin JS has had a chance to run too.
    await page.locator('.navbar, #root .main, .main-tabs').first().waitFor({ timeout: 30000 });

    expect(pageErrors, `uncaught JS errors on home page:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('scene page injects the six AI sidebar tabs without errors', async ({ page }) => {
    const pageErrors = collectPageErrors(page);

    // The CI `ui` job scans synthetic media, so scene 1 exists.
    await page.goto('/scenes/1', { waitUntil: 'domcontentloaded' });

    // Stash's own scene tab bar must render first; the plugin waits on it too.
    await page.locator('.scene-tabs').first().waitFor({ timeout: 30000 });

    // The plugin's injected nav items.
    const tabNav = page.locator('.stash-copilot-tab-nav');
    await tabNav.first().waitFor({ timeout: 20000 });
    await expect(tabNav).toHaveCount(EXPECTED_TAB_COUNT);

    // Each AI tab key should be present (catches a partial/garbled injection).
    for (const key of [
      'scene-copilot-analyze',
      'scene-copilot-similar',
      'scene-copilot-recs',
      'scene-copilot-gaps',
      'scene-copilot-tags',
      'scene-copilot-scripts',
    ]) {
      await expect(page.locator(`a[data-rb-event-key="${key}"]`)).toHaveCount(1);
    }

    expect(pageErrors, `uncaught JS errors on scene page:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });

  test('clicking an AI tab activates its panel', async ({ page }) => {
    const pageErrors = collectPageErrors(page);

    await page.goto('/scenes/1', { waitUntil: 'domcontentloaded' });
    await page.locator('.scene-tabs').first().waitFor({ timeout: 30000 });
    await page.locator('.stash-copilot-tab-nav').first().waitFor({ timeout: 20000 });

    // Clear Stash's one-time changelog dialog so it can't intercept the click.
    await dismissStashDialogs(page);

    // The Analyze tab is purely client-side (no backend task) until the user
    // clicks "Analyze", so opening it must not require the plugin backend.
    await page.locator('a[data-rb-event-key="scene-copilot-analyze"]').click();
    await expect(page.locator('#scene-copilot-analyze-panel')).toBeVisible({ timeout: 10000 });

    expect(pageErrors, `uncaught JS errors after tab click:\n${pageErrors.map((e) => e.stack || e.message).join('\n')}`).toEqual([]);
  });
});
