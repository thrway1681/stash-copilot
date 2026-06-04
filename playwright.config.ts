import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for the plugin's UI tests (issue #5 testing infra).
 *
 * The tests drive a real Stash instance with the plugin UI installed:
 *   - locally: `docker compose -f docker-compose.dev.yml up -d` + install the
 *     plugin (scripts/ci/install_plugin_ui.sh) + `reloadPlugins`, then
 *     `npx playwright test`.
 *   - in CI: the `ui` job in .github/workflows/ci.yml does the same headlessly.
 *
 * STASH_URL points at that instance (default matches the dev compose: host
 * 3000 -> container 9999). CI runs chromium-only for speed/determinism.
 */
const STASH_URL = process.env.STASH_URL || 'http://localhost:3000';

export default defineConfig({
  testDir: './tests/e2e',
  /* Artifacts (screenshots, video, traces) — uploaded as CI artifacts. */
  outputDir: './test-recordings',
  fullyParallel: true,
  /* Fail the build on CI if a test.only was left in the source. */
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  /* The plugin injects into a shared Stash instance; serialize on CI to avoid
     cross-test interference (one Stash, shared plugin asset files). */
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [['github'], ['list'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: STASH_URL,
    /* Always capture evidence — these tests are the cross-stack UI guard. */
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'on',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
