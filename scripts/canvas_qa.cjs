const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

const root = path.resolve(__dirname, '..');
const evidence = process.env.FLUX3_PE_CANVAS_EVIDENCE || path.join(root, 'artifacts', 'canvas-qa');
fs.mkdirSync(evidence, { recursive: true });
const mode = process.argv[2] || 'inspect';
const baseURL = process.env.COMFY_URL || 'http://127.0.0.1:8267';
const names = ['01_prompt_enhancement', '02_layout_enhancement', '03_pe_to_flux3', '04_reference_editing'];
const cleanPrompt = prompt => Object.fromEntries(Object.entries(prompt).map(([id, node]) => [id, { class_type: node.class_type, inputs: node.inputs }]));
const version = fs.readFileSync(path.join(root, 'flux3_pe/__init__.py'), 'utf8').match(/__version__\s*=\s*"([^"]+)"/)[1];
// Keep the runtime identity contract identical to flux3_pe/identity.py.
const coreFiles = ['__init__.py', 'nodes.py', 'flux3_pe/__init__.py', 'flux3_pe/engine.py', 'flux3_pe/protocol.py', 'flux3_pe/bbox.py', 'flux3_pe/identity.py', 'flux3_pe/resources/system_prompts.json', 'flux3_pe/wheel_backend.py'];
const identityHash = crypto.createHash('sha256');
for (const name of coreFiles.sort()) identityHash.update(name).update('\0').update(fs.readFileSync(path.join(root, name))).update('\0');
const sourceIdentity = identityHash.digest('hex');
const inputFiles = names.map(name => `examples/${name}.api.json`).concat(fs.readdirSync(path.join(root, 'examples/reference_images')).filter(name => name.endsWith('.png')).map(name => `examples/reference_images/${name}`));
const inputHashes = Object.fromEntries(inputFiles.sort().map(name => [name, crypto.createHash('sha256').update(fs.readFileSync(path.join(root, name))).digest('hex')]));
const qaScriptHash = crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(baseURL, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => Boolean(window.comfyAPI?.app?.app?.graph && window.LiteGraph?.registered_node_types?.Flux3PromptEnhance), null, { timeout: 60000 });
    await page.locator('#splash-loader').waitFor({ state: 'hidden', timeout: 60000 });
    const load = async (workflow, name) => {
      await page.evaluate(async ({ workflow, name }) => {
        const { app } = await import('/scripts/app.js');
        await app.loadGraphData(workflow, true, true, `${name}.json`);
      }, { workflow, name });
      await page.waitForTimeout(300);
    };
    const current = () => page.evaluate(async () => {
      const { app } = await import('/scripts/app.js');
      return await app.graphToPrompt();
    });
    if (mode === 'png-roundtrip') {
      const report = JSON.parse(fs.readFileSync(path.join(evidence, 'execute.json'), 'utf8'));
      const generation = report.runs.find(item => item.name === '03_pe_to_flux3');
      const image = generation.history.outputs['19'].images[0];
      const bytes = fs.readFileSync(path.join(root, 'artifacts/comfy-output', image.subfolder, image.filename));
      await page.evaluate(async base64 => {
        const { app } = await import('/scripts/app.js');
        const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
        await app.handleFile(new File([bytes], 'FLUX3_PE_tested_output.png', { type: 'image/png' }));
      }, bytes.toString('base64'));
      await page.waitForTimeout(500);
      assert.deepEqual(cleanPrompt((await current()).output), cleanPrompt(generation.history.prompt[2]), 'The generated PNG did not restore its actual canvas');
      assert.deepEqual(errors, []);
      fs.writeFileSync(path.join(evidence, 'png-roundtrip.json'), JSON.stringify({ image, prompt_id: generation.prompt_id, version, source_identity: sourceIdentity, qa_script_sha256: qaScriptHash, image_sha256: crypto.createHash('sha256').update(bytes).digest('hex'), frontend_imported: true, graph_matches_execution: true, errors }, null, 2));
      console.log('PASS generated PNG restores the complete executed canvas');
    }
    const save = async name => {
      await page.getByRole('button', { name: '图形模式，工作流操作', exact: true }).click();
      await page.getByText('另存为', { exact: true }).click();
      await page.locator('input[type="text"]').last().fill(`FLUX3_PE/${name}`);
      const responsePromise = page.waitForResponse(response => response.request().method() === 'POST' && response.url().includes('/userdata/'), { timeout: 15000 });
      await page.getByRole('button', { name: '确认', exact: true }).click();
      await page.waitForTimeout(350);
      const overwriteText = await page.locator('body').innerText();
      if (/覆盖|已存在|already exists/i.test(overwriteText)) {
        console.log(JSON.stringify({ save_dialog: overwriteText.slice(-1200) }));
        await page.screenshot({ path: path.join(evidence, 'overwrite-dialog.png') });
        await page.getByText('覆盖', { exact: true }).click();
      }
      const response = await responsePromise;
      assert.equal(response.status(), 200, await response.text());
      const saved = await page.request.get(`${baseURL}/userdata/${encodeURIComponent(`workflows/FLUX3_PE/${name}.json`)}`);
      assert.equal(saved.status(), 200, await saved.text());
      return await saved.json();
    };
    const run = async (name, iteration) => {
      const queued = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith('/prompt'), { timeout: 20000 });
      // The user-facing Run button exercises widget serialization and queue validation.
      await page.getByRole('button', { name: /^\s*运行$/ }).click();
      const response = await queued;
      const body = await response.json();
      assert.equal(response.status(), 200, JSON.stringify(body));
      const promptId = body.prompt_id;
      console.log(`QUEUED ${name} #${iteration}: ${promptId}`);
      const deadline = Date.now() + 900000;
      while (Date.now() < deadline) {
        const result = await page.request.get(`${baseURL}/history/${promptId}`, { timeout: 30000 });
        const history = await result.json();
        if (history[promptId]) {
          const entry = history[promptId];
          assert.equal(entry.status.status_str, 'success', JSON.stringify(entry.status));
          assert.ok(entry.outputs['3']?.text?.[0], 'The prompt preview did not receive output');
          assert.ok(entry.outputs['4']?.text?.[0], 'The details preview did not receive output');
          const details = JSON.parse(entry.outputs['4'].text[0]);
          assert.equal(details.plugin_version, version, 'The ComfyUI backend is running a different plugin version');
          assert.equal(details.source_identity, sourceIdentity, 'The ComfyUI backend is running stale Python code');
          if (process.env.FLUX3_PE_EXPECT_BACKEND) assert.equal(details.backend, process.env.FLUX3_PE_EXPECT_BACKEND, 'The workflow used a different backend');
          if (name.startsWith('04')) {
            assert.equal(details.mode, 'i2i');
            assert.equal(details.reference_sizes.length, 2);
          }
          if (name.startsWith('03')) {
            const images = entry.outputs['19']?.images;
            assert.ok(images?.length, 'No image was produced');
            for (const item of images) {
              const image = await page.request.get(`${baseURL}/view?${new URLSearchParams(item)}`);
              assert.equal(image.status(), 200, 'Output image could not be opened');
              assert.ok((await image.body()).length > 1000, 'Output image is empty');
            }
          }
          const cached = entry.status.messages.filter(([kind]) => kind === 'execution_cached').flatMap(([, data]) => data.nodes);
          assert.ok(!cached.includes('2'), 'ComfyUI skipped the PE node');
          console.log(`PASS ${name} #${iteration}: cache_hit=${details.cache_hit}`);
          await page.waitForTimeout(500);
          return { name, iteration, prompt_id: promptId, details, cached, history: entry };
        }
        await page.waitForTimeout(500);
      }
      throw new Error(`Execution timeout: ${promptId}`);
    };
    if (mode === 'roundtrip' || mode === 'execute') {
      const reportPath = path.join(evidence, `${mode}.json`);
      const report = process.argv.includes('--resume') && fs.existsSync(reportPath) ? JSON.parse(fs.readFileSync(reportPath, 'utf8')) : { url: baseURL, version, source_identity: sourceIdentity, inputs_sha256: inputHashes, qa_script_sha256: qaScriptHash, saved: [], runs: [], errors: [] };
      assert.equal(report.url, baseURL, 'Cannot resume canvas evidence from a different server');
      assert.equal(report.version, version, 'Cannot resume canvas evidence from a different version');
      assert.equal(report.source_identity, sourceIdentity, 'Cannot resume canvas evidence from different Python sources');
      assert.deepEqual(report.inputs_sha256, inputHashes, 'Cannot resume canvas evidence from different workflow inputs');
      assert.equal(report.qa_script_sha256, qaScriptHash, 'Cannot resume canvas evidence from a different QA script');
      assert.deepEqual(report.errors, [], 'Cannot resume canvas evidence with previous frontend errors');
      if (process.argv.includes('--resume')) {
        for (const name of names) {
          const saved = report.saved.filter(item => item.name === name);
          const runs = report.runs.filter(item => item.name === name);
          if (!saved.length && !runs.length) continue;
          assert.equal(saved.length, 1, `Cannot resume incomplete canvas evidence for ${name}; use a fresh server and run without --resume`);
          if (mode === 'execute') {
            const required = name.startsWith('01') ? 4 : 1;
            assert.deepEqual(runs.map(item => item.iteration), Array.from({ length: required }, (_, index) => index + 1), `Cannot resume partial canvas execution for ${name}; use a fresh server and run without --resume`);
          }
        }
      }
      report.errors = errors;
      const persist = () => fs.writeFileSync(path.join(evidence, `${mode}.json`), JSON.stringify(report, null, 2));
      try {
        assert.deepEqual(errors, [], 'Frontend JavaScript errors occurred during startup');
        for (const name of names) {
          const requiredRuns = name.startsWith('01') ? 4 : 1;
          if (process.argv.includes('--resume') && report.saved.some(item => item.name === name) && (mode !== 'execute' || report.runs.filter(item => item.name === name).length === requiredRuns)) continue;
          const workflow = JSON.parse(fs.readFileSync(path.join(root, 'examples', `${name}.json`), 'utf8'));
          const api = JSON.parse(fs.readFileSync(path.join(root, 'examples', `${name}.api.json`), 'utf8'));
          await load(workflow, name);
          assert.deepEqual(cleanPrompt((await current()).output), cleanPrompt(api), `${name}: widget/link roundtrip changed the API graph`);
          // Drag a real node header with the mouse, then verify the graph moved.
          const point = await page.evaluate(() => {
            const app = window.comfyAPI.app.app;
            const node = app.graph.getNodeById(2);
            const rect = app.canvas.canvas.getBoundingClientRect();
            return { before: [...node.pos], x: rect.left + (node.pos[0] + 180 + app.canvas.ds.offset[0]) * app.canvas.ds.scale, y: rect.top + (node.pos[1] - 14 + app.canvas.ds.offset[1]) * app.canvas.ds.scale };
          });
          await page.mouse.move(point.x, point.y);
          await page.mouse.down();
          await page.mouse.move(point.x + 16, point.y + 12, { steps: 8 });
          await page.mouse.up();
          const moved = await page.evaluate(() => [...window.comfyAPI.app.app.graph.getNodeById(2).pos]);
          assert.ok(moved[0] !== point.before[0] || moved[1] !== point.before[1], `${name}: mouse drag did not move the node`);
          // A drag probe must not accumulate offsets in the delivered workflow.
          await page.evaluate(before => {
            const app = window.comfyAPI.app.app;
            app.graph.getNodeById(2).pos = [...before];
            app.graph.setDirtyCanvas(true, true);
          }, point.before);
          const restored = await page.evaluate(() => {
            const graph = window.comfyAPI.app.app.graph;
            const pe = graph.getNodeById(2), preview = graph.getNodeById(3);
            return { pos: [...pe.pos], gap: preview.pos[0] - pe.pos[0] - pe.size[0] };
          });
          assert.deepEqual(restored.pos, point.before, `${name}: drag probe changed saved node position`);
          assert.ok(restored.gap >= 30, `${name}: PE node overlaps or crowds the prompt preview`);
          await page.evaluate(() => window.comfyAPI.app.app.canvas.deselectAllNodes());
          await page.getByRole('button', { name: '适应视图 (.)', exact: true }).click();
          const saved = await save(name);
          await load(saved, name);
          assert.deepEqual(cleanPrompt((await current()).output), cleanPrompt(api), `${name}: saved workflow could not be reopened intact`);
          assert.ok(!/发现\s*\d+\s*个错误/.test(await page.locator('body').innerText()), `${name}: frontend reports validation errors`);
          fs.writeFileSync(path.join(evidence, `${name}.json`), JSON.stringify(saved, null, 2));
          report.saved.push({ name, nodes: saved.nodes.length, links: saved.links.length, dragged: true, reopened: true, file: `workflows/FLUX3_PE/${name}.json` });
          persist();
          if (mode === 'execute') {
            const first = await run(name, 1);
            report.runs.push(first); persist();
            if (name.startsWith('01')) {
              const second = await run(name, 2);
              assert.equal(second.details.cache_hit, true, 'Identical cached request did not hit the engine result cache');
              report.runs.push(second); persist();
              await page.evaluate(() => { window.comfyAPI.app.app.graph.getNodeById(2).widgets.find(w => w.name === 'use_cache').value = false; });
              for (const iteration of [3, 4]) {
                const fresh = await run(name, iteration);
                assert.equal(fresh.details.cache_hit, false, 'Disabling cache still reused PE output');
                report.runs.push(fresh); persist();
              }
            }
          }
          await page.screenshot({ path: path.join(evidence, `${name}-${mode}.png`) });
          console.log(`SAVED AND REOPENED ${name}`);
        }
        assert.deepEqual(errors, [], 'Frontend JavaScript errors occurred');
      } finally {
        // Preserve errors even when the final assertion or a browser action fails.
        persist();
      }
    }
    if (mode === 'inspect' || mode === 'probe-save') {
      const workflow = JSON.parse(fs.readFileSync(path.join(root, 'examples/03_pe_to_flux3.json'), 'utf8'));
      const state = await page.evaluate(async workflow => {
        const { app } = await import('/scripts/app.js');
        await app.loadGraphData(workflow, true, true, 'FLUX3_PE_review.json');
        app.canvas.fitView?.();
        return {
          appKeys: Object.keys(app),
          appMethods: Object.getOwnPropertyNames(Object.getPrototypeOf(app)),
          nodes: app.graph._nodes.map(node => ({ id: node.id, type: node.type, pos: node.pos, size: node.size, widgets: node.widgets?.map(w => ({ name: w.name, value: w.value })) })),
          prompt: (await app.graphToPrompt()).output,
          buttons: [...document.querySelectorAll('button')].map(b => ({ text: b.textContent, label: b.getAttribute('aria-label'), title: b.title })).filter(b => b.text || b.label || b.title),
        };
      }, workflow);
      fs.writeFileSync(path.join(evidence, 'initial-state.json'), JSON.stringify({ ...state, errors }, null, 2));
      await page.screenshot({ path: path.join(evidence, 'initial-canvas.png') });
      console.log(JSON.stringify({ nodes: state.nodes.length, appMethods: state.appMethods, buttons: state.buttons, errors }));
      if (mode === 'probe-save') {
        await page.getByRole('button', { name: '图形模式，工作流操作', exact: true }).click();
        await page.getByText('另存为', { exact: true }).click();
        await page.waitForTimeout(600);
        console.log(JSON.stringify({ body: await page.locator('body').innerText(), inputs: await page.locator('input').evaluateAll(inputs => inputs.map(i => ({ value: i.value, placeholder: i.placeholder, type: i.type }))) }));
        await page.screenshot({ path: path.join(evidence, 'save-dialog.png') });
      }
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
