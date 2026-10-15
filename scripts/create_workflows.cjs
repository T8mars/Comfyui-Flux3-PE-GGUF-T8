// Build editable workflows with the installed ComfyUI frontend and its schemas.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'examples');
  fs.mkdirSync(output, { recursive: true });
  const read = name => JSON.parse(fs.readFileSync(path.join(output, `${name}.api.json`), 'utf8'));
  const text = read('01_prompt_enhancement');
  const layout = read('02_layout_enhancement');
  const full = read('03_pe_to_flux3');
  const editing = JSON.parse(JSON.stringify(text));
  editing['2'].inputs.prompt = 'Place the blue ball from the first image on the lawn in the second image.';
  editing['2'].inputs['images.image_1'] = ['10', 0];
  editing['2'].inputs['images.image_2'] = ['11', 0];
  editing['10'] = { class_type: 'LoadImage', inputs: { image: 'FLUX3_PE_reference_objects.png' } };
  editing['11'] = { class_type: 'LoadImage', inputs: { image: 'FLUX3_PE_reference_scene.png' } };
  for (const graph of [text, layout, full, editing]) {
    graph['4'] = { class_type: 'PreviewAny', inputs: { source: ['2', 1] }, _meta: { title: 'PE 运行详情' } };
  }
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1100 } });
    await page.goto(process.env.COMFY_URL || 'http://127.0.0.1:8267', { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => Boolean(window.comfyAPI?.app?.app?.graph && window.LiteGraph?.registered_node_types?.Flux3PromptEnhance), null, { timeout: 60000 });
    await page.locator('#splash-loader').waitFor({ state: 'hidden', timeout: 60000 });
    const records = [];
    for (const [name, graph] of [['01_prompt_enhancement', text], ['02_layout_enhancement', layout], ['03_pe_to_flux3', full], ['04_reference_editing', editing]]) {
      const result = await page.evaluate(async ({ name, graph }) => {
        const { app } = await import('/scripts/app.js');
        await app.loadApiJson(graph, `${name}.json`);
        const isFull = name.startsWith('03');
        const isEdit = name.startsWith('04');
        const positions = isFull ? {
          '1': [60, 100, 340, 130], '2': [450, 100, 500, 480],
          '3': [1010, 100, 600, 290], '4': [1010, 850, 600, 310],
          '10': [60, 650, 340, 120], '11': [60, 820, 340, 140],
          '17': [60, 1040, 340, 110], '15': [450, 820, 320, 170],
          '12': [1010, 470, 310, 230], '13': [1360, 470, 230, 90],
          '14': [1360, 620, 230, 90], '16': [1660, 470, 330, 360],
          '18': [1660, 1000, 290, 100], '19': [2060, 470, 440, 540],
        } : {
          '1': [60, 100, 340, 130], '2': [450, 100, 500, 540],
          '3': [1010, 100, 650, 540], '4': [1010, 700, 650, 350],
          '10': [60, 300, 340, 300], '11': [450, 720, 500, 330],
        };
        for (const node of app.graph._nodes) {
          const box = positions[node.id];
          if (box) { node.pos = box.slice(0, 2); node.setSize(box.slice(2)); }
          if (String(node.id) === '3') node.title = '增强后的提示词';
          if (String(node.id) === '4') node.title = 'PE 运行详情（缓存 / 恢复 / 耗时）';
          if (isEdit && String(node.id) === '10') node.title = '参考图 1：物体';
          if (isEdit && String(node.id) === '11') node.title = '参考图 2：场景';
          for (const widget of node.widgets || []) {
            if (widget.name === 'control_after_generate') widget.value = 'fixed';
          }
        }
        app.canvas.ds.scale = isFull ? 0.64 : 0.84;
        app.canvas.ds.offset = [35, 45];
        app.graph.setDirtyCanvas(true, true);
        return await app.graphToPrompt();
      }, { name, graph });
      fs.writeFileSync(path.join(output, `${name}.json`), JSON.stringify(result.workflow, null, 2));
      fs.writeFileSync(path.join(output, `${name}.api.json`), JSON.stringify(result.output, null, 2));
      records.push({ name, nodes: result.workflow.nodes.length, links: result.workflow.links.length });
    }
    console.log(JSON.stringify(records));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
