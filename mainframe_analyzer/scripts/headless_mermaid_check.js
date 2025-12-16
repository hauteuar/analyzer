const fs = require('fs');
const path = require('path');
const htmlPath = path.resolve(__dirname, '..', 'work_flow_automation.html');
const content = fs.readFileSync(htmlPath, 'utf8');
const blocks = Array.from(content.matchAll(/<pre class=\"mermaid\">([\s\S]*?)<\/pre>/g)).map(m=>m[1].trim());
console.log('found_blocks', blocks.length);

(async function(){
    let mermaid;
    try {
        mermaid = require('mermaid');
        console.log('loaded mermaid package via require:', typeof mermaid.parse === 'function' ? 'parse available' : 'no parse');
    } catch (e) {
        // try importing ESM build directly from global mermaid-cli install
        const esmPath = 'file:///C:/Program%20Files/nodejs/node_modules/@mermaid-js/mermaid-cli/node_modules/mermaid/dist/mermaid.esm.mjs';
        try {
            const mod = await import(esmPath);
            mermaid = mod && (mod.default || mod);
            console.log('loaded mermaid package via ESM import');
        } catch (e2) {
            // fallback to reading and evaluating UMD bundle in a VM
            try {
                const vm = require('vm');
                const mermaidUMDPath = 'C:/Program Files/nodejs/node_modules/@mermaid-js/mermaid-cli/node_modules/mermaid/dist/mermaid.js';
                const code = fs.readFileSync(mermaidUMDPath, 'utf8');
                const sandbox = { window: {}, document: {}, navigator: {}, module: {}, exports: {}, global: {} };
                vm.runInNewContext(code + '\nmodule.exports = window.mermaid || global.mermaid;', sandbox);
                mermaid = sandbox.module.exports || sandbox.window.mermaid || sandbox.global.mermaid;
                console.log('loaded mermaid via UMD eval');
            } catch (e3) {
                console.error('Failed to load mermaid via require/import/eval:', e.message, e2 && e2.message, e3 && e3.message);
                process.exit(2);
            }
        }
    }

    try {
        if (typeof mermaid.initialize === 'function') mermaid.initialize({startOnLoad:false});
    } catch (e) { /* ignore */ }

    let errors = [];
    blocks.forEach((b, idx) => {
        try {
            if (typeof mermaid.parse !== 'function') {
                // try mermaid.mermaidAPI.parse
                if (mermaid.mermaidAPI && typeof mermaid.mermaidAPI.parse === 'function') {
                    mermaid.mermaidAPI.parse(b);
                } else {
                    throw new Error('mermaid parse API not found');
                }
            } else {
                mermaid.parse(b);
            }
            console.log(`block ${idx+1}: OK`);
        } catch (err) {
            console.error(`block ${idx+1}: ERROR -> ${err && err.message ? err.message : String(err)}`);
            const lines = b.split('\n');
            const preview = lines.slice(0, 10).map((l,i)=>`${i+1}: ${l}`).join('\n');
            console.error('preview:\n' + preview);
            errors.push({block: idx+1, message: err.message || String(err), preview});
        }
    });

    if (errors.length===0) {
        console.log('NO_ERRORS');
        process.exit(0);
    } else {
        console.log('ERROR_COUNT', errors.length);
        process.exit(1);
    }
})();
