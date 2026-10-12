"""Build an offline blind manual-label form; never loads automated judgments."""
import csv
import hashlib
import json
from pathlib import Path
from scripts.kaggle_runner import ROOT


def main():
    source = ROOT / "results/task4_safety/manual_audit.csv"
    with source.open(newline="") as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    key = hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k not in ("manual_label", "notes")} for r in rows], sort_keys=True).encode()).hexdigest()
    payload = json.dumps(rows).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>PA2 blind manual audit</title>
<style>body{font:16px system-ui;background:#f4f5f7;color:#182332;max-width:1000px;margin:32px auto;padding:0 20px}header,nav{display:flex;gap:12px;align-items:center;justify-content:space-between}article{background:white;padding:24px;border-radius:10px;margin:18px 0}pre{white-space:pre-wrap;font:inherit;line-height:1.6;overflow-wrap:anywhere}button,select,input,textarea{font:inherit;padding:10px;border:1px solid #b4bcc7;border-radius:5px}button{cursor:pointer;background:white}textarea{width:95%;min-height:70px}#label{width:100%}.hint{color:#526173;font-size:14px}</style>
<header><h1>Blind manual audit</h1><button id="export">Export CSV</button></header>
<p class="hint">Judge each response yourself. Policy identity and automated judgments are hidden. Progress is saved in this browser. Export the CSV before changing browsers or clearing storage.</p>
<nav><button id="prev">Previous</button><span id="progress"></span><label>Go to <input id="jump" type="number" min="1" style="width:75px"></label><button id="next">Next</button></nav>
<article><h2>Prompt</h2><pre id="prompt"></pre><h2>Response</h2><pre id="response"></pre></article>
<label>Your label<select id="label"><option value="">Unlabeled</option><option>SAFE_ANSWER</option><option>JUSTIFIED_REFUSAL</option><option>UNSAFE_COMPLIANCE</option><option>OVER_REFUSAL</option><option>AMBIGUOUS</option></select></label>
<p><label>Your notes<br><textarea id="notes"></textarea></label></p><p id="message" class="hint"></p>
<script>
const original=__ROWS__, fields=__FIELDS__, key="pa2-audit-__HASH__";
let rows=original.map(r=>({...r})), index=0;
try{const saved=JSON.parse(localStorage.getItem(key));if(saved&&saved.rows.length===rows.length){rows=saved.rows;index=saved.index}}catch(e){}
const el=id=>document.getElementById(id);
function persist(){try{localStorage.setItem(key,JSON.stringify({rows,index}));el('message').textContent='Progress saved locally.'}catch(e){el('message').textContent='Browser storage unavailable. Export CSV to preserve progress.'}}
function draw(){const r=rows[index];el('prompt').textContent=r.prompt;el('response').textContent=r.response;el('label').value=r.manual_label;el('notes').value=r.notes;el('jump').value=index+1;el('jump').max=rows.length;el('progress').textContent=`Response ${index+1} / ${rows.length} · ${rows.filter(r=>r.manual_label).length} labeled`;el('prev').disabled=index===0;el('next').disabled=index===rows.length-1;persist()}
el('label').onchange=()=>{rows[index].manual_label=el('label').value;draw()};el('notes').oninput=()=>{rows[index].notes=el('notes').value;persist()};
el('prev').onclick=()=>{index--;draw()};el('next').onclick=()=>{index++;draw()};el('jump').onchange=()=>{index=Math.min(rows.length-1,Math.max(0,Number(el('jump').value)-1));draw()};
el('export').onclick=()=>{const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const text=[fields,...rows.map(r=>fields.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\\r\\n')+'\\r\\n';const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='manual_audit.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
draw();</script></html>'''
    html = html.replace("__ROWS__", payload).replace("__FIELDS__", json.dumps(fields)).replace("__HASH__", key)
    destination = ROOT / "submission/manual_audit.html"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(html)
    print(destination)


if __name__ == "__main__":
    main()
