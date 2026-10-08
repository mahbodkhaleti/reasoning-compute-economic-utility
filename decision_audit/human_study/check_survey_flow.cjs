// Static state-machine diagnostic with a mock document; no browser is launched.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'survey.html'),'utf8');
const code=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(code);
const elements={};for(const id of ['intro','trial','finish','pid','adult','start','msg','progress','cards','testnote','chooseA','chooseB','stop','download'])elements[id]={value:'',checked:false,textContent:'',innerHTML:'',classList:{toggle(){}},onclick:null};
const context=vm.createContext({document:{getElementById:id=>elements[id]},performance:{now:()=>100},Date,Blob:class{},URL:{createObjectURL(){throw Error('Export not exercised in static test')}},setTimeout:()=>0});vm.runInContext(code,context);
elements.pid.value='TEST_FLOW';elements.start.onclick();assert(elements.msg.textContent.includes('confirm'));
elements.adult.checked=true;elements.start.onclick();let design=vm.runInContext('deck',context);assert.equal(design.length,12);assert.equal(design.filter(x=>x.block==='equal_fee').length,6);assert.equal(new Set(design.map(x=>x.trial_id)).size,12);
for(let i=0;i<12;i++)elements[i%2?'chooseB':'chooseA'].onclick();const result=JSON.parse(vm.runInContext('JSON.stringify({...session,responses:records})',context));assert.equal(result.responses.length,12);assert(result.is_test);for(const r of result.responses)assert.equal(r.chosen_profile,r.choice==='A'?r.left_profile:r.right_profile);
elements.start.onclick();assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(deck)',context)),JSON.parse(JSON.stringify(design)));elements.stop.onclick();assert.equal(vm.runInContext('records.length',context),0);assert.equal(vm.runInContext('session',context),null);
console.log('Static survey checks passed: validation, deterministic order, 12 paired choices, test flag and stop behavior. No browser render or observed human response claimed.');
