const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../extension/content.js'), 'utf8');
const math = source.split('// BEGIN DISPLAY MATH')[1].split('// END DISPLAY MATH')[0];
const context = vm.createContext({});
vm.runInContext('// BEGIN DISPLAY MATH' + math, context);
const {typicalSeconds, displaySettings, resignationIndicator, timeLabel} = context;
const near = (actual, expected) => assert.ok(Math.abs(actual-expected) < 1e-10, `${actual} != ${expected}`);
function distribution(entries) { const values = Array(121).fill(0); for (const [s,p] of entries) values[s] = p; return values; }
test('temperature one and top-p one recover the full mean, including long tails', () => {
  const p = distribution([[1,.15],[2,.4],[3,.3],[120,.15]]);
  near(typicalSeconds(p,1,1),19.85);
  assert.ok(typicalSeconds(p,.5,1) < typicalSeconds(p,1,1));
  assert.ok(Math.abs(typicalSeconds(p,.05,1)-2) < .004);
});
test('top-p retains the threshold-crossing bin and renormalizes', () => {
  const p = distribution([[1,.15],[2,.4],[3,.3],[20,.15]]);
  near(typicalSeconds(p,1,.7), (2*.4+3*.3)/.7);
  near(typicalSeconds(p,1,.4),2);
});
test('ties, zero bins, single overflow bin and invalid distributions', () => {
  near(typicalSeconds(distribution([[2,.5],[4,.5]]),.05,1),3);
  near(typicalSeconds(distribution([[2,.5],[4,.5]]),.05,.5),2);
  near(typicalSeconds(distribution([[120,1]]),.05,.01),120);
  for (const p of [[],Array(121).fill(0),distribution([[1,NaN]]),distribution([[1,-1]])])
    assert.equal(typicalSeconds(p,.5,1),undefined);
});
test('settings clamp invalid values and an old host cannot fake typical timing', () => {
  const settings = displaySettings({timeTemperature:0,timeTopP:3,timeStatistic:'bad',includeResignation:'yes'});
  assert.equal(settings.timeStatistic,'mode'); assert.equal(settings.includeResignation,false);
  assert.equal(Object.hasOwn(settings,'timeTemperature'),false);
  assert.equal(Object.hasOwn(settings,'timeTopP'),false);
  assert.equal(timeLabel({modeSeconds:120},settings),'120s+');
  assert.equal(timeLabel({meanSeconds:7}, {...settings,timeStatistic:'typical'}),undefined);
  assert.equal(timeLabel({meanSeconds:7}, {...settings,timeStatistic:'mean'}),'≈7.0s');
});
test('resignation stays separate from policy and the switch hides its indicator', () => {
  const result={moves:[{uci:'a',probability:.8},{uci:'b',probability:.2}],
    wdl:{white:.5,draws:.12,black:.38},resignation:{probability:.54}};
  const before=JSON.stringify(result);
  assert.equal(resignationIndicator(result,{includeResignation:true}),.54);
  assert.equal(resignationIndicator(result,{includeResignation:false}),undefined);
  assert.equal(resignationIndicator({...result,terminal:'checkmate'},{includeResignation:true}),undefined);
  assert.equal(resignationIndicator({...result,resignation:{probability:NaN}},{includeResignation:true}),undefined);
  assert.equal(JSON.stringify(result),before);
  near(result.moves.reduce((sum,m)=>sum+m.probability,0),1);
  assert.deepEqual(result.wdl,{white:.5,draws:.12,black:.38});
});
test('terminal and older host results remain unchanged', () => {
  for(const result of [{terminal:'checkmate'},{moves:[],wdl:{}},{resignation:{probability:NaN}}])
    assert.equal(resignationIndicator(result,{includeResignation:true}),undefined);
});

test('model setting and sub-second labels', () => {
  assert.equal(displaySettings({}).model, '79m');
  assert.equal(displaySettings({model:'23m'}).model, '23m');
  assert.equal(displaySettings({model:'7b'}).model, '79m');
  const timing = {modeSeconds:.48, typicalSeconds:.55, meanSeconds:12.34};
  assert.equal(timeLabel(timing,{timeStatistic:'mode'}),'0.48s');
  assert.equal(timeLabel(timing,{timeStatistic:'typical'}),'≈0.55s');
  assert.equal(timeLabel(timing,{timeStatistic:'mean'}),'≈12.3s');
  assert.equal(timeLabel({modeSeconds:119.7},{timeStatistic:'mode'}),'120s+');
});
