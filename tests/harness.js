const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
let initial = START, historyMoves = [], database = localStorage.getItem('test-database') || 'lichess', explorerOpen = true, failNext = false;
let configOpen = false, playerName = '';
let emptyNative = false;
const api = async message => (await fetch('/api', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(message)})).json();
window.chrome.runtime = {sendMessage:async message => {
  if (failNext && message.type === 'analyse') { failNext = false; return {ok:false,error:'Simulated missing native host. Run install.ps1, then Retry.'}; }
  return api(message);
}};
const ctrl = {node:{fen:START},nodeList:[{fen:START}],data:{game:{variant:{key:'standard'}}},
  ongoing:false,
  playUci:async uci => {historyMoves.push(uci);await update();},
  redraw:()=>renderNative(),
  explorer:{enabled:()=>explorerOpen,db:()=>database,setHovering:()=>{},config:{
    data:{db:value=>{database=value;localStorage.setItem('test-database',value);},
      open:value=>value===undefined?configOpen:(configOpen=value),playerName:{value:()=>playerName}},
    selectPlayer:name=>{playerName=name==='me'?'TestPlayer':name;database='player';localStorage.setItem('test-database',database);},
    toggleOpen:()=>{configOpen=!configOpen;}
  },reload:()=>renderNative()}};
function renderNative() {
  document.querySelector('.explorer-box').classList.toggle('reduced', emptyNative);
  document.querySelector('.native-empty').hidden = !emptyNative || configOpen;
  document.querySelector('.native-title').textContent = `${database === 'masters' ? 'Masters' : database === 'player' ? 'Player' : 'Lichess'} database fixture`;
  document.querySelector('.native-config').hidden = !configOpen;
  document.querySelector('.native-counts').hidden = configOpen || emptyNative;
}
window.lichess = {analysis:{navigate:{ctrl}}};
async function update() {
  const reply = await api({type:'position',initialFen:initial,moves:historyMoves});
  if (!reply.ok) throw Error(reply.error);
  ctrl.nodeList = reply.result.nodes;ctrl.node=ctrl.nodeList.at(-1);
  document.querySelector('.fen').value = ctrl.node.fen;
  document.querySelector('.moves').textContent=ctrl.nodeList.slice(1).map(n=>n.san).join(' ');
  const board = document.querySelector('.board');board.replaceChildren();
  for (let rank=8;rank>=1;rank--) for (let f=0;f<8;f++) {
    const square = document.createElement('div');square.className=`square ${(rank+f)%2 ? 'dark' : ''}`;
    square.textContent=reply.result.pieces[String.fromCharCode(97+f)+rank] || '';board.append(square);
  }
}
document.querySelector('#start').onclick=()=>{initial=START;historyMoves=[];update();};
document.querySelector('#e4').onclick=()=>{initial=START;historyMoves=['e2e4'];update();};
document.querySelector('#back').onclick=()=>{historyMoves.pop();update();};
document.querySelector('#castle').onclick=()=>{initial='r3k2r/ppp2ppp/2n1bn2/3qp3/8/2NPBN2/PPP2PPP/R3K2R w KQkq - 2 9';historyMoves=[];update();};
document.querySelector('#promotion').onclick=()=>{initial='8/P6k/8/8/8/8/7p/K7 w - - 0 1';historyMoves=[];update();};
document.querySelector('#mate').onclick=()=>{initial=START;historyMoves=['f2f3','e7e5','g2g4','d8h4'];update();};
document.querySelector('#masters').onclick=()=>{ctrl.explorer.config.data.db('masters');ctrl.explorer.reload();};
document.querySelector('#lichess').onclick=()=>{ctrl.explorer.config.data.db('lichess');ctrl.explorer.reload();};
document.querySelector('#player').onclick=()=>{ctrl.explorer.config.selectPlayer('me');ctrl.explorer.reload();};
document.querySelector('#configure').onclick=()=>{ctrl.explorer.config.toggleOpen();ctrl.redraw();};
document.querySelector('#redraw').onclick=()=>{const box=document.querySelector('.explorer-box');box.querySelector('.me-root')?.remove();};
document.querySelector('#close').onclick=()=>{explorerOpen=!explorerOpen;document.querySelector('.explorer-box').style.display=explorerOpen?'':'none';document.querySelector('#close').textContent=explorerOpen?'Close explorer':'Open explorer';};
document.querySelector('#failure').onclick=()=>{failNext=true;initial=START;historyMoves=['d2d4'];update();};
document.querySelector('#empty-native').onclick=()=>{emptyNative=true;renderNative();};
document.querySelector('#restore-native').onclick=()=>{emptyNative=false;renderNative();};
document.querySelector('#chapter').onclick=()=>{
  ctrl.study = {};
  window.history.replaceState({}, '', '/study/STUDY123/CHAPTER2');
  initial='8/P6k/8/8/8/8/7p/K7 w - - 0 1';historyMoves=[];update();
};
document.querySelector('#branch').onclick=()=>{
  initial=START;historyMoves=['d2d4','d7d5'];update();
};
document.querySelector('#ongoing').onclick=()=>{ctrl.ongoing=!ctrl.ongoing;};
renderNative();update();
