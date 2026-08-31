const pptxgen = require('pptxgenjs');
const D = require('../deck.json');

const INK='10161C', PAPER='FFFFFF', GRAPH='3A4750', VIRID='0E7C66',
      BRASS='C2872B', SLATE='93A1AA', MIST='EEF1F3', RULE='D7DDE2', WHITE='FFFFFF';
const H='Cambria', B='Calibri';
const W=13.33, HT=7.5, M=0.62;

const p = new pptxgen();
p.layout='LAYOUT_WIDE';
p.author='AgentDescent'; p.title='AlgoTune: Eight Tasks, Measured Twice';

const shadow=()=>({type:'outer',color:'8899A6',blur:10,offset:2,angle:90,opacity:0.18});

function dark(s){ s.background={color:INK}; }
function head(s,kicker,title,sub,subH){
  s.addText(kicker,{x:M,y:0.42,w:W-2*M,h:0.26,fontFace:B,fontSize:11,bold:true,
    color:VIRID,charSpacing:2.2,isTextBox:true,margin:0});
  s.addText(title,{x:M,y:0.72,w:W-2*M,h:0.62,fontFace:H,fontSize:33,bold:true,
    color:INK,isTextBox:true,margin:0});
  if(sub) s.addText(sub,{x:M,y:1.36,w:W-2*M,h:subH||0.48,fontFace:B,fontSize:13.5,
    color:GRAPH,lineSpacing:18,isTextBox:true,margin:0});
}
function foot(s,t){
  s.addText(t,{x:M,y:HT-0.66,w:W-2*M,h:0.46,fontFace:B,fontSize:9.5,color:SLATE,
    lineSpacing:12,isTextBox:true,margin:0});
}
function chip(s,x,y,txt,w){
  s.addShape(p.ShapeType.roundRect,{x,y,w:w||0.82,h:0.24,fill:{color:MIST},
    line:{color:RULE,width:0.5},rectRadius:0.1});
  s.addText(txt,{x,y,w:w||0.82,h:0.24,fontFace:B,fontSize:9,color:GRAPH,
    align:'center',valign:'middle',isTextBox:true,margin:0});
}
const tblBase={fontFace:B,fontSize:11.5,color:GRAPH,border:{type:'solid',color:RULE,pt:0.5},
               valign:'middle',autoPage:false};
const hdr=t=>({text:t,options:{bold:true,color:WHITE,fill:{color:INK},fontSize:10.5,align:'center'}});

/* ---------------------------------------------------- 1  title */
let s=p.addSlide(); dark(s);
s.addShape(p.ShapeType.rect,{x:0,y:0,w:W,h:HT,fill:{color:INK}});
s.addText('AlgoTune  ·  eight tasks  ·  one configuration  ·  two seeds',
  {x:M,y:2.05,w:W-2*M,h:0.3,fontFace:B,fontSize:12,bold:true,color:VIRID,
   charSpacing:2.4,isTextBox:true,margin:0});
s.addText('Measured Twice',
  {x:M,y:2.42,w:9.4,h:1.0,fontFace:H,fontSize:54,bold:true,color:WHITE,isTextBox:true,margin:0});
s.addText('An ERA flat-PUCT tree search on AgentDescent, put beside MetaEvolve, its untrained baseline, OpenEvolve and the AlgoTune leaderboard — and what turns out not to be the same measurement.',
  {x:M,y:3.5,w:8.4,h:0.9,fontFace:B,fontSize:15,color:'C6D0D6',lineSpacing:24,isTextBox:true,margin:0});
[['2.285x / 2.254x','8-task harmonic mean, seeds 0 and 1'],
 ['16','complete runs, 10.7 h, 4.62 M tokens'],
 ['1.4%','distance between the two seeds']].forEach((r,i)=>{
  const x=M+i*4.05;
  s.addText(r[0],{x,y:5.05,w:3.8,h:0.62,fontFace:H,fontSize:30,bold:true,color:VIRID,isTextBox:true,margin:0});
  s.addText(r[1],{x,y:5.68,w:3.8,h:0.5,fontFace:B,fontSize:11,color:'9FB0B9',isTextBox:true,margin:0});
});
s.addText('AgentDescent · examples/era · AlgoTune upstream dff9914 · 2026-08-29',
  {x:M,y:HT-0.6,w:W-2*M,h:0.3,fontFace:B,fontSize:10,color:'6A7C86',isTextBox:true,margin:0});
s.addNotes('Eight AlgoTune tasks, one search configuration, run twice. Headline 2.285x and 2.254x harmonic mean. The deck argues two things: the number is reproducible, and half the published comparison is not measuring the same problem.');

/* ---------------------------------------------------- 2  our harness */
s=p.addSlide();
head(s,'WHAT WE RAN','The harness, and the model',
  'Everything below is this port. The next slide is everyone else.');
const specs=[
 ['Search','ERA flat-PUCT tree search (FUTS) ported onto AgentDescent. One tree per task; every node is a whole program, never a patch.'],
 ['Model','deepseek-v4-flash-ga-260731, temperature 0.7, thinking disabled, 8 000 max tokens, 4 repair draws per expansion.'],
 ['Budget','99 rollouts per task, 3 workers, asynchronous, c_puct 2.5, model-prior exponent 2.0, staleness policy full.'],
 ['Tasks','Upstream AlgoTune task files at commit dff9914, at upstream’s own calibrated n from reports/generation.json. Nothing rescaled.'],
 ['Timing','Bubblewrap sandbox, network off. Reference and candidate timed in the same process, min of 3 runs. Evaluations hold a mutex so no two are timed at once.'],
 ['Scoring','6 scoring shards of 2 problems; 3 shards of 2 held back and never seen by the search. Any invalid answer voids the task.'],
];
specs.forEach((r,i)=>{
  const col=i%2, row=Math.floor(i/2);
  const x=M+col*6.15, y=1.95+row*1.62;
  s.addShape(p.ShapeType.roundRect,{x,y,w:5.9,h:1.42,fill:{color:MIST},
    line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
  s.addText(r[0],{x:x+0.26,y:y+0.16,w:5.4,h:0.28,fontFace:B,fontSize:10.5,bold:true,
    color:VIRID,charSpacing:1.6,isTextBox:true,margin:0});
  s.addText(r[1],{x:x+0.26,y:y+0.46,w:5.4,h:0.86,fontFace:B,fontSize:11.5,color:GRAPH,
    lineSpacing:15,isTextBox:true,margin:0});
});
foot(s,'16 runs · 2 498 model calls · 0 API failures · 4 618 649 tokens · 10.7 h wall');
s.addNotes('The one thing to stress: n comes from upstream generation.json, unmodified. That is the axis the rest of the deck turns on.');

/* ---------------------------------------------------- 3  the others */
s=p.addSlide();
head(s,'WHO ELSE IS IN THE TABLE','Four other systems, four different setups',
  'Harness and model differ everywhere. So, it turns out, does the problem size.');
const rows=[[hdr('System'),hdr('Harness'),hdr('Model'),hdr('Search budget'),hdr('Problem size n')]];
const others=[
 ['This port','own AlgoTune domain\non AgentDescent','deepseek-v4-flash','99 rollouts/task\n× 2 seeds','upstream calibrated\n(verified)'],
 ['Qwen3-14B, no RL\nMetaEvolve’s baseline','OpenEvolve’s search\narXiv:2607.21971','Qwen3-14B\nno training','50 rounds','not stated'],
 ['MetaEvolve\narXiv:2607.21971','OpenEvolve’s search','Qwen3-14B\n+ RL on synthesised\nevolution trajectories','50 rounds','not stated'],
 ['OpenEvolve','own','gemini-2.5-flash 0.8\n+ gemini-2.5-pro 0.2','100 iterations,\npopulation 1000','10x–4337x below\ncalibrated, on 6 of 8'],
 ['AlgoTune\nleaderboard','AlgoTuner\n(the benchmark’s own)','18 models\n(9 complete here)','the harness’s own','upstream calibrated'],
];
others.forEach((r,i)=>{
  const mine=i===0;
  rows.push(r.map((c,j)=>({text:c,options:{
    bold:(j===0||mine), color: mine?VIRID:(j===4&&(i===1||i===2||i===3)?BRASS:GRAPH),
    fill:{color: mine?'E6F2EF':(i%2?WHITE:MIST)}, fontSize:10.5,
    align:j===0?'left':'center'}})));
});
s.addTable(rows,{...tblBase,x:M,y:1.95,w:W-2*M,colW:[2.05,2.35,2.4,2.15,3.14],
  rowH:[0.34,0.72,0.56,0.56,0.7,0.7]});
foot(s,'The first two rows are one paper’s ablation of a single variable: same search, same model, RL or not. Neither states its problem sizes, and all three run OpenEvolve’s search — whose AlgoTune harness takes n from a config field rather than from AlgoTune’s calibration. DeepMind’s AlphaEvolve is absent because it has never been run on AlgoTune and is not publicly available; the row often labelled with its name is this paper’s untrained baseline.');
s.addNotes('Set up the comparability question here without answering it yet.');

/* ---------------------------------------------------- 4  headline, AlgoTune metric */
s=p.addSlide();
head(s,'THE HEADLINE','Everything published on these eight tasks',
  'One metric — AlgoTune’s own — over the same eight tasks. Nobody excluded, nothing selected.',0.30);
const RK=D.algotune_metric_rank, hf=Math.ceil(RK.length/2), mv=2.45;
// 25 rows now that OpenEvolve is split by phase; tighten the pitch to fit.
[[0,hf,M],[hf,RK.length,7.0]].forEach(([a0,b0,x])=>{
  RK.slice(a0,b0).forEach((r,i)=>{
    const y=1.98+i*0.348, mine=r[2];
    const col = mine?VIRID:(r[3]==='OpenEvolve'?BRASS:SLATE);
    s.addShape(p.ShapeType.roundRect,{x,y,w:5.71,h:0.31,
      fill:{color:mine?'E6F2EF':(i%2?WHITE:MIST)},line:{color:RULE,width:0.6},rectRadius:0.05});
    s.addText(String(a0+i+1),{x:x+0.1,y,w:0.36,h:0.31,fontFace:B,fontSize:9,bold:mine,
      color:mine?VIRID:SLATE,align:'right',valign:'middle',isTextBox:true,margin:0});
    s.addText(r[0],{x:x+0.52,y,w:2.72,h:0.31,fontFace:B,fontSize:9,bold:mine,
      color:col,valign:'middle',isTextBox:true,margin:0});
    s.addText(r[3],{x:x+3.28,y,w:0.86,h:0.31,fontFace:B,fontSize:7,color:SLATE,
      valign:'middle',isTextBox:true,margin:0});
    s.addShape(p.ShapeType.rect,{x:x+4.18,y:y+0.105,w:Math.max(0.92*r[1]/mv,0.03),h:0.10,
      fill:{color:col},line:{width:0}});
    s.addText(r[1].toFixed(3),{x:x+5.12,y,w:0.5,h:0.31,fontFace:B,fontSize:9,bold:mine,
      color:mine?VIRID:GRAPH,align:'right',valign:'middle',isTextBox:true,margin:0});
  });
});
s.addShape(p.ShapeType.roundRect,{x:M,y:6.42,w:W-2*M,h:0.84,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.05});
s.addText([{text:'The third column is the harness. ',options:{bold:true,color:WHITE}},
 {text:'Eighteen AlgoTuner rows run the benchmark’s own agent at its calibrated n; three amber rows run OpenEvolve’s search, and none of those three states its problem sizes. Read that before the order — the top four and the AlgoTuner field may not be measuring the same problems.  ',options:{color:'D6E0E4'}},
 {text:'OpenEvolve* is its own published 1.984x (the final of four phases). Its per-task list is a cross-phase discoveries log, so the 2.267x that list averages to is not a score anyone achieved in one configuration.',options:{color:'E8C98A'}}],
 {x:M+0.26,y:6.50,w:W-2*M-0.52,h:0.68,fontFace:B,fontSize:10.5,lineSpacing:14,
  isTextBox:true,margin:0});
s.addNotes('Under AlgoTune’s own scoring nobody is below 1.0 and no model needs excluding, so this replaces both the “winners only” chart and the four-task workaround.');

/* ---------------------------------------------------- 4b the metric */
s=p.addSlide();
head(s,'THE METRIC, AND A CORRECTION','AlgoTune’s score reproduces exactly',
  'An earlier version of this deck said their aggregation could not be reproduced. That was wrong, and the rule it missed changes what the numbers mean.',0.52);
s.addShape(p.ShapeType.roundRect,{x:M,y:2.06,w:6.0,h:1.62,fill:{color:MIST},
  line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
s.addText('The rule, from AlgoTune’s own description',{x:M+0.26,y:2.18,w:5.5,h:0.28,
  fontFace:B,fontSize:12,bold:true,color:INK,isTextBox:true,margin:0});
s.addText([{text:'“solutions that yield invalid outputs or that have a speedup of under 1× ',options:{color:GRAPH}},
 {text:'assigned a speedup of 1×',options:{bold:true,color:INK}},{text:'”, a task with no result also counts 1×, then the ',options:{color:GRAPH}},
 {text:'harmonic mean',options:{bold:true,color:INK}},{text:' over every task.',options:{color:GRAPH}}],
 {x:M+0.26,y:2.52,w:5.5,h:1.02,fontFace:B,fontSize:12,lineSpacing:17,isTextBox:true,margin:0});
const pr=[[hdr('model'),hdr('published'),hdr('recomputed')]];
D.metric_proof.forEach((r,i)=>pr.push([
 {text:r[0],options:{align:'left',fill:{color:i%2?WHITE:MIST},fontSize:10.5}},
 {text:r[1].toFixed(2),options:{align:'center',fill:{color:i%2?WHITE:MIST},fontSize:10.5,color:SLATE}},
 {text:r[2].toFixed(3),options:{align:'center',fill:{color:i%2?WHITE:MIST},fontSize:10.5,bold:true,color:VIRID}}]));
s.addTable(pr,{...tblBase,x:6.9,y:2.06,w:5.81,colW:[2.85,1.42,1.54],rowH:0.315});
[['What it invalidates','The six models this deck previously showed “below 1.0” — 0.848x down to 0.665x — are at 1.0 under the benchmark’s own rule. Those were raw ratios, which AlgoTune deliberately does not score.'],
 ['What it repairs','Nine models were dropped for having no score on some task. A missing task counts 1×, so nothing needs excluding and all eighteen rank on all eight tasks.'],
 ['What survives','The underlying observation still holds and is still worth stating: those models did submit solvers slower than the reference, mostly by adding a .tolist() the reference does not do. AlgoTune simply chooses not to punish it.']]
.forEach((r,i)=>{
  const y=3.86+i*0.96;
  s.addShape(p.ShapeType.roundRect,{x:M,y,w:W-2*M,h:0.86,fill:{color:MIST},
    line:{color:RULE,width:0.75},rectRadius:0.06});
  s.addText(r[0],{x:M+0.26,y:y+0.1,w:2.4,h:0.28,fontFace:B,fontSize:11.5,bold:true,
    color:i===2?VIRID:BRASS,isTextBox:true,margin:0});
  s.addText(r[1],{x:M+2.8,y:y+0.1,w:9.2,h:0.68,fontFace:B,fontSize:11.5,color:GRAPH,
    lineSpacing:15,isTextBox:true,margin:0});
});
s.addNotes('Say the correction out loud. The reproduction is the evidence that the metric here is the benchmark’s, not one invented for this deck.');

/* ---------------------------------------------------- 4bb hints */
s=p.addSlide();
head(s,'HOW MUCH THE PROMPT WAS TOLD','OpenEvolve, split by phase',
  'OpenEvolve’s report is four phases with different prompts and a score for each. Their headline is the last one.',0.32);
D.hint_ladder.forEach((r,i)=>{
  const y=1.94+i*1.13, mine=i===0;
  s.addShape(p.ShapeType.roundRect,{x:M,y,w:W-2*M,h:1.02,fill:{color:mine?'E6F2EF':MIST},
    line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
  s.addText(r[0],{x:M+0.28,y:y+0.12,w:2.5,h:0.3,fontFace:B,fontSize:13,bold:true,
    color:mine?VIRID:INK,isTextBox:true,margin:0});
  s.addText(r[3]+'x',{x:M+0.28,y:y+0.46,w:2.5,h:0.44,fontFace:H,fontSize:mine?19:24,bold:true,
    color:mine?VIRID:(i===3?BRASS:GRAPH),isTextBox:true,margin:0});
  s.addText(r[1],{x:M+3.0,y:y+0.16,w:8.95,h:0.74,fontFace:B,fontSize:12,color:GRAPH,
    lineSpacing:16,isTextBox:true,margin:0});
});
s.addShape(p.ShapeType.roundRect,{x:M,y:6.62,w:W-2*M,h:0.84,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.05});
s.addText('Their phases are not ordered by hint strength: the specific-hints arm (1.886x) scored below the generic one (1.984x), and they flagged it for overfitting. What matters for reading the table is that even the 1.984x prompt names JAX with a promised 100x+ and tells the model to try lower interpolation orders — which is most of the answer to affine_transform_2d. This port’s arm carries neither, and no bare-versus-hinted pair was run on these eight tasks, so no number here compares the two directly.',
  {x:M+0.26,y:6.70,w:W-2*M-0.52,h:0.68,fontFace:B,fontSize:10.5,color:'D6E0E4',
   lineSpacing:14,isTextBox:true,margin:0});
s.addNotes('The port’s own hint ablation was run on a different task batch and the feature was then removed: naming four techniques was worth 3/8 draws reaching for a compiler against 0/8, and a one-sentence invitation to use the listed packages bought nothing (0/8 against the bare list’s 1/8 on polynomial_real).');

/* ---------------------------------------------------- 4c who else has published */
s=p.addSlide();
head(s,'EVERYTHING ELSE PUBLISHED ON ALGOTUNE','Six sources, and what each one actually is',
  'Searched arXiv, GitHub, HuggingFace and Epoch AI. DeepMind’s AlphaEvolve is absent from this list because it has never been run on AlgoTune and is not publicly available.',0.52);
const os=[[hdr('Source'),hdr('Harness'),hdr('Model'),hdr('What it reports'),hdr('Where')]];
D.other_sources.forEach((r,i)=>os.push(r.map((c,j)=>({text:c,options:{
  align:j===0?'left':'left', bold:j===0, color:GRAPH,
  fill:{color:i%2?WHITE:MIST}, fontSize:9.5}}))));
s.addTable(os,{...tblBase,x:M,y:2.06,w:W-2*M,colW:[2.35,2.72,2.42,3.16,1.44],
  rowH:[0.3,0.42,0.42,0.42,0.72,0.42,0.42]});
s.addShape(p.ShapeType.roundRect,{x:M,y:5.66,w:W-2*M,h:1.1,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.05});
s.addText('Four of the six run OpenEvolve’s converter or its search, and not one of those four states a problem size. Only the AlgoTune leaderboard is known to sit at the calibrated n — which is why this deck’s comparison against those eighteen models is the one it puts weight on, and the three OpenEvolve-derived rows the ones it flags.',
  {x:M+0.26,y:5.76,w:W-2*M-0.52,h:0.9,fontFace:B,fontSize:12,color:'D6E0E4',
   lineSpacing:17,isTextBox:true,margin:0});
s.addNotes('Checked and not on AlgoTune: CodeEvolve (AlphaEvolve suite), ThetaEvolve (circle packing), ParEVO (PBBS/ParEval), RL4RLA, ProgramBench.');

/* ---------------------------------------------------- 5  per-task */
s=p.addSlide();
head(s,'PER TASK','All eight, nothing selected',
  'Both our seeds shown. The last two columns are the spread across AlgoTune’s own 14–18 models at the same n.');
const T=D.tasks;
const r5=[[hdr('Task'),hdr('n'),hdr('ours s0'),hdr('ours s1'),hdr('Qwen3-14B\nno RL'),hdr('Meta\nEvolve'),hdr('Open\nEvolve'),hdr('AlgoTune max'),hdr('AlgoTune median')]];
T.forEach((t,i)=>{
  const o=D.ours[t], q=D.published[t], u=D.upstream_per_task[t];
  const over=v=>v>u.max*1.001;
  r5.push([
   {text:t,options:{bold:true,align:'left'}},
   {text:String(D.n[t]),options:{align:'center',color:SLATE}},
   {text:o[0].toFixed(3),options:{align:'center',bold:true,color:VIRID}},
   {text:o[1].toFixed(3),options:{align:'center',bold:true,color:VIRID}},
   {text:q[0].toFixed(3),options:{align:'center',color:over(q[0])?BRASS:GRAPH,bold:over(q[0])}},
   {text:q[1].toFixed(3),options:{align:'center',color:over(q[1])?BRASS:GRAPH,bold:over(q[1])}},
   {text:q[2].toFixed(2),options:{align:'center',color:over(q[2])?BRASS:GRAPH,bold:over(q[2])}},
   {text:u.max.toFixed(3),options:{align:'center'}},
   {text:u.median.toFixed(3),options:{align:'center',color:SLATE}},
  ].map(c=>({...c,options:{...c.options,fill:{color:i%2?WHITE:MIST},fontSize:11}})));
});
r5.push([{text:'harmonic mean, raw',options:{bold:true,align:'left',fill:{color:INK},color:WHITE}},
 ...['—',D.ours_hm[0].toFixed(3),D.ours_hm[1].toFixed(3),'1.392','2.045','1.984','—','—']
   .map((v,j)=>({text:v,options:{align:'center',bold:true,fill:{color:INK},
     color:(j===1||j===2)?'6FD9BF':WHITE,fontSize:11}}))]);
r5.push([{text:'AlgoTune score (clipped)',options:{bold:true,align:'left',fill:{color:INK},color:WHITE}},
 ...['—',D.ours_score[0].toFixed(3),D.ours_score[1].toFixed(3),'1.392','2.045','1.984','—','—']
   .map((v,j)=>({text:v,options:{align:'center',bold:true,fill:{color:INK},
     color:(j===1||j===2)?'6FD9BF':WHITE,fontSize:11}}))]);
s.addTable(r5,{...tblBase,x:M,y:1.92,w:W-2*M,colW:[2.72,0.95,1.02,1.02,0.95,0.95,1.0,1.36,2.12],
  rowH:0.345});
s.addText('amber = above the best of all 14–18 AlgoTune models at this n.  “Clipped” is AlgoTune’s own rule: anything under 1.0 counts as 1.0.  The OpenEvolve column is that project’s cross-phase discoveries log rather than one run, so its per-task cells and its published 1.984x score do not come from the same configuration.',
  {x:M,y:HT-1.06,w:W-2*M,h:0.34,fontFace:B,fontSize:9.5,color:BRASS,lineSpacing:12,isTextBox:true,margin:0});
foot(s,'Max and median use every model with a result on that task. The nine with no score on at least one of the eight — claude-opus-4, 4.5 and 4.6, sonnet-4.5, gemini-2.5-pro, gemini-3-pro, gemini-3.1-pro, gpt-oss-120b, qwen3-coder — still have solver files upstream; only the score is missing, and they include the newest models on the board. AlgoTune publishes no GLM-5 result at all.');
s.addNotes('The amber cells are the whole argument of slide 7.');

/* ---------------------------------------------------- 6  reproducibility */
s=p.addSlide();
head(s,'WHAT THE BUDGET BOUGHT','Not a better score — a repeatable one',
  'Going from 45 to 99 rollouts moved the level 4%. It moved the variance by a factor of forty.');
const cmp=[[hdr(''),hdr('45 rollouts'),hdr('99 rollouts')],
 ['seed-to-seed spread, median','1.38x','1.02x'],
 ['seed-to-seed spread, worst','180.11x','3.97x'],
 ['aggregate, seed 0','3.645x','2.285x'],
 ['aggregate, seed 1','2.213x','2.254x'],
 ['gap between the seeds','65%','1.4%']];
const r6=[cmp[0]]; cmp.slice(1).forEach((r,i)=>r6.push(r.map((c,j)=>({text:c,
  options:{align:j?'center':'left',bold:j===2,color:j===2?VIRID:GRAPH,
  fill:{color:i%2?WHITE:MIST},fontSize:12}}))));
s.addTable(r6,{...tblBase,x:M,y:2.0,w:6.35,colW:[3.05,1.65,1.65],rowH:0.46});
[['polynomial_real','540.172 / 2.999','544.454 / 137.005'],
 ['lu_factorization','4.464 / 0.957','7.000 / 7.139']].forEach((r,i)=>{
  const y=2.05+i*1.32;
  s.addShape(p.ShapeType.roundRect,{x:7.35,y,w:5.36,h:1.14,fill:{color:MIST},
    line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
  s.addText(r[0],{x:7.6,y:y+0.12,w:4.9,h:0.28,fontFace:B,fontSize:11.5,bold:true,
    color:INK,isTextBox:true,margin:0});
  s.addText([{text:'45 rollouts   ',options:{color:SLATE}},{text:r[1],options:{color:GRAPH,bold:true}}],
    {x:7.6,y:y+0.44,w:4.9,h:0.26,fontFace:B,fontSize:11,isTextBox:true,margin:0});
  s.addText([{text:'99 rollouts   ',options:{color:SLATE}},{text:r[2],options:{color:VIRID,bold:true}}],
    {x:7.6,y:y+0.74,w:4.9,h:0.26,fontFace:B,fontSize:11,isTextBox:true,margin:0});
});
s.addShape(p.ShapeType.roundRect,{x:7.35,y:4.72,w:5.36,h:1.6,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.06});
s.addText('At 45 rollouts the same configuration returned 3.645x on one seed and 2.213x on another. That was a draw, not a measurement. Two tasks were coin flips on whether the search found the direction at all; at 99 it finds them every time.',
  {x:7.62,y:4.9,w:4.82,h:1.26,fontFace:B,fontSize:12,color:'D6E0E4',lineSpacing:17,
   isTextBox:true,margin:0});
foot(s,'Same configuration throughout; only --iterations and the staleness policy changed.');
s.addNotes('This is why the comparison is now worth making at all.');

/* ---------------------------------------------------- 7  comparability */
s=p.addSlide();
head(s,'THE FINDING','Half the comparison is not the same problem',
  'AlgoTune’s own 14–18 models, at the calibrated n, bracket what is achievable. Three systems fall outside the bracket — in both directions.');
const A=[['Above the best of every AlgoTune model',BRASS,
  [['eigenvectors_complex','max 1.039x over 18 models','no-RL 1.432 · MetaEvolve 1.474 · OpenEvolve 1.48'],
   ['affine_transform_2d','max 1.015x over 14 models','MetaEvolve 6.945 · OpenEvolve 3.22'],
   ['fft_convolution','max 1.021x over 15 models','MetaEvolve 1.346 · OpenEvolve 1.38']]],
 ['Below AlgoTune’s median',SLATE,
  [['psd_cone_projection','median 8.728x over 18 models','no-RL 1.795 · MetaEvolve 1.914 · OpenEvolve 1.94'],
   ['convolve2d_full_fill','median 144.938x over 18 models','MetaEvolve 78.128'],
   ['',' ',' ']]]];
A.forEach((col,ci)=>{
  const x=M+ci*6.15;
  s.addText(col[0],{x,y:2.02,w:5.9,h:0.3,fontFace:B,fontSize:12.5,bold:true,color:col[1],
    charSpacing:0.6,isTextBox:true,margin:0});
  col[2].forEach((r,i)=>{
    if(!r[0]) return;
    const y=2.44+i*1.06;
    s.addShape(p.ShapeType.roundRect,{x,y,w:5.9,h:0.92,fill:{color:MIST},
      line:{color:RULE,width:0.75},rectRadius:0.06});
    s.addText(r[0],{x:x+0.24,y:y+0.1,w:5.45,h:0.26,fontFace:B,fontSize:11.5,bold:true,
      color:INK,isTextBox:true,margin:0});
    s.addText('AlgoTune  '+r[1],{x:x+0.24,y:y+0.36,w:5.45,h:0.24,fontFace:B,fontSize:10.5,
      color:SLATE,isTextBox:true,margin:0});
    s.addText(r[2],{x:x+0.24,y:y+0.6,w:5.45,h:0.24,fontFace:B,fontSize:10.5,bold:true,
      color:col[1],isTextBox:true,margin:0});
  });
});
s.addShape(p.ShapeType.roundRect,{x:M,y:5.66,w:W-2*M,h:1.06,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.06});
s.addText('That two-sided pattern is the signature of a smaller problem, not a better search. Shrink n and the tasks whose only headroom is Python overhead become winnable; the tasks whose win is genuinely algorithmic at scale lose theirs. All three columns run OpenEvolve’s search, and three of the eight agree with OpenEvolve’s own numbers to within 2.5%.',
  {x:M+0.28,y:5.82,w:W-2*M-0.56,h:0.76,fontFace:B,fontSize:13,color:'D6E0E4',
   lineSpacing:19,isTextBox:true,margin:0});
s.addNotes('Both directions matter. One direction alone would be explainable as a stronger search.');

/* ---------------------------------------------------- 8  the proof */
s=p.addSlide();
head(s,'THE PROOF CASE  ·  EIGENVECTORS_COMPLEX','The ceiling at n = 463 is 1.103x',
  'One task, taken apart, because it is the one where all three published numbers exceed what any model achieves.');
s.addShape(p.ShapeType.roundRect,{x:M,y:1.98,w:6.0,h:2.06,fill:{color:MIST},
  line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
s.addText('Where the reference’s 123.20 ms goes',{x:M+0.26,y:2.12,w:5.5,h:0.3,
  fontFace:B,fontSize:12,bold:true,color:INK,isTextBox:true,margin:0});
[['np.linalg.eig — which every valid answer must call','111.71 ms','93%'],
 ['the sort, the normalise, the .tolist()','11.5 ms','7%']].forEach((r,i)=>{
  const y=2.52+i*0.44;
  s.addText(r[0],{x:M+0.26,y,w:3.9,h:0.3,fontFace:B,fontSize:11,color:GRAPH,valign:'middle',isTextBox:true,margin:0});
  s.addText(r[1],{x:M+4.2,y,w:1.0,h:0.3,fontFace:B,fontSize:11,color:GRAPH,align:'right',valign:'middle',isTextBox:true,margin:0});
  s.addText(r[2],{x:M+5.2,y,w:0.55,h:0.3,fontFace:B,fontSize:11,bold:true,color:i?SLATE:VIRID,align:'right',valign:'middle',isTextBox:true,margin:0});
});
s.addText('Ceiling 1.103x — and .tolist() is not optional: is_solution rejects an ndarray. Best valid program: 1.053x. This port: 1.039x.',
  {x:M+0.26,y:3.40,w:5.5,h:0.56,fontFace:B,fontSize:11,color:INK,lineSpacing:15,isTextBox:true,margin:0});
s.addShape(p.ShapeType.roundRect,{x:6.9,y:1.98,w:5.81,h:2.06,fill:{color:MIST},
  line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
s.addText('Two escapes, both closed',{x:7.16,y:2.12,w:5.3,h:0.3,fontFace:B,fontSize:12,
  bold:true,color:INK,isTextBox:true,margin:0});
s.addText([
 {text:'More cores. ',options:{bold:true,color:INK}},
 {text:'np.linalg.eig barely threads — 116.81 ms on 1 core, 110.54 ms on 4. Four times the cores buys 5.7%.\n',options:{color:GRAPH}},
 {text:'Faster hardware. ',options:{bold:true,color:INK}},
 {text:'AlgoTune calibrates n so the reference takes ~100 ms on your machine, so a faster LAPACK raises n, not the ceiling.',options:{color:GRAPH}}],
 {x:7.16,y:2.5,w:5.3,h:1.42,fontFace:B,fontSize:11,lineSpacing:15,isTextBox:true,margin:0});
const r8=[[hdr('n'),hdr('reference'),hdr('eig alone'),hdr('eig share'),hdr('best valid'),hdr('ceiling')]];
[[20,'0.200 ms','0.106 ms','53%','1.710x','1.889x'],
 [50,'1.144 ms','0.812 ms','71%','1.354x','1.410x'],
 [100,'6.142 ms','5.575 ms','91%','1.069x','1.102x'],
 [200,'20.776 ms','18.929 ms','91%','1.057x','1.098x'],
 [463,'120.430 ms','109.215 ms','91%','1.053x','1.103x']].forEach((r,i)=>{
  const cal=r[0]===463;
  r8.push(r.map((c,j)=>({text:String(c),options:{align:'center',
    bold:cal||j===0, color:cal?VIRID:GRAPH,
    fill:{color:cal?'E6F2EF':(i%2?WHITE:MIST)},fontSize:11}})));
});
s.addTable(r8,{...tblBase,x:M,y:4.28,w:6.0,colW:[0.62,1.2,1.14,0.9,1.05,1.09],rowH:0.318});
s.addShape(p.ShapeType.roundRect,{x:6.9,y:4.28,w:5.81,h:1.9,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.06});
s.addText('1.432x, 1.474x and 1.48x all sit in the n ≈ 50 band.',
  {x:7.16,y:4.42,w:5.3,h:0.62,fontFace:H,fontSize:15,bold:true,color:WHITE,lineSpacing:21,isTextBox:true,margin:0});
s.addText('Not wrong numbers — answers to a different question. Ours answers “how much is there at n = 463”; theirs answers “how much is there when the Python overhead is still a third of the work”. Putting them in one column compares two different tasks.',
  {x:7.16,y:5.06,w:5.3,h:1.0,fontFace:B,fontSize:11.5,color:'D6E0E4',lineSpacing:16,isTextBox:true,margin:0});
foot(s,'Measured on the same 4-core box the searches ran on; reference and candidate timed interleaved, min of 6–9 runs.');
s.addNotes('The two escapes are what makes this a conclusion rather than a suspicion. Say so.');

/* ---------------------------------------------------- 9  headroom */
s=p.addSlide();
head(s,'WHERE WE ACTUALLY STAND','Against AlgoTune’s own best, task by task',
  'The right yardstick is not the papers — it is the 14–18 models measured at the same n.');
const hr=T.map(t=>({t,ours:D.ours[t][0],max:D.upstream_per_task[t].max}))
  .sort((a,b)=>(b.max/b.ours)-(a.max/a.ours));
hr.forEach((r,i)=>{
  const y=1.98+i*0.575, ratio=r.max/r.ours, open=ratio>1.2;
  s.addShape(p.ShapeType.roundRect,{x:M,y,w:W-2*M,h:0.5,
    fill:{color:open?MIST:'E6F2EF'},line:{color:RULE,width:0.6},rectRadius:0.05});
  s.addText(r.t,{x:M+0.24,y:y+0.02,w:3.0,h:0.46,fontFace:B,fontSize:11.5,bold:true,
    color:INK,valign:'middle',isTextBox:true,margin:0});
  chip(s,M+3.3,y+0.13,'n = '+D.n[r.t],1.15);
  s.addText('ours '+r.ours.toFixed(3)+'x',{x:M+4.65,y:y+0.02,w:1.5,h:0.46,fontFace:B,
    fontSize:11.5,bold:true,color:VIRID,valign:'middle',isTextBox:true,margin:0});
  s.addText('AlgoTune best '+r.max.toFixed(3)+'x',{x:M+6.3,y:y+0.02,w:2.5,h:0.46,
    fontFace:B,fontSize:11.5,color:GRAPH,valign:'middle',isTextBox:true,margin:0});
  s.addText(open?(ratio.toFixed(1)+'x of headroom left'):'at or above the field’s best',
    {x:M+8.95,y:y+0.02,w:3.0,h:0.46,fontFace:B,fontSize:11.5,bold:true,
     color:open?BRASS:VIRID,valign:'middle',isTextBox:true,margin:0});
});
s.addText('Our own outlier, stated: polynomial_real at 544x is 3.9x above the best of 14 published models, and fft_cmplx_scipy_fftpack at 4.759x is above the best of 18. Both were re-verified against upstream’s own is_solution, but “far above every published result” is a claim that deserves its own scrutiny, not a victory lap.',
  {x:M,y:6.60,w:W-2*M,h:0.62,fontFace:B,fontSize:10.5,color:BRASS,lineSpacing:14,isTextBox:true,margin:0});
s.addNotes('Correcting an earlier read: affine is nearly closed, lu and psd are where the room is.');

/* ---------------------------------------------------- 10 other tasks */
s=p.addSlide();
head(s,'BEYOND THE EIGHT','The other tasks run with the model prior',
  'Ten tasks have a prior-arm run. The two outside the comparison set have no MetaEvolve number, so AlgoTune is the only yardstick.');
const extra=[['lp_centering',1.001,1.039,1.000,12,'At the field’s median and within 4% of its best. Nothing here for anyone.'],
 ['least_squares',null,3.810,1.910,13,'A genuine miss: the search never found a valid improvement, while half the field is above 1.9x.']];
extra.forEach((r,i)=>{
  const y=2.1+i*1.6;
  s.addShape(p.ShapeType.roundRect,{x:M,y,w:W-2*M,h:1.4,fill:{color:MIST},
    line:{color:RULE,width:0.75},rectRadius:0.06,shadow:shadow()});
  s.addText(r[0],{x:M+0.3,y:y+0.16,w:3.1,h:0.34,fontFace:B,fontSize:14,bold:true,
    color:INK,isTextBox:true,margin:0});
  s.addText(r[1]===null?'no valid\nimprovement':r[1].toFixed(3)+'x',
    {x:M+0.3,y:y+0.56,w:3.1,h:0.68,fontFace:H,fontSize:r[1]===null?15:26,bold:true,
     color:r[1]===null?BRASS:VIRID,isTextBox:true,margin:0});
  s.addText([{text:'AlgoTune best  ',options:{color:SLATE}},
             {text:r[2].toFixed(3)+'x',options:{color:GRAPH,bold:true}},
             {text:'      median  ',options:{color:SLATE}},
             {text:r[3].toFixed(3)+'x',options:{color:GRAPH,bold:true}},
             {text:'      over '+r[4]+' models',options:{color:SLATE}}],
    {x:M+3.7,y:y+0.24,w:8.2,h:0.3,fontFace:B,fontSize:11.5,isTextBox:true,margin:0});
  s.addText(r[5],{x:M+3.7,y:y+0.66,w:8.2,h:0.56,fontFace:B,fontSize:12.5,color:INK,
    lineSpacing:17,isTextBox:true,margin:0});
});
s.addShape(p.ShapeType.roundRect,{x:M,y:5.42,w:W-2*M,h:1.28,fill:{color:INK},
  line:{color:INK,width:0},rectRadius:0.06});
s.addText('The port has results on 29 AlgoTune tasks in all, but 19 of them predate the model prior and are a different configuration. Only these ten were run with it, so only these ten are shown — mixing arms is exactly the best-of column this deck replaced.',
  {x:M+0.28,y:5.6,w:W-2*M-0.56,h:0.94,fontFace:B,fontSize:13,color:'D6E0E4',
   lineSpacing:19,isTextBox:true,margin:0});
s.addNotes('Show the miss. A deck that only shows wins is not a measurement.');

/* ---------------------------------------------------- 11 conclusion */
s=p.addSlide(); dark(s);
s.addShape(p.ShapeType.rect,{x:0,y:0,w:W,h:HT,fill:{color:INK}});
s.addText('WHAT THIS DECK CLAIMS',{x:M,y:0.6,w:W-2*M,h:0.3,fontFace:B,fontSize:11,
  bold:true,color:VIRID,charSpacing:2.2,isTextBox:true,margin:0});
s.addText('Three things, and one it does not',{x:M,y:0.94,w:W-2*M,h:0.6,fontFace:H,
  fontSize:33,bold:true,color:WHITE,isTextBox:true,margin:0});
[['01','The number is reproducible.','2.285x and 2.254x on two seeds of one configuration, 1.4% apart, with identical win counts against both papers. At 45 rollouts the same configuration swung 65% between seeds. That is what the extra budget bought — not a better score, a repeatable one.'],
 ['02','Half the published comparison is not the same problem.','On three tasks all three systems report above the best of every AlgoTune model at the calibrated n; on two others they fall below AlgoTune’s median. For eigenvectors_complex the ceiling at n = 463 is 1.103x and cannot be raised by cores or hardware. Their numbers sit in the n ≈ 50 band.'],
 ['03','Measured against the right yardstick, the room is elsewhere.','AlgoTune’s own models at the same n put the headroom in lu_factorization (5.0x) and psd_cone_projection (2.0x) — not in the three tasks that look like failures. eigenvectors_complex is at 99% of its physical ceiling.']]
.forEach((r,i)=>{
  const y=1.86+i*1.44;
  s.addText(r[0],{x:M,y,w:0.8,h:0.5,fontFace:H,fontSize:26,bold:true,color:VIRID,isTextBox:true,margin:0});
  s.addText(r[1],{x:M+0.95,y:y+0.04,w:11.3,h:0.32,fontFace:B,fontSize:14.5,bold:true,
    color:WHITE,isTextBox:true,margin:0});
  s.addText(r[2],{x:M+0.95,y:y+0.42,w:11.3,h:0.86,fontFace:B,fontSize:12,color:'B9C6CD',
    lineSpacing:17,isTextBox:true,margin:0});
});
s.addText('And one it does not: that this port leads the field. Under AlgoTune’s own scoring it is 1st and 2nd of 23 on these eight tasks, but the three rows above the AlgoTuner field all run OpenEvolve’s search at sizes none of them states. lu_factorization at 7.0x is the reference’s .tolist() serialisation, not a faster factorisation; discount it to 1.19x and the eight-task aggregate is 1.905x. Three tasks were run at sizes we can only infer, and one seed each is still one seed.',
  {x:M,y:6.20,w:W-2*M,h:0.86,fontFace:B,fontSize:12,color:BRASS,lineSpacing:17,isTextBox:true,margin:0});
s.addNotes('End on the limits. The credibility of slides 4-9 depends on saying this one out loud.');

p.writeFile({fileName:'/tmp/claude-0/-home-user-agentdescent/fb2fcee5-712c-5980-96a9-578815787660/scratchpad/deck/algotune-eight-tasks.pptx'})
 .then(f=>console.log('wrote',f));
