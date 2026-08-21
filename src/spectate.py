#!/usr/bin/env python3
"""Watch a match in a browser, live or as a replay.

    python3 src/spectate.py                    # newest match, live
    python3 src/spectate.py <match-id>
    python3 src/spectate.py <match-id> --replay --speed 4

Standalone and read-only, deliberately decoupled from the arena: a wedged or
crashed viewer can never affect a match, and it can be restarted mid-fight.

Binds 127.0.0.1 ONLY. The feed must never be reachable from inside the arena -
an agent that could poll it would see its opponent's every command and its
output, which would destroy reconnaissance as a skill. The battle network is
--internal, so containers cannot reach host loopback.

Match state is not reimplemented here: it comes from watch.MatchState, whose
apply() already reconstructs a whole match from the event stream.
"""

import argparse
import json
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ROOT / "matches"


def read_events(path, start=0):
    """Yield (position, event) from a JSONL file, resumable.

    readline() rather than iteration: tell() is disabled inside a for-loop over
    a text file, so an earlier version of this pattern re-read line 1 forever.
    """
    events, pos = [], start
    if not path.exists():
        return events, pos
    with path.open("r", errors="replace") as fh:
        fh.seek(pos)
        while True:
            line = fh.readline()
            if not line or not line.endswith("\n"):
                break
            pos = fh.tell()
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events, pos


def snapshot(path):
    """Full current state, so a viewer joining mid-match renders immediately
    instead of replaying the whole stream."""
    state = watch.MatchState()
    last_seq = 0
    for event in read_events(path)[0]:
        state.apply(event)
        last_seq = max(last_seq, event.get("seq") or 0)
    return {
        "mode": state.mode, "models": state.models, "round": state.round,
        "elapsed": state.elapsed, "bank_granted": state.bank_granted,
        "agents": state.agents, "finished": state.finished,
        "elapsed_at_snapshot": state.elapsed,
        # Everything up to here is already in the page, so the stream must
        # resume AFTER it - otherwise the viewer sees the whole match twice.
        "last_seq": last_seq,
        "feed": [{"t": when, "text": strip_ansi(line), "cls": cls}
                 for when, line, cls in state.feed[-40:]],
        "terrain": dict(state.terrain),
        "attacks": [{"t": t, "from": src, "kind": kind}
                    for t, src, _, kind in state.attacks[-20:]],
    }


def strip_ansi(text):
    out, skipping = [], False
    for char in text:
        if char == "\033":
            skipping = True
        elif skipping:
            if char.isalpha():
                skipping = False
        else:
            out.append(char)
    return "".join(out)


class Spectator(BaseHTTPRequestHandler):
    server_version = "DuelBenchSpectator/1.0"
    events_path = None
    replay = False
    speed = 1.0

    def do_GET(self):
        if self.path == "/":
            # The snapshot is INLINED rather than fetched. A fetch races the
            # first paint, so a viewer opening mid-match - or a screenshot -
            # sees an empty board until the stream catches up. "</" is escaped
            # so match text can never close the script element early.
            boot = json.dumps(snapshot(self.events_path)).replace("</", "<\\/")
            page = PAGE.replace("/*BOOTSTRAP*/null", boot)
            return self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
        if self.path == "/state":
            return self._send(200, "application/json",
                              json.dumps(snapshot(self.events_path)).encode())
        if self.path.startswith("/events"):
            return self._stream()
        self._send(404, "text/plain", b"not found")

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _since(self):
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        try:
            return int(urllib.parse.parse_qs(query).get("from", ["0"])[0])
        except (ValueError, TypeError):
            return 0

    def _stream(self):
        """Server-Sent Events. One connection per viewer; ThreadingHTTPServer
        gives each its own thread."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            since = self._since()
            if self.replay:
                self._stream_replay(since)
            else:
                self._stream_live(since)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _emit(self, event):
        # An agent controls its own stdout, so it can echo a record carrying any
        # payload under its own src. The Python reducer already refuses
        # arena-only events from the wrong source, but the browser applies this
        # stream directly - so a forged match_end would end the spectator's
        # match even though the terminal viewer was immune. Filter here rather
        # than re-implementing the rule in JavaScript: one place, one rule.
        if not watch.MatchState.trusted(event.get("event"), event.get("src")):
            return
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_replay(self, since=0):
        events, _ = read_events(self.events_path)
        previous = 0.0
        for event in events:
            if (event.get("seq") or 0) <= since:
                continue
            gap = (event.get("t") or 0.0) - previous
            previous = event.get("t") or previous
            if gap > 0 and self.speed > 0:
                time.sleep(min(gap / self.speed, 3.0))
            self._emit(event)

    def _stream_live(self, since=0):
        events, pos = read_events(self.events_path)
        for event in events:
            if (event.get("seq") or 0) <= since:
                continue          # already inlined into the page
            self._emit(event)
        idle_since = time.time()
        while True:
            fresh, pos = read_events(self.events_path, pos)
            for event in fresh:
                self._emit(event)
                if event.get("event") == "match_end":
                    return
            if fresh:
                idle_since = time.time()
            elif time.time() - idle_since > 900:
                return
            else:
                self.wfile.write(b": keepalive\n\n")   # hold the connection open
                self.wfile.flush()
            time.sleep(0.4)

    def log_message(self, fmt, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duel-Bench</title>
<style>
:root{--bg:#0b0e13;--panel:#131720;--line:#222b3a;--dim:#68728a;--fg:#e6ecf6;
      --a:#5cc8ff;--b:#ffcf5c;--good:#5ce49a;--bad:#ff6b6b;--warn:#ffa94d;
      --attack:#ff6b6b;--defense:#5ce49a;--recon:#68728a;--terrain:#4fa3ff;--err:#ff8a8a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:baseline;gap:18px;padding:12px 20px;
       border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0;letter-spacing:.14em;text-transform:uppercase}
.meta{color:var(--dim);font-size:12px}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
.vs{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:start;margin-bottom:16px}
@media(max-width:860px){.vs{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;
      border-top:3px solid var(--line)}
.card.a{border-top-color:var(--a)} .card.b{border-top-color:var(--b)}
.card.down{opacity:.55}
.role{font-weight:700;letter-spacing:.08em;font-size:15px}
.card.a .role{color:var(--a)} .card.b .role{color:var(--b)}
.model{color:var(--dim);font-size:12px;margin-top:2px;word-break:break-all}
.state{float:right;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.alive{color:var(--good)} .dead{color:var(--bad)}
.clock{margin:14px 0 6px;height:10px;background:#0a0c11;border-radius:5px;overflow:hidden}
.clock i{display:block;height:100%;background:var(--good);transition:width .4s linear}
.clock i.low{background:var(--warn)} .clock i.out{background:var(--bad)}
.nums{display:flex;justify-content:space-between;color:var(--dim);font-size:12px}
.secs{font-size:20px;color:var(--fg);font-variant-numeric:tabular-nums}
.cmd{margin-top:12px;background:#0a0c11;border:1px solid var(--line);border-radius:4px;
     padding:8px 10px;font-size:12px;min-height:34px;white-space:pre-wrap;word-break:break-all}
.cmd b{color:var(--dim);font-weight:400}
.cmdlist{margin-top:8px;display:flex;flex-direction:column;gap:4px}
.cmdlist .item{padding:3px 8px;border-left:2px solid;border-radius:2px;font-size:11.5px;
  white-space:pre-wrap;word-break:break-all}
.cmdlist .item.recon{border-color:var(--recon);color:var(--recon)}
.cmdlist .item.attack{border-color:var(--attack);color:#ffadb8}
.cmdlist .item.defense{border-color:var(--defense);color:#a9f0c2}
.cmdlist .item.error{border-color:var(--err);color:var(--err)}
.think{color:var(--a);font-size:12px;margin-top:8px}
.terrain{margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:12px}
.terrain .bar{display:flex;height:8px;background:#0a0c11;border-radius:4px;overflow:hidden;margin-top:6px}
.terrain .bar i{height:100%}
.terrain .bar .s{background:var(--good)}
.terrain .bar .p{background:var(--attack)}
.arrow{text-align:center;color:var(--dim);font-size:22px;padding:8px 0}
.filters{display:flex;gap:8px;margin:16px 0 8px;flex-wrap:wrap}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:16px;
      padding:3px 12px;font-size:12px;cursor:pointer;color:var(--dim)}
.chip.on{color:var(--fg);border-color:var(--a)}
.feed{margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:6px;
      padding:10px 14px;max-height:50vh;overflow:auto}
.feed div{padding:2px 0;border-bottom:1px solid #1a2029;font-size:12.5px;white-space:pre-wrap}
.feed div:last-child{border-bottom:0}
.t{color:var(--dim);display:inline-block;width:58px}
.cls-recon{color:var(--recon)} .cls-attack{color:var(--attack)}
.cls-defense{color:var(--defense)} .cls-terrain{color:var(--terrain)}
.cls-error{color:var(--err)} .cls-verdict{font-weight:700}
.verdict{margin-top:20px;padding:16px;border:1px solid var(--line);border-radius:8px;
         background:var(--panel);font-size:15px}
.verdict.win{border-color:var(--good)} .verdict.draw{border-color:var(--warn)}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;background:#0a0c11;
       color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.tercard{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.tercard .cell{background:#0a0c11;border:1px solid var(--line);border-radius:4px;padding:6px 10px}
.tercard .num{font-size:18px;font-variant-numeric:tabular-nums}
.health{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);font-size:11.5px}
.health .ok{color:var(--good)} .health .bad{color:var(--attack)}
.winbar{height:10px;background:#0a0c11;border-radius:5px;overflow:hidden;
        display:flex;margin:0 0 16px}
.winbar .left{background:var(--a);transition:width .5s linear}
.winbar .right{background:var(--b);transition:width .5s linear}
.winlabel{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);
         margin:0 2px 4px}
</style></head><body>
<header><h1>Duel-Bench</h1>
  <span class="meta" id="mode">—</span><span class="meta" id="round"></span>
  <span class="meta" id="clock"></span><span class="meta" id="conn"></span></header>
<div class="wrap">
  <div class="winlabel"><span id="win-a"></span><span id="win-mid"></span><span id="win-b"></span></div>
  <div class="winbar"><div class="left" id="wl"></div><div class="right" id="wr"></div></div>
  <div class="vs" id="vs">
    <div class="card a" id="card-agent-a"></div>
    <div class="arrow" id="arrow"></div>
    <div class="card b" id="card-agent-b"></div>
  </div>
  <div id="verdict"></div>
  <div class="filters" id="filters"></div>
  <div class="feed" id="feed"></div>
</div>
<script>
const S={agents:{},models:{},bank:null,mode:"—",round:null,t:0,done:null,
         terrain:{},attacks:[],filter:"all"};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
const agent=r=>S.agents[r]||(S.agents[r]={alive:true,cmds:0,bank:null,last:"",think:null,
                                          passes:0,forfeits:0,stop:null,lastcmds:[]});
const terr=r=>S.terrain[r]||(S.terrain[r]={score:0,spoofs:0});

function classify(cmd){
  const c=cmd.toLowerCase();
  if(c.includes("kill")||c.includes("pkill")||c.includes("nc ")||c.includes("flood"))return"attack";
  if(c.includes("curl")&&c.toUpperCase().includes("POST"))return"attack";
  if(c.includes("curl")&&(c.includes("/debug")||c.includes("/telemetry")))return"defense";
  if(c.includes("respawn")||c.includes("while true"))return"defense";
  return"recon";
}

function apply(e){
  S.t=Math.max(S.t,e.t||0);
  const r=e.agent||(String(e.src||"").startsWith("agent-")?e.src:null);
  switch(e.event){
    case "match_start":
      S.mode=e.mode; S.models={"agent-a":e.model_a,"agent-b":e.model_b};
      S.bank=(e.mode_config||{}).time_bank; note(e.t,"match starts — "+e.model_a+" vs "+e.model_b,"verdict"); break;
    case "go": note(e.t,"FIGHT","verdict"); break;
    case "move_start": if(r){const a=agent(r); a.think=0; if(e.bank_remaining!=null)a.bank=e.bank_remaining;
                       if(e.round)S.round=e.round;} break;
    case "thinking": if(r){const a=agent(r); a.think=e.elapsed;
                     if(e.bank_remaining!=null)a.bank=e.bank_remaining;} break;
    case "completion": if(r){const a=agent(r); a.think=null;
                       if(e.bank_remaining!=null)a.bank=e.bank_remaining;} break;
    case "command_start": if(r){const a=agent(r); a.cmds++; a.last=e.command||"";
      a.lastcmds.unshift({cmd:e.command||"",cls:classify(e.command||"")});
      if(a.lastcmds.length>3)a.lastcmds.pop();
      note(e.t,r+" $ "+(e.command||""),classify(e.command||""));} break;
    case "command_result": if(r&&e.exit_code!==0&&e.exit_code!=null)
                           note(e.t,"   "+r+" exit "+e.exit_code,"error"); break;
    case "terrain_defended": if(r){const t=terr(r); t.score=e.score||0;
                           note(e.t,r+" defends (score "+t.score+")","defense");} break;
    case "terrain_signal_hijacked": case "terrain_telemetry_flooded":
      if(r){const t=terr(r); t.spoofs=(t.spoofs||0)+1;
      S.attacks.push({t:e.t||S.t,from:r,kind:e.event}); if(S.attacks.length>20)S.attacks.shift();
      note(e.t,r+" "+e.event,"attack");} break;
    case "terrain_hit": if(r)note(e.t,r+": terrain hit ("+(e.endpoint||"")+")","terrain"); break;
    case "pass": if(r){agent(r).passes++; note(e.t,r+" passes","recon");} break;
    case "move_forfeit": if(r){agent(r).forfeits++; note(e.t,r+" forfeits the round","error");} break;
    case "bank_exhausted": if(r){agent(r).bank=0; note(e.t,r+" is out of time","error");} break;
    case "idle": if(r){agent(r).stop=e.reason; note(e.t,r+" stops: "+e.reason,"recon");} break;
    case "agent_down": if(e.agent){agent(e.agent).alive=false; note(e.t,e.agent+" is down ("+e.how+")","error");} break;
    case "snapshot":
      if(e.round)S.round=e.round;
      for(const[k,v]of Object.entries(e.agents||{})){const a=agent(k);
        if(v.alive!=null)a.alive=v.alive;
        if(v.commands_run!=null)a.cmds=Math.max(a.cmds,v.commands_run);
        if(v.stop_reason)a.stop=v.stop_reason;
        if(v.terrain){S.terrain[k]={score:v.terrain.score||0,spoofs:v.terrain.spoofs||0};}
        if(v.health)a.health=v.health;}
      for(const[k,v]of Object.entries(e.banks||{}))if(v!=null)agent(k).bank=v;
      break;
    case "match_end": S.done=e; note(e.t,e.outcome+" — "+e.winner,"verdict"); break;
  }
  render();
}

function note(t,text,cls){
  if(S.filter!=="all"&&cls!==S.filter&&cls!=="verdict")return;
  const d=document.createElement("div");
  d.className="cls-"+(cls||"recon");
  d.innerHTML='<span class="t">'+(t||0).toFixed(1)+'s</span>'+esc(text);
  const f=$("feed"); f.appendChild(d);
  while(f.children.length>300)f.removeChild(f.firstChild);
  f.scrollTop=f.scrollHeight;
}

function healthHtml(h){
  if(!h)return'<div class="health">health: no data</div>';
  const wr=h.battle_writable===true?"ok":(h.battle_writable===false?"bad":"");
  const free=h.free_bytes?(h.free_bytes/1048576).toFixed(1)+" MB":"?";
  const pids=h.pids!=null&&h.pid_limit?(h.pids+"/"+h.pid_limit):"?";
  const sf=h.spawn_failures_consecutive||0;
  return'<div class="health">health: '
    +'<span class="'+wr+'">'+(h.battle_writable===true?"writable":h.battle_writable===false?"BLOCKED":"?")+'</span>'
    +' · '+free+' free · '+pids+' pids'
    +(sf?' · <span class="bad">'+sf+' spawn failures</span>':'')
    +'</div>';
}

function winPct(role){
  const o=role==="agent-a"?"agent-b":"agent-a";
  let p=0.5;
  const ta=terr(role), to=terr(o);
  const a=agent(role), b=agent(o);
  if(!a.alive)return 0.0; if(!b.alive)return 1.0;
  // 40% each: both alive
  p=0.4;
  if(ta.spoofs<to.spoofs)p+=0.20; else if(to.spoofs<ta.spoofs)p-=0.20;
  if(a.cmds>b.cmds)p+=0.15; else if(b.cmds>a.cmds)p-=0.15;
  const ha=a.health||{}, hb=b.health||{};
  const okA=ha.battle_writable===true, okB=hb.battle_writable===true;
  if(okA&&!okB)p+=0.15; else if(!okA&&okB)p-=0.15;
  if(a.bank!=null&&b.bank!=null&&a.bank>b.bank)p+=0.10; else if(a.bank!=null&&b.bank!=null&&b.bank>a.bank)p-=0.10;
  return Math.max(0.02,Math.min(0.98,p));
}

function card(role){
  const a=agent(role), granted=S.bank, left=a.bank;
  const pct=granted&&left!=null?Math.max(0,Math.min(100,100*left/granted)):null;
  const cls=pct==null?"":(pct<=0?"out":pct<25?"low":"");
  const t=terr(role), tot=t.score+t.spoofs, sp=tot?100*t.score/tot:50, pp=tot?100*t.spoofs/tot:50;
  const cmdhtml=a.lastcmds.length?a.lastcmds.map(c=>'<div class="item '+c.cls+'"><b>$</b> '+esc(c.cmd)+'</div>').join("")
    :'<div class="cmd"><b>—</b></div>';
  return '<span class="state '+(a.alive?"alive":"dead")+'">'+(a.alive?"alive":"down")+'</span>'
    +'<div class="role">'+role+'</div><div class="model">'+esc(S.models[role]||"")+'</div>'
    +(pct!=null?'<div class="clock"><i class="'+cls+'" style="width:'+pct+'%"></i></div>'
       +'<div class="nums"><span class="secs">'+left.toFixed(1)+'s</span><span>'+a.cmds+' cmds'
       +(a.forfeits?' · '+a.forfeits+' ff':'')+(a.passes?' · '+a.passes+' pass':'')+'</span></div>'
      :'<div class="nums" style="margin-top:14px"><span class="secs">'+a.cmds+'</span><span>commands'
       +(a.forfeits?' · '+a.forfeits+' ff':'')+(a.passes?' · '+a.passes+' pass':'')+'</span></div>')
    +(a.think!=null?'<div class="think">thinking '+a.think.toFixed(0)+'s…</div>':'')
    +(a.stop?'<div class="think" style="color:var(--dim)">'+esc(a.stop)+'</div>':'')
    +'<div class="cmdlist">'+cmdhtml+'</div>'
    +'<div class="terrain">terrain: <b>'+t.score+'</b> defended · <b style="color:var(--attack)">'+t.spoofs+'</b> spoofed'
    +'<div class="bar"><i class="s" style="width:'+sp+'%"></i><i class="p" style="width:'+pp+'%"></i></div></div>'
    +healthHtml(a.health);
}

function render(){
  $("mode").textContent=S.mode;
  $("round").textContent=S.round?"round "+S.round:"";
  $("clock").textContent=S.t.toFixed(0)+"s";
  const arrows={};
  for(const atk of S.attacks.slice(-8))arrows[atk.from]=(arrows[atk.from]||0)+1;
  $("arrow").innerHTML=(arrows["agent-a"]?"← "+arrows["agent-a"]:"")+"<br>"+(arrows["agent-b"]?arrows["agent-b"]+" →":"");
  const wl=winPct("agent-a"), wr=1-winPct("agent-a");
  $("wl").style.width=(wl*100)+"%";
  $("wr").style.width=(wr*100)+"%";
  $("win-a").textContent="agent-a "+(wl*100).toFixed(0)+"%";
  $("win-b").textContent=(wr*100).toFixed(0)+"% agent-b";
  $("win-mid").textContent="win eval";
  for(const r of["agent-a","agent-b"]){
    const el=$("card-"+r); el.innerHTML=card(r);
    el.classList.toggle("down",!agent(r).alive);
  }
  if(S.done){const d=S.done, won=d.winner==="agent-a"||d.winner==="agent-b";
    const ta=terr("agent-a"), tb=terr("agent-b");
    $("verdict").innerHTML='<div class="verdict '+(won?"win":"draw")+'">'
      +'<span class="badge">'+esc(d.outcome)+'</span> '
      +(won?"<b>"+esc(d.winner)+"</b> wins":"draw")
      +' <span class="meta">in '+(d.duration||0)+'s</span>'
      +(d.rated?'':' <span class="badge">unrated: '+esc(d.unrated_reason||"")+'</span>')+'</div>'
      +'<div class="tercard"><div class="cell">agent-a: <span class="num" style="color:var(--good)">'+ta.score+'</span> defended · <span class="num" style="color:var(--attack)">'+ta.spoofs+'</span> spoofed</div>'
      +'<div class="cell">agent-b: <span class="num" style="color:var(--good)">'+tb.score+'</span> defended · <span class="num" style="color:var(--attack)">'+tb.spoofs+'</span> spoofed</div></div>';}
}

function filters(){
  const kinds=["all","attack","defense","terrain","error","recon"];
  $("filters").innerHTML=kinds.map(k=>'<span class="chip'+(S.filter===k?" on":"")+'" data-f="'+k+'">'+k+'</span>').join("");
  for(const el of document.querySelectorAll(".chip"))el.onclick=()=>{S.filter=el.dataset.f;
    $("feed").innerHTML=""; filters(); render();};
}
filters();

const BOOT = /*BOOTSTRAP*/null;
if(BOOT){
  S.mode=BOOT.mode||"—"; S.models=BOOT.models||{}; S.bank=BOOT.bank_granted;
  S.round=BOOT.round; S.t=BOOT.elapsed||0; S.done=BOOT.finished||null;
  S.terrain=BOOT.terrain||{}; S.attacks=BOOT.attacks||[];
  for(const[k,v]of Object.entries(BOOT.agents||{})){
    const a=agent(k);
    a.alive=v.alive!==false; a.cmds=v.commands||0; a.bank=v.bank;
    a.last=v.last||""; a.passes=v.passes||0; a.forfeits=v.forfeits||0;
    a.stop=v.stop_reason||null;
  }
  (BOOT.feed||[]).forEach(e=>note(e.t,e.text,e.cls||"recon"));
}
render();
const es=new EventSource("/events?from="+((BOOT&&BOOT.last_seq)||0));
es.onopen=()=>$("conn").textContent="● live";
es.onmessage=m=>apply(JSON.parse(m.data));
es.onerror=()=>{$("conn").textContent="○ disconnected"; es.close();};
</script></body></html>
"""


def resolve(target):
    if target:
        path = Path(target)
        if path.is_dir():
            return path
        guess = MATCHES / target
        if guess.is_dir():
            return guess
        return None
    candidates = [d for d in MATCHES.glob("*/") if (d / "events.jsonl").exists()]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="watch a Duel-Bench match in a browser")
    parser.add_argument("match", nargs="?", default=None)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    match_dir = resolve(args.match)
    if match_dir is None:
        sys.exit("no match found; pass a directory or id, or run a match first")
    stream = match_dir / "events.jsonl"
    if not stream.exists():
        sys.exit(f"{stream} does not exist yet")

    Spectator.events_path = stream
    Spectator.replay = args.replay
    Spectator.speed = args.speed
    # 127.0.0.1, never 0.0.0.0: nothing inside the arena may reach this.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Spectator)
    except OSError as exc:
        sys.exit(f"cannot listen on 127.0.0.1:{args.port}: {exc}\n"
                 f"Another viewer is probably already running — "
                 f"open http://127.0.0.1:{args.port}/ or pass --port.")
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{match_dir.name}  ({'replay' if args.replay else 'live'})  ->  {url}", flush=True)
    if not args.no_open:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
