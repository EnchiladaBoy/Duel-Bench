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
    for event in read_events(path)[0]:
        state.apply(event)
    return {
        "mode": state.mode, "models": state.models, "round": state.round,
        "elapsed": state.elapsed, "bank_granted": state.bank_granted,
        "agents": state.agents, "finished": state.finished,
        "feed": [strip_ansi(line) for line in state.feed[-40:]],
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
            return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
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

    def _stream(self):
        """Server-Sent Events. One connection per viewer; ThreadingHTTPServer
        gives each its own thread."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            if self.replay:
                self._stream_replay()
            else:
                self._stream_live()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _emit(self, event):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_replay(self):
        events, _ = read_events(self.events_path)
        previous = 0.0
        for event in events:
            gap = (event.get("t") or 0.0) - previous
            previous = event.get("t") or previous
            if gap > 0 and self.speed > 0:
                time.sleep(min(gap / self.speed, 3.0))
            self._emit(event)

    def _stream_live(self):
        events, pos = read_events(self.events_path)
        for event in events:
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
:root{--bg:#0d0f14;--panel:#161a22;--line:#242a36;--dim:#7c869b;--fg:#e8ecf4;
      --a:#5cc8ff;--b:#ffcf5c;--good:#5ce49a;--bad:#ff6b6b;--warn:#ffa94d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:baseline;gap:16px;padding:14px 20px;
       border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0;letter-spacing:.14em;text-transform:uppercase}
.meta{color:var(--dim);font-size:12px}
.wrap{max-width:1100px;margin:0 auto;padding:20px}
.agents{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.agents{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;
      border-top:3px solid var(--line)}
.card.a{border-top-color:var(--a)} .card.b{border-top-color:var(--b)}
.card.down{opacity:.55}
.role{font-weight:700;letter-spacing:.08em} .card.a .role{color:var(--a)} .card.b .role{color:var(--b)}
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
.think{color:var(--a);font-size:12px;margin-top:8px}
.feed{margin-top:20px;background:var(--panel);border:1px solid var(--line);border-radius:6px;
      padding:12px 16px;max-height:44vh;overflow:auto}
.feed div{padding:2px 0;border-bottom:1px solid #1b2029;font-size:12.5px;white-space:pre-wrap}
.feed div:last-child{border-bottom:0}
.t{color:var(--dim);display:inline-block;width:58px}
.verdict{margin-top:20px;padding:16px;border:1px solid var(--line);border-radius:6px;
         background:var(--panel);font-size:15px}
.verdict.win{border-color:var(--good)} .verdict.draw{border-color:var(--warn)}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;background:#0a0c11;
       color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
</style></head><body>
<header><h1>Duel-Bench</h1>
  <span class="meta" id="mode">—</span><span class="meta" id="round"></span>
  <span class="meta" id="clock"></span><span class="meta" id="conn"></span></header>
<div class="wrap">
  <div class="agents">
    <div class="card a" id="card-agent-a"></div>
    <div class="card b" id="card-agent-b"></div>
  </div>
  <div id="verdict"></div>
  <div class="feed" id="feed"></div>
</div>
<script>
const S={agents:{},models:{},bank:null,mode:"—",round:null,t:0,done:null};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
const agent=r=>S.agents[r]||(S.agents[r]={alive:true,cmds:0,bank:null,last:"",think:null,
                                          passes:0,forfeits:0,stop:null});

function apply(e){
  S.t=Math.max(S.t,e.t||0);
  const r=e.agent||(String(e.src||"").startsWith("agent-")?e.src:null);
  switch(e.event){
    case "match_start":
      S.mode=e.mode; S.models={"agent-a":e.model_a,"agent-b":e.model_b};
      S.bank=(e.mode_config||{}).time_bank; note(e.t,"match starts — "+e.model_a+" vs "+e.model_b); break;
    case "go": note(e.t,"⚔ FIGHT"); break;
    case "move_start": if(r){const a=agent(r); a.think=0; if(e.bank_remaining!=null)a.bank=e.bank_remaining;
                       if(e.round)S.round=e.round;} break;
    case "thinking": if(r){const a=agent(r); a.think=e.elapsed;
                     if(e.bank_remaining!=null)a.bank=e.bank_remaining;} break;
    case "completion": if(r){const a=agent(r); a.think=null;
                       if(e.bank_remaining!=null)a.bank=e.bank_remaining;} break;
    case "command_start": if(r){const a=agent(r); a.cmds++; a.last=e.command||"";
                          note(e.t,r+" $ "+(e.command||""),r);} break;
    case "command_result": if(r&&e.exit_code!==0&&e.exit_code!=null)
                           note(e.t,"   "+r+" exit "+e.exit_code); break;
    case "pass": if(r){agent(r).passes++; note(e.t,r+" passes");} break;
    case "move_forfeit": if(r){agent(r).forfeits++; note(e.t,r+" forfeits the round — too slow");} break;
    case "bank_exhausted": if(r){agent(r).bank=0; note(e.t,"⏳ "+r+" is out of time");} break;
    case "idle": if(r){agent(r).stop=e.reason; note(e.t,r+" stops: "+e.reason);} break;
    case "agent_down": if(e.agent){agent(e.agent).alive=false; note(e.t,"💀 "+e.agent+" is down ("+e.how+")");} break;
    case "snapshot":
      if(e.round)S.round=e.round;
      for(const[k,v]of Object.entries(e.agents||{})){const a=agent(k);
        if(v.alive!=null)a.alive=v.alive;
        if(v.commands_run!=null)a.cmds=Math.max(a.cmds,v.commands_run);
        if(v.stop_reason)a.stop=v.stop_reason;}
      for(const[k,v]of Object.entries(e.banks||{}))if(v!=null)agent(k).bank=v;
      break;
    case "match_end": S.done=e; note(e.t,"■ "+e.outcome+" — "+e.winner); break;
  }
  render();
}

function note(t,text,role){
  const d=document.createElement("div");
  d.innerHTML='<span class="t">'+(t||0).toFixed(1)+'s</span>'+esc(text);
  if(role)d.style.color=role==="agent-a"?"var(--a)":"var(--b)";
  const f=$("feed"); f.appendChild(d);
  while(f.children.length>300)f.removeChild(f.firstChild);
  f.scrollTop=f.scrollHeight;
}

function card(role){
  const a=agent(role), granted=S.bank, left=a.bank;
  const pct=granted&&left!=null?Math.max(0,Math.min(100,100*left/granted)):null;
  const cls=pct==null?"":(pct<=0?"out":pct<25?"low":"");
  return '<span class="state '+(a.alive?"alive":"dead")+'">'+(a.alive?"alive":"down")+'</span>'
    +'<div class="role">'+role+'</div><div class="model">'+esc(S.models[role]||"")+'</div>'
    +(pct!=null?'<div class="clock"><i class="'+cls+'" style="width:'+pct+'%"></i></div>'
       +'<div class="nums"><span class="secs">'+left.toFixed(1)+'s</span><span>'+a.cmds+' cmds'
       +(a.forfeits?' · '+a.forfeits+' ff':'')+(a.passes?' · '+a.passes+' pass':'')+'</span></div>'
      :'<div class="nums" style="margin-top:14px"><span class="secs">'+a.cmds+'</span><span>commands'
       +(a.forfeits?' · '+a.forfeits+' ff':'')+(a.passes?' · '+a.passes+' pass':'')+'</span></div>')
    +'<div class="cmd">'+(a.last?'<b>$</b> '+esc(a.last):'<b>—</b>')+'</div>'
    +(a.think!=null?'<div class="think">thinking '+a.think.toFixed(0)+'s…</div>':'')
    +(a.stop?'<div class="think" style="color:var(--dim)">'+esc(a.stop)+'</div>':'');
}

function render(){
  $("mode").textContent=S.mode;
  $("round").textContent=S.round?"round "+S.round:"";
  $("clock").textContent=S.t.toFixed(0)+"s";
  for(const r of["agent-a","agent-b"]){
    const el=$("card-"+r); el.innerHTML=card(r);
    el.classList.toggle("down",!agent(r).alive);
  }
  if(S.done){const d=S.done, won=d.winner==="agent-a"||d.winner==="agent-b";
    $("verdict").innerHTML='<div class="verdict '+(won?"win":"draw")+'">'
      +'<span class="badge">'+esc(d.outcome)+'</span> '
      +(won?"<b>"+esc(d.winner)+"</b> wins":"draw")
      +' <span class="meta">in '+(d.duration||0)+'s</span>'
      +(d.rated?'':' <span class="badge">unrated: '+esc(d.unrated_reason||"")+'</span>')+'</div>';}
}

fetch("/state").then(r=>r.json()).then(s=>{
  S.mode=s.mode; S.models=s.models||{}; S.bank=s.bank_granted; S.round=s.round; render();
}).catch(()=>{});
const es=new EventSource("/events");
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
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Spectator)
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
