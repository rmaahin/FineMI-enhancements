"""Measure how well concurrent processes share this GPU. Usage: bench_parallel.py [worker]"""
import os
import subprocess
import sys
import time


def worker():
    import torch, torch.nn as nn, torch.optim as optim
    import fine_mi as F
    device, _ = F.setup_determinism()
    m = F.build_model('cwt', 62).to(device)
    m.apply(F.init_weights)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 62, F.N_FREQS, F.T_OUT, device=device)
    y = torch.randint(0, 2, (32,), device=device)
    for _ in range(10):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{dt:.4f} {torch.cuda.max_memory_allocated()/1e9:.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker()
    else:
        py = sys.executable
        here = os.path.dirname(os.path.abspath(__file__))
        base = None
        print(f"{'procs':>6} {'wall(s)':>9} {'steps/s':>9} {'speedup':>8} {'VRAM/proc':>10}")
        for n in (1, 2, 4, 6):
            t0 = time.perf_counter()
            procs = [subprocess.Popen([py, __file__, 'worker'], cwd=here,
                                      stdout=subprocess.PIPE, text=True) for _ in range(n)]
            outs = [p.communicate()[0].strip().split() for p in procs]
            wall = time.perf_counter() - t0
            if any(len(o) < 2 for o in outs):
                print(f"{n:>6}  FAILED (likely OOM)")
                break
            vram = max(float(o[1]) for o in outs)
            total_steps = 200 * n
            thru = total_steps / wall
            if base is None:
                base = thru
            print(f"{n:>6} {wall:>9.2f} {thru:>9.1f} {thru/base:>7.2f}x {vram:>9.2f}G")
