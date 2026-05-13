"""
مقایسه‌ی زمان و حافظه‌ی هم‌ردیفی: MAFFT, ClustalW, PRANK و روش پیشنهادی
با استفاده از فایل 00-main_sac_parameter_correlation_heatmap - v2.py
"""
import os, sys, subprocess, time, tempfile, csv, threading, shutil

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil نصب نیست. حافظه N/A ثبت می‌شود.")

# مسیرها (همان‌هایی که در اختیار داشتید)
MAFFT_PATH = r"E:\PhD\1. first semester\Dr Mansouri\AE_DQN\Code_V3\Classic\mafft-win\mafft.bat"
CLUSTALW_PATH = r"C:\Program Files (x86)\ClustalW2\clustalw2.exe"
PRANK_PATH = r"E:\PhD\comparison\prank-msa-v.251117\binaries\previous_version\prank\bin\prank.exe"

# نام فایل روش پیشنهادی (همان که ارسال کردید)
PROPOSED_FILENAME = "00-main_sac_parameter_correlation_heatmap - v2.py"
PROPOSED_SCRIPT = os.path.join(os.path.dirname(__file__), PROPOSED_FILENAME)
PYTHON_EXE = sys.executable

INPUT_FOLDER = "input_compare"
OUTPUT_CSV = "runtime_memory_comparison.csv"
PRANK_PARAMS = ["-f=fasta", "-F", "-once"]

# توابع FASTA
def read_fasta(path):
    seqs = []
    with open(path) as f:
        curr = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr:
                    seqs.append(curr)
                    curr = ""
            else:
                curr += line
        if curr:
            seqs.append(curr)
    return seqs

def write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")

# اجرای خارجی با سنجش زمان و حافظه (استفاده از psutil)
def run_external(cmd, timeout=600, env=None, cwd=None):
    if not HAS_PSUTIL:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, cwd=cwd)
        out, err = p.communicate(timeout=timeout)
        runtime = time.perf_counter() - t0
        return out, err, runtime, None

    peak_mem = 0
    stop = threading.Event()
    def monitor(pid):
        nonlocal peak_mem
        try:
            proc = psutil.Process(pid)
            while not stop.is_set():
                try:
                    mem = proc.memory_info().rss
                    if mem > peak_mem:
                        peak_mem = mem
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                stop.wait(0.1)
        except:
            pass

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, cwd=cwd)
    pid = p.pid
    t = threading.Thread(target=monitor, args=(pid,), daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        out, err = p.communicate(timeout=timeout)
        runtime = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        runtime = timeout
    finally:
        stop.set()
        t.join(timeout=1)
    return out, err, runtime, peak_mem

# MAFFT
def run_mafft(seqs, timeout=600):
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.fasta")
        write_fasta(seqs, inp)
        out, err, t, mem = run_external([MAFFT_PATH, "--auto", inp], timeout)
        if err and "error" in err.lower():
            print(f"   MAFFT stderr: {err}")
        out_path = os.path.join(tmp, "out.fasta")
        with open(out_path, "w") as f:
            f.write(out)
        aligned = read_fasta(out_path)
        return aligned, t, mem

# ClustalW
def parse_clustal(path):
    d, ord_ = {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("CLUSTAL") or line.startswith("MUSCLE"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, frag = parts[0], parts[1]
            if all(c in "*:." for c in frag):
                continue
            if name not in d:
                d[name] = []
                ord_.append(name)
            d[name].append(frag)
    raw = {n: "".join(d[n]) for n in d}
    aln = [raw[n] for n in ord_]
    max_len = max(len(s) for s in aln)
    return [s + "-" * (max_len - len(s)) for s in aln]

def run_clustalw(seqs, timeout=600):
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.fasta")
        write_fasta(seqs, inp)
        aln_path = os.path.join(tmp, "in.aln")
        out, err, t, mem = run_external([CLUSTALW_PATH, "-INFILE=" + inp], timeout)
        if not os.path.exists(aln_path):
            print("   ClustalW output .aln not found")
            return None, None, None
        return parse_clustal(aln_path), t, mem

# PRANK با نمایش stdout و stderr کامل
def run_prank(fasta_path, timeout=1):
    prank_tmp = os.path.join(os.getcwd(), "prank_tmp")
    os.makedirs(prank_tmp, exist_ok=True)
    try:
        local = os.path.join(prank_tmp, "in.fasta")
        shutil.copy(fasta_path, local)
        pref = os.path.join(prank_tmp, "out")
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(PRANK_PATH) + os.pathsep + env["PATH"]
        cmd = [PRANK_PATH, f"-d={local}", f"-o={pref}"] + PRANK_PARAMS
        print(f"   PRANK cmd: {' '.join(cmd)}")
        out, err, t, mem = run_external(cmd, timeout, env=env)
        if err:
            print(f"   PRANK stderr: {err}")
        if out:
            print(f"   PRANK stdout (first 200 chars): {out[:200]}")
        best = pref + ".best.fas"
        if not os.path.exists(best):
            # شاید در دایرکتوری جاری ساخته شده
            alt_best = os.path.join(os.getcwd(), os.path.basename(best))
            if os.path.exists(alt_best):
                best = alt_best
            else:
                print("   PRANK output .best.fas not found in either tmp or current dir.")
                return None, None, None
        aligned = read_fasta(best)
        return aligned, t, mem
    finally:
        shutil.rmtree(prank_tmp, ignore_errors=True)

# روش پیشنهادی (اجرای فایل 00-main_sac_...)
def run_proposed(fasta_path, timeout=1800):
    if not os.path.exists(PROPOSED_SCRIPT):
        print(f"   ❌ فایل '{PROPOSED_SCRIPT}' یافت نشد. لطفاً نام فایل را بررسی کنید.")
        return None, None

    work_dir = tempfile.mkdtemp(prefix="proposed_")
    try:
        shutil.copy(fasta_path, os.path.join(work_dir, "input.fasta"))
        # اجرای اسکریپت
        out, err, t, mem = run_external(
            [PYTHON_EXE, PROPOSED_SCRIPT],
            timeout=timeout,
            cwd=work_dir,
            env={**os.environ, "PYTHONPATH": os.path.dirname(PROPOSED_SCRIPT)}
        )
        if err:
            print(f"   Proposed stderr:\n{err}")
        # زمان اجرا کل است
        return t, mem
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"پوشه {INPUT_FOLDER} وجود ندارد.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.fasta','.fa','.fna'))]
    if not files:
        print("هیچ فایل FASTA یافت نشد.")
        return

    results = []
    for fname in files:
        fpath = os.path.join(INPUT_FOLDER, fname)
        seqs = read_fasta(fpath)
        n = len(seqs)
        print(f"\n📄 {fname} – {n} توالی")

        # MAFFT
        print("   ▶ MAFFT ...")
        _, t_m, mem_m = run_mafft(seqs)
        if t_m:
            print(f"   ✅ MAFFT: {t_m:.2f}s, حافظه: {mem_m/1024/1024:.1f}MB" if mem_m else f"   ✅ MAFFT: {t_m:.2f}s")
        else:
            print("   ❌ MAFFT failed")

        # ClustalW
        print("   ▶ ClustalW ...")
        _, t_c, mem_c = run_clustalw(seqs)
        if t_c:
            print(f"   ✅ ClustalW: {t_c:.2f}s, حافظه: {mem_c/1024/1024:.1f}MB" if mem_c else f"   ✅ ClustalW: {t_c:.2f}s")
        else:
            print("   ❌ ClustalW failed")

        # PRANK
        print("   ▶ PRANK ...")
        _, t_p, mem_p = run_prank(fpath)
        if t_p:
            print(f"   ✅ PRANK: {t_p:.2f}s, حافظه: {mem_p/1024/1024:.1f}MB" if mem_p else f"   ✅ PRANK: {t_p:.2f}s")
        else:
            print("   ❌ PRANK failed (check messages above)")

        # Proposed
        print("   ▶ روش پیشنهادی (MOFSACEO-MSA) ...")
        t_pp, mem_pp = run_proposed(fpath)
        if t_pp:
            print(f"   ✅ Proposed: {t_pp:.2f}s, حافظه: {mem_pp/1024/1024:.1f}MB" if mem_pp else f"   ✅ Proposed: {t_pp:.2f}s")
        else:
            print("   ❌ Proposed failed")

        # ذخیره نتایج
        def mb(val):
            return round(val/(1024*1024), 2) if val else "N/A"
        results.append({
            "filename": fname,
            "n_seq": n,
            "mafft_time": f"{t_m:.2f}" if t_m else "FAIL",
            "mafft_mem_MB": mb(mem_m),
            "clustalw_time": f"{t_c:.2f}" if t_c else "FAIL",
            "clustalw_mem_MB": mb(mem_c),
            "prank_time": f"{t_p:.2f}" if t_p else "FAIL",
            "prank_mem_MB": mb(mem_p),
            "proposed_time": f"{t_pp:.2f}" if t_pp else "FAIL",
            "proposed_mem_MB": mb(mem_pp),
        })

    # نوشتن CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=[
            "filename","n_seq","mafft_time","mafft_mem_MB",
            "clustalw_time","clustalw_mem_MB","prank_time","prank_mem_MB",
            "proposed_time","proposed_mem_MB"
        ])
        w.writeheader()
        w.writerows(results)

    print("\n📊 گزارش نهایی در", OUTPUT_CSV)

if __name__ == "__main__":
    main()
