"""
MOFSACEO‑MSA with Deterministic Flexible Gap Representation
Fair comparison: same initial population (zero vector = MAFFT seed) for SAC and Fixed.
Generates: convergence, parameter evolution, and correlation heatmap.
"""
import os, subprocess, tempfile, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
from typing import List
import warnings
warnings.filterwarnings('ignore')

MAFFT_BAT = r"E:\PhD\1. first semester\Dr Mansouri\AE_DQN\Code_V3\Classic\mafft-win\mafft.bat"

SEED = 42                      # for reproducibility of other random ops

# ========================= Scoring Functions =========================
def sp_score(alignment, match=2, mismatch=-1, gap=-2):
    n = len(alignment); L = len(alignment[0]); score = 0
    for pos in range(L):
        col = [alignment[i][pos] for i in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                a, b = col[i], col[j]
                score += gap if (a == '-' or b == '-') else (match if a == b else mismatch)
    return score

def cs_score(alignment):
    n = len(alignment); L = len(alignment[0])
    if L == 0: return 0.0
    exact = sum(1 for pos in range(L)
                if len(set(c for c in (alignment[i][pos] for i in range(n)) if c != '-')) == 1)
    return exact / L

def gap_ratio(alignment):
    if not alignment: return 0.0
    n = len(alignment); L = len(alignment[0])
    return 100.0 * sum(s.count('-') for s in alignment) / (n * L) if L else 0.0

# ========================= FASTA & MAFFT =========================
def read_fasta(fp):
    seqs = []; cur = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur: seqs.append(''.join(cur)); cur = []
            else: cur.append(line.upper())
        if cur: seqs.append(''.join(cur))
    return seqs

def run_mafft_seed(seqs):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f_in:
        for i, s in enumerate(seqs): f_in.write(f">seq{i}\n{s}\n")
        inp = f_in.name
    out = inp + ".aln"
    cmd = [MAFFT_BAT, "--auto", inp]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not os.path.exists(out):
            with open(out, 'w') as f: subprocess.run(cmd, stdout=f, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"MAFFT error: {e.stderr.decode()}")
        return [s.ljust(max(len(x) for x in seqs), '-') for s in seqs]
    finally:
        if os.path.exists(inp): os.unlink(inp)
    aln = []; cur = []
    with open(out) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur: aln.append(''.join(cur)); cur = []
            else: cur.append(line.replace(' ', ''))
        if cur: aln.append(''.join(cur))
    if os.path.exists(out): os.unlink(out)
    return aln

# ========================= Deterministic Flexible Gap Representation =========================
class FlexibleGapRepresentation:
    def __init__(self, seed_alignment, max_shift=15):
        self.seed = [list(s) for s in seed_alignment]
        self.n = len(seed_alignment)
        self.m = len(seed_alignment[0])
        self.max_shift = max_shift
        self.gap_positions = [(i,j) for i in range(self.n) for j in range(self.m) if self.seed[i][j]=='-']
        self.dim = len(self.gap_positions)
        self.orig_chars = [[self.seed[i][j] for j in range(self.m)] for i in range(self.n)]

    def get_dimension(self): return self.dim

    def decode(self, vector):
        aln = [row[:] for row in self.seed]
        m = self.m
        rng = np.random.RandomState(SEED)   # deterministic random for insert positions
        for idx, (i, j) in enumerate(self.gap_positions):
            if idx >= len(vector): break
            v = vector[idx]
            if aln[i][j] != '-': continue
            if abs(v) < 0.2: continue
            if v >= 0.8:   # deletion
                orig_char = self.orig_chars[i][j]
                aln[i][j] = orig_char if orig_char != '-' else 'A'
                continue
            if v <= -0.8:  # insertion
                non_gap_cols = [k for k in range(m) if aln[i][k] != '-']
                if non_gap_cols:
                    pick_idx = int((abs(v) - 0.8) * 5 * len(non_gap_cols)) % len(non_gap_cols)
                    ins_col = non_gap_cols[pick_idx]
                    aln[i][ins_col] = '-'
                continue
            # shift
            shift = int(round(v * self.max_shift))
            if shift == 0: continue
            new_col = j + shift
            new_col = max(0, min(m-1, new_col))
            if new_col == j: continue
            if aln[i][new_col] != '-':
                aln[i][new_col], aln[i][j] = '-', aln[i][new_col]
        return [''.join(row) for row in aln]

# ========================= Enhanced EO (now with initial vector) =========================
class EnhancedEO:
    def __init__(self, dim, pop_size, bounds=(-1,1), initial_vector=None):
        self.dim=dim; self.pop_size=pop_size; self.L,self.U=bounds
        if initial_vector is not None:
            # use provided vector for all individuals (no randomness)
            self.pop = np.tile(initial_vector, (pop_size, 1))
        else:
            self.pop = np.random.uniform(self.L, self.U, (pop_size, dim))
        self.fit = np.zeros(pop_size)
        self.best = None
        self.best_fit = -np.inf
        self.stag = 0
        self.p_mut = 0.1; self.alpha_levy = 0.01; self.GP_max = 0.5
    def levy(self, size):
        v = np.random.randn(size)
        return 1 / (np.abs(v) ** (1/1.5) + 1e-10)
    def eq_pool(self):
        idx = np.argsort(self.fit)[::-1]
        top = self.pop[idx[:4]]
        return np.vstack([top, np.mean(top, axis=0)])
    def evolve(self, fitness_func, t, Tmax):
        T = t / Tmax
        for i in range(self.pop_size):
            self.fit[i] = fitness_func(self.pop[i])
        bi = np.argmax(self.fit)
        if self.fit[bi] > self.best_fit:
            self.best_fit = self.fit[bi]
            self.best = self.pop[bi].copy()
            self.stag = 0
        else:
            self.stag += 1
        eq = self.eq_pool()
        new = np.zeros_like(self.pop)
        for i in range(self.pop_size):
            xeq = eq[np.random.randint(len(eq))]
            lam = np.random.rand(self.dim)
            F = np.exp(-lam * T)
            r = np.random.rand(self.dim)
            G0 = (2 * np.random.rand() - 1) * (1 - 2 * r)
            R = 2 * np.random.rand(self.dim) - 1
            Xn = xeq + (self.pop[i] - xeq) * F + G0 * R * (1 - T) * (self.U - self.L)
            Xn += self.alpha_levy * self.levy(self.dim)
            Xn += np.random.randn(self.dim) * 0.5 * (1 - T)
            Xn = np.clip(Xn, self.L, self.U)
            GP = np.random.uniform(0, self.GP_max)
            mask = np.random.rand(self.dim) < GP
            Xn[mask] = np.random.uniform(self.L, self.U, np.sum(mask))
            new[i] = Xn
        self.pop = new
        if self.stag > 10:
            self.p_mut = min(0.5, 1.5 * self.p_mut)
            worst = np.argsort(self.fit)[:self.pop_size // 4]
            self.pop[worst] = np.random.uniform(self.L, self.U, (len(worst), self.dim))
        div = np.nan_to_num(np.std(self.pop, axis=0).mean(), nan=0.0)
        return {'best_fitness': self.best_fit, 'mean_fitness': np.mean(self.fit),
                'diversity': div, 'stagnation': self.stag}
    def update_params(self, d):
        self.p_mut = np.clip(self.p_mut + d.get('dp_mut', 0), 0.01, 0.5)
        self.alpha_levy = np.clip(self.alpha_levy + d.get('d_levy', 0), 0.001, 0.1)
        self.GP_max = np.clip(self.GP_max + d.get('d_GP', 0), 0.1, 0.8)

# ========================= SAC (unchanged) =========================
class Actor(nn.Module):
    def __init__(self,sdim,adim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(sdim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.mean=nn.Linear(256,adim); self.logstd=nn.Linear(256,adim)
    def forward(self,s):
        x=self.net(s); return self.mean(x), torch.clamp(self.logstd(x),-5,2)
    def sample(self,s):
        mean,logstd=self.forward(s)
        mean=torch.nan_to_num(mean,0.0); logstd=torch.nan_to_num(logstd,0.0)
        std=logstd.exp().clamp(1e-6)
        dist=torch.distributions.Normal(mean,std)
        x=dist.rsample(); action=torch.tanh(x)
        logp=dist.log_prob(x)-torch.log(1-action.pow(2)+1e-6)
        return action, logp.sum(1,keepdim=True)
class Critic(nn.Module):
    def __init__(self,sdim,adim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(sdim+adim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,s,a): return self.net(torch.cat([s,a],1))
class SAC:
    def __init__(self,sdim,adim):
        self.dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor=Actor(sdim,adim).to(self.dev); self.actor_opt=optim.Adam(self.actor.parameters(),3e-4)
        self.c1=Critic(sdim,adim).to(self.dev); self.c2=Critic(sdim,adim).to(self.dev)
        self.c1t=Critic(sdim,adim).to(self.dev); self.c2t=Critic(sdim,adim).to(self.dev)
        self.c1t.load_state_dict(self.c1.state_dict()); self.c2t.load_state_dict(self.c2.state_dict())
        self.c1_opt=optim.Adam(self.c1.parameters(),3e-4); self.c2_opt=optim.Adam(self.c2.parameters(),3e-4)
        self.alpha=0.2; self.gamma=0.99; self.tau=0.005; self.target_ent=-adim
        self.log_alpha=torch.zeros(1,requires_grad=True,device=self.dev); self.a_opt=optim.Adam([self.log_alpha],3e-4)
        self.buf=deque(maxlen=300000)
    def state(self, eo_stats, sp, cs, gr, it, maxit):
        s = np.array([
            eo_stats['diversity'],
            eo_stats['mean_fitness']/1000,
            eo_stats['best_fitness']/1000,
            eo_stats['stagnation']/10,
            sp/100,
            cs,
            gr/100,
            it/maxit
        ], dtype=np.float32)
        return np.nan_to_num(s,0.0)
    def act(self,s,eval=False):
        s=torch.FloatTensor(s).unsqueeze(0).to(self.dev)
        if eval:
            with torch.no_grad(): m,_=self.actor(s); return torch.tanh(m).cpu().numpy()[0]
        a,_=self.actor.sample(s); return a.detach().cpu().numpy()[0]
    def store(self,*t): self.buf.append(t)
    def update(self, bs=128):
        if len(self.buf)<bs: return {}
        batch=random.sample(self.buf,bs); s,a,r,ns,d=zip(*batch)
        s=torch.FloatTensor(np.array(s)).to(self.dev); a=torch.FloatTensor(np.array(a)).to(self.dev)
        r=torch.FloatTensor(r).unsqueeze(1).to(self.dev); ns=torch.FloatTensor(np.array(ns)).to(self.dev)
        d=torch.FloatTensor(d).unsqueeze(1).to(self.dev)
        with torch.no_grad():
            na,nlp=self.actor.sample(ns); q1n=self.c1t(ns,na); q2n=self.c2t(ns,na)
            qn=torch.min(q1n,q2n); vn=qn-self.alpha*nlp; qt=r+(1-d)*self.gamma*vn
        q1=self.c1(s,a); q2=self.c2(s,a)
        l1=nn.MSELoss()(q1,qt); l2=nn.MSELoss()(q2,qt)
        self.c1_opt.zero_grad(); l1.backward(); torch.nn.utils.clip_grad_norm_(self.c1.parameters(),1); self.c1_opt.step()
        self.c2_opt.zero_grad(); l2.backward(); torch.nn.utils.clip_grad_norm_(self.c2.parameters(),1); self.c2_opt.step()
        na,lp=self.actor.sample(s); q1n=self.c1(s,na); q2n=self.c2(s,na); qn=torch.min(q1n,q2n)
        aloss=(self.alpha*lp-qn).mean()
        self.actor_opt.zero_grad(); aloss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1); self.actor_opt.step()
        alphaloss=-(self.log_alpha*(lp+self.target_ent).detach()).mean()
        self.a_opt.zero_grad(); alphaloss.backward(); torch.nn.utils.clip_grad_norm_([self.log_alpha],1); self.a_opt.step()
        self.alpha=self.log_alpha.exp().item()
        for p,tp in zip(self.c1.parameters(),self.c1t.parameters()): tp.data.copy_(self.tau*p+(1-self.tau)*tp)
        for p,tp in zip(self.c2.parameters(),self.c2t.parameters()): tp.data.copy_(self.tau*p+(1-self.tau)*tp)
        return {}

# ========================= MOFSACEO‑MSA (uses zero vector as initial population) =========================
class MOFSACEO_MSA:
    def __init__(self, seqs, seed_alignment, pop_size=200, max_iter=3000):
        self.seqs=seqs; self.seed=seed_alignment
        self.rep = FlexibleGapRepresentation(seed_alignment, max_shift=15)
        self.dim=self.rep.get_dimension()
        print(f"Representation dimension: {self.dim}")
        # initial vector = zero => no gap shifts, exact MAFFT alignment
        init_vec = np.zeros(self.dim)
        self.eo=EnhancedEO(self.dim, pop_size, bounds=(-1,1), initial_vector=init_vec)
        self.sac=SAC(8,4)
        self.max_iter=max_iter
        self.hist={'sp':[],'diversity':[],'levy':[],'pmut':[],'gap_ratio':[],'cs':[]}
    def decode(self,x): return self.rep.decode(x)
    def fitness(self,x):
        aln=self.decode(x); return sp_score(aln)
    def run(self):
        prev_sp=-np.inf
        for t in range(1,self.max_iter+1):
            st=self.eo.evolve(self.fitness, t, self.max_iter)
            best_aln=self.decode(self.eo.best)
            sp=sp_score(best_aln); cs=cs_score(best_aln); gr=gap_ratio(best_aln)
            state=self.sac.state(st, sp, cs, gr, t, self.max_iter)
            action=self.sac.act(state)
            self.eo.update_params({'dp_mut':action[0]*0.01,'d_levy':action[2]*0.001,'d_GP':action[3]*0.01})
            reward=(sp-prev_sp)/100.0
            prev_sp=sp
            self.sac.store(state, action, reward, state, False)
            if t%2==0: self.sac.update(bs=128)
            self.hist['sp'].append(sp); self.hist['diversity'].append(st['diversity'])
            self.hist['levy'].append(self.eo.alpha_levy); self.hist['pmut'].append(self.eo.p_mut)
            self.hist['gap_ratio'].append(gr); self.hist['cs'].append(cs)
            if t%500==0 or t==1:
                print(f"Iter {t:5d}  SP={sp:6.1f}  diversity={st['diversity']:.3f}  α_levy={self.eo.alpha_levy:.4f}  p_mut={self.eo.p_mut:.3f}")
        return self.eo.best, self.hist['sp'][-1], self.hist

# ========================= Plotting functions (same) =========================
def plot_convergence(hist_sac, hist_fix, initial_sp, filename='Figure_convergence.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(hist_sac['sp'], label='MOFSACEO-MSA (SAC)', linewidth=2)
    plt.plot(hist_fix['sp'], label='EO (Fixed parameters)', linestyle='--', linewidth=2)
    plt.axhline(initial_sp, color='gray', linestyle=':', label='MAFFT seed')
    plt.xlabel('Iteration'); plt.ylabel('SP Score')
    plt.title('Convergence behavior on RF00014')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.show()

def plot_parameter_evolution(hist_sac, hist_fix, filename='Figure_parameters.png'):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
    ax1.plot(hist_sac['levy'], label='α Lévy (SAC)', color='green')
    ax1.axhline(hist_fix['levy'][0], color='red', linestyle='--', label='Fixed α Lévy')
    ax1.set_ylabel('Lévy Scale'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(hist_sac['pmut'], label='p_mut (SAC)', color='purple')
    ax2.axhline(hist_fix['pmut'][0], color='red', linestyle='--', label='Fixed p_mut')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Mutation Probability')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle('Evolution of SAC-controlled EO parameters')
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.show()

def plot_correlation_heatmap(hist_sac, filename='Figure_heatmap.png'):
    import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    data = {
        'Iteration': np.arange(1, len(hist_sac['sp'])+1),
        'SP Score': hist_sac['sp'],
        'Column Score': hist_sac['cs'],
        'Gap Ratio': hist_sac['gap_ratio'],
        'Diversity': hist_sac['diversity'],
        'Lévy Scale': hist_sac['levy'],
        'Mutation Prob': hist_sac['pmut'],
        'SP Improvement': np.diff(hist_sac['sp'], prepend=hist_sac['sp'][0])
    }
    df = pd.DataFrame(data)
    constant_cols = [c for c in df.columns if df[c].nunique()<=1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Correlation between SAC-adjusted parameters and fitness improvement')
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.show()

# ========================= Main Experiment (both start from zero vector) =========================
def run_experiment():
    fasta_file = "input.fasta"
    if not os.path.exists(fasta_file): print("input.fasta missing"); return
    seqs = read_fasta(fasta_file)
    print(f"Loaded {len(seqs)} sequences, max len {max(len(s) for s in seqs)}")
    seed = run_mafft_seed(seqs)
    if not seed: seed = [s.ljust(max(len(s) for s in seqs), '-') for s in seqs]
    print(f"Seed length={len(seed[0])}, gaps={sum(s.count('-') for s in seed)}")
    initial_sp = sp_score(seed)
    print(f"Initial SP={initial_sp}")

    # SAC run (starts from zero vector)
    print("\n=== MOFSACEO-MSA with SAC (pop=200, iter=3000) ===")
    opt_sac = MOFSACEO_MSA(seqs, seed, pop_size=200, max_iter=3000)
    _, sp_sac, hist_sac = opt_sac.run()

    # Fixed parameter run (starts from zero vector)
    print("\n=== EO with Fixed Parameters (pop=200, iter=3000) ===")
    opt_fix = MOFSACEO_MSA(seqs, seed, pop_size=200, max_iter=3000)
    hist_fix = {'sp':[],'diversity':[],'levy':[],'pmut':[],'gap_ratio':[],'cs':[]}
    for t in range(1,3001):
        st = opt_fix.eo.evolve(opt_fix.fitness, t, 3000)
        aln = opt_fix.decode(opt_fix.eo.best)
        sp = sp_score(aln); cs = cs_score(aln); gr = gap_ratio(aln)
        hist_fix['sp'].append(sp); hist_fix['diversity'].append(st['diversity'])
        hist_fix['levy'].append(opt_fix.eo.alpha_levy); hist_fix['pmut'].append(opt_fix.eo.p_mut)
        hist_fix['gap_ratio'].append(gr); hist_fix['cs'].append(cs)
        if t%500==0: print(f"Iter {t:5d}  SP={sp:6.1f}  diversity={st['diversity']:.3f}")
    sp_fix = hist_fix['sp'][-1]

    print(f"\nFinal SP: SAC={sp_sac:.1f}, Fixed={sp_fix:.1f} (Initial={initial_sp})")
    print(f"Improvement: SAC +{sp_sac-initial_sp:.1f}, Fixed +{sp_fix-initial_sp:.1f}")

    # Generate required figures
    plot_convergence(hist_sac, hist_fix, initial_sp, 'Figure_convergence.png')
    plot_parameter_evolution(hist_sac, hist_fix, 'Figure_parameters.png')
    plot_correlation_heatmap(hist_sac, 'Figure_heatmap.png')

    # Table
    import pandas as pd
    df = pd.DataFrame({
        'Method': ['MAFFT', 'MOFSACEO-MSA (SAC)', 'EO (Fixed)'],
        'SP': [initial_sp, sp_sac, sp_fix],
        'CS': [cs_score(seed), hist_sac['cs'][-1], hist_fix['cs'][-1]],
        'Gap%': [gap_ratio(seed), hist_sac['gap_ratio'][-1], hist_fix['gap_ratio'][-1]]
    })
    print("\nQuantitative Comparison:")
    print(df.to_string(index=False))
    df.to_csv('final_comparison.csv', index=False)

if __name__ == '__main__':
    run_experiment()
