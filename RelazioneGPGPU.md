# Relazione GPGPU — Implementazione di `findall` su GPU

> **Target device:** AMD Radeon RX 5700 XT (RDNA 1, `gfx1010:xnack-`)
> **Data:** 2026-03-26
> **File sorgente:** `findall_final.{c,ocl}` · `findall_lmem.{c,ocl}` · `findall_lmem_v2.{c,ocl}`
> **Backend Julia:** `KernelAbstractions.jl` + `AMDGPU.jl` (ROCm)

---

## Indice

1. [Contesto e obiettivi](#1-contesto-e-obiettivi)
2. [Approccio Julia — KernelAbstractions.jl](#2-approccio-julia--kernelabstractionsjl)
   - [Problematiche e limitazioni riscontrate](#21-problematiche-e-limitazioni-riscontrate)
   - [Implementazione in Julia](#22-implementazione-in-julia)
3. [Approccio OpenCL — Algoritmo e architettura](#3-approccio-opencl--algoritmo-e-architettura)
   - [`findall_final` — Kogge-Stone Global Ping-Pong](#31-findall_final--kogge-stone-global-ping-pong)
   - [`findall_lmem` — 2-Level Local Scan](#32-findall_lmem--2-level-local-scan)
   - [`findall_lmem_v2` — 3-Level Recursive Local Scan](#33-findall_lmem_v2--3-level-recursive-local-scan)
   - [Confronto riepilogativo](#34-confronto-riepilogativo)
4. [Bug identificati e corretti](#4-bug-identificati-e-corretti)
5. [Test e risultati](#5-test-e-risultati)
6. [Limiti hardware — AMD RX 5700 XT](#6-limiti-hardware--amd-rx-5700-xt)
7. [Conclusioni](#7-conclusioni)
8. [Compilazione e benchmark](#8-compilazione-e-benchmark)

---

## 1. Contesto e obiettivi

Il progetto si inserisce nel contesto di `firesim.jl`, una simulazione di incendi che richiede, ad
ogni iterazione, l'individuazione di tutti gli elementi positivi della matrice `has_burning_neibs`
(una matrice di `Int8` che codifica i possibili stati da −2 a +2). L'operazione fondamentale è
equivalente a `findall(x -> x > 0, has_burning_neibs)` di Julia, da accelerare su GPU AMD.

L'obiettivo è duplice:

- **Esplorazione Julia:** su consiglio del Prof. Bilotta, valutare `KernelAbstractions.jl` +
  `AMDGPU.jl` per kernel portabili tra backend diversi, mantenendo un alto livello di astrazione
  pur rispettando le esigenze prestazionali di un'operazione di scan.
- **Implementazione OpenCL:** sviluppare, in C + OpenCL, tre versioni progressive dell'algoritmo
  prefix-sum + scatter, caratterizzate da complessità di memoria e limiti di dimensione differenti,
  fino alla versione `findall_lmem_v2`.

---

## 2. Approccio Julia — KernelAbstractions.jl

### 2.1 Problematiche e limitazioni riscontrate

L'uso combinato di `KernelAbstractions.jl` e `AMDGPU.jl` ha evidenziato una serie di limitazioni
pratiche e tecniche ancora aperte al momento dello sviluppo:

| # | Problema | Dettaglio |
|---|---|---|
| 1 | **Assenza di `@shared`** | La macro per la memoria locale condivisa tra work-item non è disponibile in `KernelAbstractions`. |
| 2 | **No strutture dinamiche** | Operazioni come `setindex!` non sono supportate: `unsupported dynamic function invocation`. |
| 3 | **Debugging complesso** | `Cthulhu.jl` è disponibile, ma il debugging su CPU è ostacolato dai bug [#544](https://github.com/JuliaGPU/KernelAbstractions.jl/issues/544) e [#262](https://github.com/JuliaGPU/KernelAbstractions.jl/issues/262). A volte inizializzare la variabile nel `for` risolve il problema. Le funzioni come `@print()` sono disponibili **solo** sul backend CPU. |
| 4 | **`@groupsize()` e `@ndrange()`** | Effettuano una query sul backend: `@groupsize()` restituisce una `Tuple` che causa errori di tipo a compile-time (probabilmente correlato al bug #544). **Soluzione adottata:** passare `groupsize` e `ndrange` come parametri espliciti. |
| 5 | **Broadcasting su ROCm** | Non tutte le operazioni di broadcasting (`.+=`, `.*`) sono supportate nativamente su GPU ROCm; in alcuni casi non viene effettuato parallelismo automatico, rendendo necessario l'uso esplicito di kernel. |

### 2.2 Implementazione in Julia

Il problema `findall` viene scomposto in cinque step, segmentati su kernel separati per facilitare
il debugging:

```
Input: has_burning_neibs (Int8[M, N])
   │
   ▼  Step 1 — Flagging
       Ogni work-item calcola: flags[i] = (input[i] > 0) ? 1 : 0
   │
   ▼  Step 2 — Scan locale (per blocco)
       Scan esclusivo sui propri elementi usando memoria locale.
       Produce scan_out parziale + block_sums[].
   │
   ▼  Step 3 — Scan globale dei block_sums
       Scan per ottenere gli offset globali di ciascun blocco.
   │
   ▼  Step 4 — Fix-up
       d_scan_global[i] = scan_out[i] + block_offset[group]
   │
   ▼  Step 5 — Scrittura finale
       output[d_scan_global[i]] = linear_index(i)  (se flags[i] = 1)
   │
   ▼  Conversione lineare → cartesiana
       (row, col) = divrem(idx - 1, nrows) .+ (1, 1)
```

> **Nota:** la matrice `has_burning_neibs` viene convertita in un array monodimensionale prima
> dell'elaborazione, per semplificare la gestione degli indici. Gli indici lineari vengono poi
> riconvertiti in coordinate cartesiane tramite:
> ```julia
> function linear_to_cartesian(linear_indices, nrows)
>     return [(divrem(idx - 1, nrows) .+ (1, 1))[end:-1:1] for idx in linear_indices]
> end
> ```

#### Seconda implementazione (`findallv2.jl`)

Una seconda versione usa un approccio **ibrido**: con `lws` piccoli viene usata la versione seriale,
altrimenti una versione *unrolled* (ciclo espanso manualmente). Questo approccio non è scalabile,
ma evita i problemi di compilazione e di tipo su GPU ROCm (su CPU causa `LoadError: UndefVarError: i
not defined in Main`, bug #544).

---

## 3. Approccio OpenCL — Algoritmo e architettura

Tutte e tre le implementazioni risolvono lo stesso problema: data un array `char[N]`, raccogliere
gli indici di ogni elemento **positivo** tramite una pipeline GPU prefix-sum (scan) + scatter.

### 3.1 `findall_final` — Kogge-Stone Global Ping-Pong

```
Input (char[N])
   │
   ▼  mark_positive_kernel          scrive 0/1 flag → d_scan[0]
      ┌─ clEnqueueCopyBuffer ──────► d_flags   (copia permanente per la scatter)
   │
   ▼  scan_stride_kernel × log₂(N)  passo Kogge-Stone con stride 2^s
      ping-pong tra d_scan[0] ↔ d_scan[1]
   │
   ▼  legge count = d_scan_result[N-1]
   │
   ▼  findall_gpu                   scatter: output[scan[i]-1] = i  (se flag[i]=1)
```

**Approccio:** Scan Kogge-Stone puramente in memoria globale. Ogni stride è un kernel separato
(= barriera globale implicita). Nessuna memoria locale.

| Proprietà | Valore |
|---|---|
| Traffico memoria globale | **O(N log N)** — ogni stride legge/scrive tutti gli N elementi |
| Kernel lanciati | 2 + log₂(N) + 1 |
| Funziona su PoCL/CPU? | ✅ Sì — nessuna barriera locale richiesta |
| Limite di dimensione | Nessuno |
| Uso consigliato | Test di correttezza; portabilità massima |

**Pro:** codice semplicissimo; sempre corretto; nessun vincolo su block-size.
**Contro:** spreca banda a N grande per il traffico O(N log N).

---

### 3.2 `findall_lmem` — 2-Level Local Scan

```
Input (char[N])
   │
   ▼  mark_and_scan_local   (1 kernel, N work-item)
      ├─ carica flag in lmem[]
      ├─ Hillis-Steele inclusive scan dentro il work-group (local memory)
      ├─ scrive d_flags[], d_scan_local[]
      └─ scrive d_block_sums[group]  ← somma di ogni blocco
   │
   ▼  scan_block_sums       (1 kernel, num_groups work-item, singolo WG)
      ├─ inclusive scan di d_block_sums[] → offset esclusivi
      └─ scrive d_block_offsets[]
   │
   ▼  apply_offsets         (1 kernel, N work-item)
      └─ d_scan_global[i] = d_scan_local[i] + d_block_offsets[group]
   │
   ▼  legge count = d_scan_global[N-1]
   │
   ▼  findall_gpu           scatter
```

**Approccio:** Reduce-poi-scan a due livelli. Ogni work-group scansiona il proprio tile in
memoria locale (32 KB/CU su RX 5700 XT). Un secondo kernel single-WG scansiona le somme per
blocco. Un terzo kernel propaga gli offset.

| Proprietà | Valore |
|---|---|
| Traffico memoria globale | **O(N)** — fattore costante ~4–5 passate |
| Kernel lanciati | 4 |
| Memoria locale per WG | `lws × sizeof(int)` |
| **Limite di dimensione** | **N ≤ lws²** (num_groups deve stare in un solo WG al passo 2) |
| Max N su RX 5700 XT | ~65 K (lws=256), ~1 M (lws=1024) |

**Pro:** traffico O(N); buona occupancy GPU.
**Contro:** limite di scalabilità (N ≤ lws²).

---

### 3.3 `findall_lmem_v2` — 3-Level Recursive Local Scan ⭐ Migliore

```
Input (char[N])                   ng0 = ⌈N/lws⌉
   │
   ▼  scan_local_k0    (N elementi)
      ├─ d_flags[],  d_scan0[]    ← scan inclusivo per blocco
      └─ d_sums0[]                ← ng0 totali per blocco
   │
   ▼  scan_local_k1    (ng0 elementi)   ng1 = ⌈ng0/lws⌉
      ├─ d_scan1[]                ← scan inclusivo per blocco di sums0
      └─ d_sums1[]                ← ng1 totali per blocco
   │
   ▼  scan_local_k2    (ng1 elementi, singolo WG)
      └─ d_offsets1[]             ← scan esclusivo di sums1
   │
   ▼  apply_k1         (ng0 elementi)
      └─ d_scan1g[i] = d_scan1[i] + d_offsets1[group]
   │
   ▼  apply_k0         (N elementi)
      └─ d_scan_global[i] = d_scan0[i] + d_scan1g[group]
   │
   ▼  legge count = d_scan_global[N-1]
   │
   ▼  findall_gpu                 scatter
```

**Approccio:** Scan ricorsivo a tre livelli (analogo a CUB/Thrust `DeviceScan`). Rispecchia
l'approccio a 2 livelli su due strati aggiuntivi, eliminando il vincolo di dimensione.

| Proprietà | Valore |
|---|---|
| Traffico memoria globale | **O(N)** — ~6–7 passate, fattore costante |
| Kernel lanciati | 6 |
| Memoria locale per WG | `lws × sizeof(int)` |
| **Limite di dimensione** | **N ≤ lws³** |
| Max N su RX 5700 XT | ~16 M (lws=256), ~**1 miliardo** (lws=1024) |

**Pro:** traffico O(N); nessun limite pratico di dimensione; massimo parallelismo.
**Contro:** codice più complesso; 2 kernel launch extra rispetto alla v1.

---

### 3.4 Confronto riepilogativo

| | `findall_final` | `findall_lmem` | `findall_lmem_v2` |
|---|:---:|:---:|:---:|
| Traffico memoria | O(N log N) | O(N) | O(N) |
| Kernel lanciati | 2+log₂N+1 | 4 | 6 |
| Memoria locale | ❌ | ✅ | ✅ |
| Max N (lws=256) | illimitato | ~65 K | ~16 M |
| Max N (lws=1024) | illimitato | ~1 M | ~1 G |
| **Uso consigliato** | test/portabilità | array medi | **produzione** ✅ |

---

## 3. Bug identificati e corretti

Durante lo sviluppo per portare il codice da Julia a OpenCL, ho effettuato diverse iterazioni per capire quale fosse il problema dei kernel, da qui le 3 diverse implementazioni, in totale sono stati identificati e corretti tre bug.

---

### Bug 1 & 2 — `findall_lmem_v2.ocl`: Indice errato nella somma dell'ultimo blocco parziale

**Kernel interessati:** `scan_local_k0` (Bug 1) e `scan_local_k1` (Bug 2)

**Causa:** Quando `N` non è multiplo di `lws`, l'ultimo work-group è *parziale*: alcuni
work-item hanno `gi >= N` e non eseguono lavoro reale. Il codice originale cercava l'ultimo indice
locale **valido** tramite `min(lws-1, N-1-grp*lws)`, per evitare di leggere uno slot riempito
di zeri. Tuttavia:

- L'espressione `N - 1 - grp * lws` può diventare **negativa** per gruppi oltre il range dei
  dati, producendo un indice unsigned non definito (con wrap).
- Anche se non negativo, la logica è inutile: i work-item fuori range caricano già `lmem[li] = 0`
  (nessun contributo), quindi dopo lo scan inclusivo `lmem[lws-1]` è **sempre** uguale al totale
  del blocco, indipendentemente dal numero di thread attivi.

**Sintomi:** conteggio totale errato e indici scatter sbagliati per qualsiasi `N` non multiplo di `lws`.

**Fix applicato a `scan_local_k0`:**
```diff
-    if (li == lws - 1) {
-        int last_li = min(lws - 1, nels - 1 - grp * lws);
-        d_sums0[grp] = lmem[last_li];          // <- indice sbagliato, possibile underflow
-    }
+    /* lmem[lws-1] è sempre il totale del blocco: le lane fuori range hanno caricato 0,
+     * quindi lo scan inclusivo all'ultima lane è pari alla somma dei flag validi. */
+    if (li == lws - 1)
+        d_sums0[grp] = lmem[li];               // <- sempre corretto
```

Fix identico applicato a `scan_local_k1` per `d_sums1[grp]`.

---

### Bug 3 — `findall_lmem.ocl`: Race condition nella lettura della somma di blocco dalla memoria globale

**Kernel interessato:** `mark_and_scan_local`

**Causa:** Dopo aver scritto `d_scan_local[gi]` in memoria globale, l'ultimo work-item
del blocco tentava di rileggere la somma di blocco da `d_scan_local[last]` (globale):

```c
// BUG:
if (li == lws - 1) {
    int last = min((grp + 1) * lws - 1, nels - 1);
    d_block_sums[grp] = d_scan_local[last];   // legge dalla globale — non ancora liberata!
}
```

Questa è una **race condition**: la scrittura su `d_scan_local[last]` avviene nello stesso
kernel, e OpenCL non garantisce la visibilità dei write globali agli altri work-item nello stesso
kernel senza una barriera globale (impossibile in un singolo kernel su GPU). Il valore è già
disponibile in `lmem[lws-1]`.

**Fix:**
```diff
-    if (li == lws - 1) {
-        int last = min((grp + 1) * lws - 1, nels - 1);
-        d_block_sums[grp] = d_scan_local[last];
-    }
+    /* lmem[lws-1] è il totale del blocco: le lane fuori range hanno caricato 0 */
+    if (li == lws - 1)
+        d_block_sums[grp] = lmem[li];
```

---


## 5. Test e risultati

Test eseguiti su: AMD Radeon RX 5700 XT (`gfx1010:xnack-`) via AMD OpenCL.

### Test funzionali

| Test | N | lws | ng0 | Scopo | Risultato |
|---|---|---|---|---|---|
| Blocco parziale piccolo | 10 | 4 | 3 | ultimo blocco parziale, N dispari | ✅ **PASSED** |
| Potenza di due | 64 | 8 | 8 | allineato, 2 livelli | ✅ **PASSED** |
| Confine cross-WG | 20 | 16 | 2 | N non multiplo di lws | ✅ **PASSED** |
| Grande | 1 000 000 | 256 | 3907 | scala reale | ✅ **PASSED** |

Dettaglio Test 1 (N=10, lws=4):
```
Input:  -1 -2 -1 -1 0 1 -1 -2 -2 1
Trovati 2 elementi positivi agli indici: 5 9
Verifica: PASSED
```

### Performance (N = 1 000 000, lws = 256, RX 5700 XT)

| Kernel | Tempo | Banda |
|---|---|---|
| `scan_k0` | 0.052 ms | 174 GB/s |
| `scan_k1` | 0.003 ms | 15 GB/s |
| `scan_k2` | 0.003 ms | 0.04 GB/s |
| `apply_k1` | 0.002 ms | 17 GB/s |
| `apply_k0` | 0.036 ms | 223 GB/s |
| `findall_gpu` | 0.251 ms | 38 GB/s |
| **TOTALE** | **3.926 ms** | **0.25 GE/s** |

> `scan_k0` e `apply_k0` saturano ~50% della banda di picco (448 GB/s), limitati dall'overhead
> di kernel launch a questa dimensione del problema. A N=16M si prevede che l'utilizzo di banda si
> avvicini ai valori di picco.

---

## 6. Limiti hardware — AMD RX 5700 XT

### Specifiche dispositivo (OpenCL)

| Proprietà | Valore |
|---|---|
| Architettura | RDNA 1 (`gfx1010`) |
| Compute Units | 40 CU |
| Stream processor | 2560 (64 per CU) |
| Dimensione wavefront | **64 thread** |
| Max work-group size | **256** (default via `clinfo`; fino a 1024 con attributi) |
| Memoria locale per WG | **32 KiB** (32 768 byte) |
| Max memoria locale (int) | 8 192 valori `int` per work-group |
| Memoria globale | 8 GB GDDR6 |
| Banda di picco globale | **448 GB/s** |
| Max constant buffer | 64 KB |
| Max allocazione `__global` | ~2 GB |

### Limiti di `lws` e `N` per algoritmo

Il vincolo critico è: `lws × sizeof(int) ≤ 32 768` byte, quindi `lws ≤ 8192`.
In pratica, `lws` deve essere multiplo della wavefront size (64), quindi i valori sensati
sono **64, 128, 256, 512, 1024**.

#### `findall_lmem` (2 livelli)

| lws | Max N = lws² |
|---|---|
| 64 | 4 096 |
| 128 | 16 384 |
| 256 | **65 536** |
| 512 | 262 144 |
| 1024 | **1 048 576** (~1 M) |

#### `findall_lmem_v2` (3 livelli) ⭐

| lws | Max N = lws³ |
|---|---|
| 64 | 262 144 (~256 K) |
| 128 | 2 097 152 (~2 M) |
| 256 | **16 777 216 (~16 M)** |
| 512 | 134 217 728 (~128 M) |
| 1024 | **1 073 741 824 (~1 G)** |

> **`lws` raccomandato per RX 5700 XT:** `256`.
> - `lws=256`: si adatta a 4 wavefront per CU, 32 KB locale esattamente sufficiente.
> - `lws=512`: 8 wavefront, ancora dentro la memoria locale, occupancy migliore a N grande.
> - Evitare `lws=1024` salvo necessità per array enormi — meno WG attivi per CU riduce il
>   latency hiding.

### Stima di occupancy (lws = 256)

- Memoria locale per WG: `256 × 4 = 1 024 byte`
- Max WG per CU limitati dalla memoria locale: `32 768 / 1 024 = 32` WG/CU
- Max WG per CU limitati dai wavefront: `256 / 64 = 4` wavefront/WG → tipicamente
  10 WG attivi/CU → **non** memory-bound a lws=256
- Con 40 CU: fino a **400 work-group concorrenti** per N grande

### Raccomandazioni pratiche

| Caso d'uso | Scelta migliore |
|---|---|
| N ≤ 1M, semplicità | `findall_lmem` con lws=1024 |
| N ≤ 16M, uso produzione | `findall_lmem_v2` con lws=256 |
| N > 16M | `findall_lmem_v2` con lws=512 o 1024 |
| Qualsiasi N, portabilità richiesta | `findall_final` (Kogge-Stone) |

---

## 7. Conclusioni

L'uso combinato di `KernelAbstractions.jl` e `AMDGPU.jl` ha dimostrato la **fattibilità teorica**
di operazioni parallele GPU complesse come `findall` in Julia, ma ha evidenziato una serie di
limitazioni pratiche ancora aperte: assenza di `@shared`, scarso supporto al debugging su GPU,
problemi di tipo con `@groupsize()`, e broadcasting non uniforme su ROCm. Per superare questi
ostacoli, sarebbe auspicabile una migliore integrazione con ROCm, l'estensione delle macro di
sincronizzazione e memoria locale, e strumenti di debug più avanzati.

Dal lato OpenCL, le tre implementazioni mostrano un chiaro progresso:

- `findall_final` è l'implementazione di riferimento per correttezza e portabilità, ma non scalabile.
- `findall_lmem` introduce la memoria locale e abbatte il traffico a O(N), ma con un soffitto N ≤ lws².
- `findall_lmem_v2` rimuove il limite portandolo a N ≤ lws³ (~1 miliardo su questo hardware),
  risultando la scelta raccomandata per la produzione.

In prospettiva, l'ambiente Julia su GPU AMD mostra grande potenziale, ma richiede ancora
maturazione per scenari produttivi complessi. L'implementazione OpenCL rimane la scelta più
robusta e controllabile per questo tipo di workload.

---

## 8. Compilazione e benchmark

### Compilare i binari

Un `Makefile` è fornito nella directory del progetto per compilare tutte e tre le implementazioni in un unico comando.

```bash
# Compila tutti e tre i binari (findall_final, findall_lmem, findall_lmem_v2)
make

# Compila solo una versione specifica
make findall_lmem_v2

# Rimuove i binari compilati
make clean
```

Il `Makefile` utilizza `gcc -O2 -Wall` e linka `-lOpenCL`. Non sono necessarie dipendenze aggiuntive oltre ai driver OpenCL installati.

### Eseguire i benchmark

Il file `benchmark.fish` esegue una serie di test di scalabilità su CPU e GPU, variando N e `lws`. Richiede la shell **fish**.

```bash
# Benchmark con la versione di default (findall_lmem_v2)
./benchmark.fish

# Benchmark su una versione specifica
./benchmark.fish findall_lmem
./benchmark.fish findall_final
```

Lo script esegue tre test:

| Test | Piattaforma | N variabile | lws fisso |
|---|---|---|---|
| Scalabilità N (CPU) | `OCL_PLATFORM=1` | 2^10 … 2^24 | 256 |
| Scalabilità N (GPU) | `OCL_PLATFORM=0` | 2^10 … 2^24 | 256 |
| GPU vs CPU (lws variabile) | entrambe | 2^24 | 256, 512, 1024 |

> **Nota:** se il binario scelto non supporta un certo N (ad esempio `findall_lmem` oltre N > lws²),
> lo script stampa `(N too large for this binary)` e continua senza interrompersi.

### Selezionare la piattaforma OpenCL

La variabile d'ambiente `OCL_PLATFORM` seleziona il dispositivo:

| Valore | Dispositivo |
|---|---|
| `OCL_PLATFORM=0` | AMD Radeon RX 5700 XT (GPU) |
| `OCL_PLATFORM=1` | CPU (PoCL) |

Esempio di esecuzione manuale:

```bash
# GPU, N=10M, lws=256
OCL_PLATFORM=0 ./findall_lmem_v2 256 10000000

# CPU, N=1M, lws=256
OCL_PLATFORM=1 ./findall_lmem_v2 256 1000000
```
