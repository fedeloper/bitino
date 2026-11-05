/*
 * QEMU TCG plugin: executed guest instructions per instruction address (profiling aid for tools/insn_count.py).
 *
 *   qemu-aarch64 -cpu max -plugin libbf_pccount.so,out=pcs.tsv  ./bitino_bench ...
 *
 * Writes "<hex vaddr>\t<count>" lines at exit (one per executed instruction address; a static non-PIE binary's
 * vaddrs are its ELF addresses, so llvm-addr2line / llvm-objdump map them to source lines). Same counting rule as
 * bf_symcount.c (an instruction counts when its translation block executes). Not cycles.
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

typedef struct {
    uint64_t count;
    uint32_t n;
    uint64_t vaddr[];
} Tb;

static GPtrArray *tbs;
static GMutex lock;
static char *out_path;

static void tb_exec(unsigned int vcpu_index, void *udata)
{
    __atomic_fetch_add(&((Tb *)udata)->count, 1, __ATOMIC_RELAXED);
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    Tb *t = g_malloc0(sizeof(Tb) + n * sizeof(uint64_t));
    t->n = n;
    for (size_t i = 0; i < n; i++) {
        t->vaddr[i] = qemu_plugin_insn_vaddr(qemu_plugin_tb_get_insn(tb, i));
    }
    g_mutex_lock(&lock);
    g_ptr_array_add(tbs, t);
    g_mutex_unlock(&lock);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, t);
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    GHashTable *pcs = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    for (guint i = 0; i < tbs->len; i++) {
        Tb *t = g_ptr_array_index(tbs, i);
        if (!t->count) {
            continue;
        }
        for (uint32_t j = 0; j < t->n; j++) {
            uint64_t *c = g_hash_table_lookup(pcs, &t->vaddr[j]);
            if (!c) {
                uint64_t *k = g_new(uint64_t, 1);
                *k = t->vaddr[j];
                c = g_new0(uint64_t, 1);
                g_hash_table_insert(pcs, k, c);
            }
            *c += t->count;
        }
    }
    FILE *f = out_path ? fopen(out_path, "w") : stderr;
    if (!f) {
        f = stderr;
    }
    GHashTableIter it;
    gpointer k, v;
    g_hash_table_iter_init(&it, pcs);
    while (g_hash_table_iter_next(&it, &k, &v)) {
        fprintf(f, "%" PRIx64 "\t%" PRIu64 "\n", *(uint64_t *)k, *(uint64_t *)v);
    }
    if (f != stderr) {
        fclose(f);
    }
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                                           int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "out=")) {
            out_path = g_strdup(argv[i] + 4);
        } else {
            fprintf(stderr, "bf_pccount: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    tbs = g_ptr_array_new();
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
