/*
 * QEMU TCG plugin: executed guest instructions per ELF symbol.
 *
 *   qemu-aarch64 -cpu max -plugin libbf_symcount.so,out=counts.tsv  ./bitino_bench ...
 *
 * Writes "<count>\t<mangled symbol>" lines (descending) at exit, plus a "#total" line.
 * Instructions outside any sized symbol are reported as "?". Counts are exact retired guest
 * instruction counts (a translation block is counted when it executes), deterministic for a
 * single-threaded run, and independent of host load. They are NOT cycles: QEMU has no timing
 * model. See docs/benchmarking.md ("Instruction counts under emulation").
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

typedef struct {
    const char *name;
    uint64_t count;
} Sym;

typedef struct {
    Sym *sym;
    uint32_t n;
} Group;

typedef struct {
    uint32_t n_groups;
    Group g[];
} TbInfo;

static GHashTable *syms;  /* name -> Sym* */
static GMutex lock;
static char *out_path;

static Sym *get_sym(const char *name)
{
    if (!name) {
        name = "?";
    }
    Sym *s = g_hash_table_lookup(syms, name);
    if (!s) {
        s = g_new0(Sym, 1);
        s->name = g_strdup(name);
        g_hash_table_insert(syms, (gpointer)s->name, s);
    }
    return s;
}

static void tb_exec(unsigned int vcpu_index, void *udata)
{
    TbInfo *t = udata;
    for (uint32_t i = 0; i < t->n_groups; i++) {
        __atomic_fetch_add(&t->g[i].sym->count, t->g[i].n, __ATOMIC_RELAXED);
    }
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);
    TbInfo *t = g_malloc0(sizeof(TbInfo) + n * sizeof(Group));
    g_mutex_lock(&lock);
    for (size_t i = 0; i < n; i++) {
        Sym *s = get_sym(qemu_plugin_insn_symbol(qemu_plugin_tb_get_insn(tb, i)));
        if (t->n_groups && t->g[t->n_groups - 1].sym == s) {
            t->g[t->n_groups - 1].n++;
        } else {
            t->g[t->n_groups].sym = s;
            t->g[t->n_groups].n = 1;
            t->n_groups++;
        }
    }
    g_mutex_unlock(&lock);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, t);
}

static gint by_count_desc(gconstpointer a, gconstpointer b)
{
    uint64_t ca = (*(Sym *const *)a)->count, cb = (*(Sym *const *)b)->count;
    return ca < cb ? 1 : (ca > cb ? -1 : 0);
}

static void at_exit(qemu_plugin_id_t id, void *p)
{
    GPtrArray *all = g_ptr_array_new();
    GHashTableIter it;
    gpointer k, v;
    uint64_t total = 0;
    g_hash_table_iter_init(&it, syms);
    while (g_hash_table_iter_next(&it, &k, &v)) {
        Sym *s = v;
        if (s->count) {
            g_ptr_array_add(all, s);
            total += s->count;
        }
    }
    g_ptr_array_sort(all, by_count_desc);
    FILE *f = out_path ? fopen(out_path, "w") : stderr;
    if (!f) {
        f = stderr;
    }
    fprintf(f, "%" PRIu64 "\t#total\n", total);
    for (guint i = 0; i < all->len; i++) {
        Sym *s = g_ptr_array_index(all, i);
        fprintf(f, "%" PRIu64 "\t%s\n", s->count, s->name);
    }
    if (f != stderr) {
        fclose(f);
    }
    g_ptr_array_free(all, TRUE);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                                           int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "out=")) {
            out_path = g_strdup(argv[i] + 4);
        } else {
            fprintf(stderr, "bf_symcount: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    syms = g_hash_table_new(g_str_hash, g_str_equal);
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
