"""Report names a function reads that are neither parameters, locals, nor
module-level. Python resolves these at runtime, so py_compile and import
both pass and the failure only appears when that branch executes."""
import ast, builtins, sys

def check(path, names=None):
    tree = ast.parse(open(path).read())
    mod = {n.name for n in tree.body
           if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name): mod.add(x.id)
        elif isinstance(n,(ast.Import, ast.ImportFrom)):
            for a in n.names: mod.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.If):                      # __main__ guards etc
            for b in ast.walk(n):
                if isinstance(b, ast.Name) and isinstance(b.ctx, ast.Store):
                    mod.add(b.id)
    bad = {}
    for fn in tree.body:
        if not isinstance(fn,(ast.FunctionDef, ast.AsyncFunctionDef)): continue
        if names and fn.name not in names: continue
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        if fn.args.vararg: params.add(fn.args.vararg.arg)
        if fn.args.kwarg: params.add(fn.args.kwarg.arg)
        assigned, loaded = set(), set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                (assigned if isinstance(n.ctx, ast.Store) else loaded).add(n.id)
            elif isinstance(n,(ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.add(n.name)
                params |= {a.arg for a in n.args.args}
                params |= {a.arg for a in n.args.kwonlyargs}
            elif isinstance(n, ast.Lambda):
                params |= {a.arg for a in n.args.args}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name): assigned.add(t.id)
            elif isinstance(n,(ast.Import, ast.ImportFrom)):
                for a in n.names: assigned.add((a.asname or a.name).split(".")[0])
        free = loaded - params - assigned - mod - set(dir(builtins))
        if free: bad[fn.name] = sorted(free)
    return bad

if __name__ == "__main__":
    fail = False
    for path in sys.argv[1:]:
        bad = check(path)
        for fn, names in bad.items():
            print(f"  {path}:{fn}() -> {names}"); fail = True
    print("  no undefined names" if not fail else "")
    sys.exit(1 if fail else 0)
