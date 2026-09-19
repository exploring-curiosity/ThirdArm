"""Find locals read before assignment, using CPython's own compiler.

Rewritten: an AST walk that tries to reimplement Python's scope rules gets
loop variables, sibling branches and try/except bindings wrong, and buries a
real bug in dozens of false positives.

Instead, compile the function and read its BYTECODE. LOAD_FAST on a local
that no preceding instruction could have stored is exactly the condition
that raises UnboundLocalError, and the compiler has already done the scope
analysis correctly.

Linear scan over the instruction stream: conservative about jumps, so it
reports a name only when NO store for it appears earlier in the stream at
all -- which is the unambiguous case, and the one that actually bites.
"""
import dis, sys, types


def _walk_code(co, path, out, name=None):
    name = name or co.co_name
    # Parameters are locals that arrive already bound, with no STORE_FAST --
    # so they must be seeded, or every function reports all its arguments.
    # co_varnames starts with the arguments, in order.
    nargs = (co.co_argcount + co.co_kwonlyargcount
             + bool(co.co_flags & 0x04)      # *args
             + bool(co.co_flags & 0x08))     # **kwargs
    stored = set(co.co_varnames[:nargs])
    # Comprehensions receive their iterable as a synthetic '.0' argument.
    stored.add(".0")
    for ins in dis.get_instructions(co):
        if ins.opname in ("LOAD_FAST", "LOAD_FAST_CHECK"):
            if ins.argval not in stored:
                out.append((path, name, ins.positions.lineno, ins.argval))
        elif ins.opname in ("STORE_FAST", "DELETE_FAST"):
            stored.add(ins.argval)
        # A name bound by an inner scope or captured is not a plain local.
        elif ins.opname in ("STORE_DEREF", "LOAD_CLOSURE"):
            stored.add(ins.argval)
    for const in co.co_consts:
        if isinstance(const, types.CodeType):
            _walk_code(const, path, out, f"{name}.{const.co_name}")


def check(path):
    src = open(path).read()
    co = compile(src, path, "exec")
    out = []
    _walk_code(co, path, out)
    # Deduplicate per (function, name): one report per real problem.
    seen, uniq = set(), []
    for p, fn, line, nm in out:
        if (fn, nm) in seen:
            continue
        seen.add((fn, nm))
        uniq.append((p, fn, line, nm))
    return uniq


if __name__ == "__main__":
    fail = False
    for path in sys.argv[1:]:
        for p, fn, line, nm in check(path):
            print(f"  {p}:{line} in {fn}(): '{nm}' may be read before assignment")
            fail = True
    print("  no use-before-assignment" if not fail else "")
    sys.exit(1 if fail else 0)
