"""The C string-literal lexer the name pin reads C files with (#287: split out of
tests/integration/test_name_strings_resolve.py, which keeps its controls).

`c_string_literals(source)` yields (line, decoded value) for every string literal in C
`source`, comments excluded and `#define` bodies included. The pin's C controls, in
tests/integration/test_name_strings_resolve.py, are what turn red when this lexer
breaks: they stay beside the resolver on purpose, because that file's main pin skips
exactly one file (itself) for the failing names its controls spell.
"""

# What a C escape sequence stands for; any other escaped character (`\"`, `\\`, `\'`)
# stands for itself. Numeric escapes are not decoded (see LIMITS in
# tests/integration/test_name_strings_resolve.py).
_C_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def c_string_literals(source: str):
    """Yield (line, value) for each string literal in C `source`, comments excluded.

    A hand-written state machine: code, a string or char literal, a `//` comment, a
    `/* */` comment. Preprocessor lines are scanned as code, so the strings of a
    `#define` body count (binding.c and cs2_types.h hold such macro strings). An
    escape is decoded to the character C sees; a char literal is skipped. A literal
    ends at its closing quote or at an unescaped newline (C rejects an unterminated
    literal, so that newline only occurs in text C never compiles, such as an
    apostrophe in an `#error` line), so a stray quote swallows one line at most.
    WHY hand-written: the maintained lexers are blind to macro bodies (measured,
    #205 part 2b). ast-grep/tree-sitter-c keep a `#define` body as one raw
    `preproc_arg` node, and pygments' CLexer yields it as Comment.Preproc; pycparser
    and libclang need preprocessed input or a new dependency.
    PITFALL: a `'"'` char literal must not open a string, and `//` or `/*` inside a
    string must not open a comment; the C controls in
    tests/integration/test_name_strings_resolve.py pin both.
    """
    i, n, line = 0, len(source), 1
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end                  # the newline is counted above
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += source.count("\n", i, end)
            i = end
        elif c in "\"'":
            first, value, i = line, [], i + 1
            while i < n and source[i] not in (c, "\n"):
                if source[i] == "\\" and i + 1 < n:
                    if source[i + 1] == "\n":          # a line continuation
                        line += 1
                    else:
                        value.append(_C_ESCAPES.get(source[i + 1], source[i + 1]))
                    i += 2
                else:
                    value.append(source[i])
                    i += 1
            if i < n and source[i] == c:
                i += 1
            if c == '"':
                yield first, "".join(value)
        else:
            i += 1
