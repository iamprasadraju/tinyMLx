"""graph nodes (UOps). one UOp per tensor op; shared by viz and (later) autodiff."""

from dataclasses import dataclass


@dataclass(eq=False)
class UOp:
    op: str | None  # "ADD"/"MATMUL"/"TRANSPOSE"...; None = leaf (LOAD)
    dtype: object
    shape: tuple
    ndim: int
    src: tuple = ()  # input UOps
    arg: object = None  # reserved for reduce axis etc.
    value: object = None

    def toposort(self) -> dict:
        """iterative post-order DFS. dict keeps insertion order (children first)."""
        cache = {}
        stack = [(self, False)]

        while stack:
            node, visited = stack.pop()

            if node in cache:
                continue

            if not visited:
                stack.append((node, True))

                for s in reversed(node.src):
                    stack.append((s, False))
            else:
                cache[node] = None

        return cache

    def __repr__(self):
        return self._repr()

    def _repr(self, indent=0):
        pad = " " * indent
        child_pad = " " * (indent + 2)

        result = (
            f"{pad}UOp({self.op},\n"
            f"{child_pad}shape={self.shape},\n"
            f"{child_pad}ndim={self.ndim},\n"
            f"{child_pad}dtype={self.dtype}"
        )

        if self.value is not None:
            result += f",\n{child_pad}value={self.value!r}"

        if self.src:
            result += ",\n"
            result += f"{child_pad}src=(\n"
            result += ",\n".join(src._repr(indent + 4) for src in self.src)
            result += f"\n{child_pad})"

        result += ")"

        return result
