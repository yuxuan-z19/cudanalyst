"""
WARNING: PORTED FROM CUGEDIT, NOT INCLUDED IN CUDANALYST AT THE MOMENT
"""

import re
import subprocess
import sysconfig
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import override

import networkx as nx
import pydot
from grakel import Graph, graph_from_networkx
from grakel.kernels import (
    GraphletSampling,
    Kernel,
    Propagation,
    SubgraphMatching,
    WeisfeilerLehman,
)
from torch.utils.cpp_extension import include_paths

GRAPHLIKE = tuple[str, Graph]


class Grapher(ABC):
    def __init__(self, arch: str, verbose: bool = False):
        self.arch = arch
        self.verbose = verbose
        self.cwd = Path.cwd()

    def _load_dot_graph(self, dot_file: Path) -> list[GRAPHLIKE]:
        dot_graphs = pydot.graph_from_dot_file(str(dot_file)) or []
        results = []
        for dot_graph in dot_graphs:
            graphs = [dot_graph] + dot_graph.get_subgraphs()
            for g in graphs:
                nxg = nx.DiGraph(nx.nx_pydot.from_pydot(g))
                for _, attr in nxg.nodes(data=True):
                    attr.setdefault("label", "")
                g = next(
                    graph_from_networkx([nxg], node_labels_tag="label", as_Graph=True)
                )

                # * get readable name with cxxfilt.demangle()
                m = re.search(r"(_Z[\w\d<>:]*)", nxg.name)
                name = m.group(1) if m else nxg.name
                results.append((name, g))

        return results

    def cleanup(self, extra_files: list[Path]) -> None:
        def _rm(f: Path) -> None:
            try:
                f.unlink()
            except FileNotFoundError:
                pass

        files = chain(self.cwd.glob("*.dot"), self.cwd.glob("*.tmp"), extra_files)
        with ThreadPoolExecutor() as executor:
            executor.map(_rm, files)

    @abstractmethod
    def process(self, dir: str | Path) -> list[GRAPHLIKE]:
        raise NotImplementedError("Subclasses should implement this method.")


class HostGrapher(Grapher):
    TORCH_INCLUDES = [f"-I{path}" for path in include_paths("cuda")] + [
        f"-I{sysconfig.get_path('include', scheme='posix_prefix')}"
    ]

    def __init__(self, arch: str = "sm_80", verbose: bool = False):
        super().__init__(arch, verbose)

    def process_ll_file(self, ll_file: Path, target_funcs: set[str]):
        func_pattern = re.compile(
            r'distinct !DISubprogram\(.*?name: "(.*?)"', re.DOTALL
        )

        content = ll_file.read_text()

        found_func_names = set(func_pattern.findall(content))
        funcs_to_process = target_funcs & found_func_names

        return list(
            map(
                lambda func_name: [
                    "opt",
                    "-passes=dot-cfg",
                    f"-cfg-func-name={func_name}",
                    ll_file.name,
                    "-disable-output",
                ],
                funcs_to_process,
            )
        )

    @override
    def process(self, target_funcs: list[str], dir: str | Path) -> list[GRAPHLIKE]:
        if isinstance(target_funcs, str):
            target_funcs = [target_funcs]
        target_funcs = set(target_funcs)

        self.cwd = Path(dir).resolve() if dir else Path.cwd()
        ll_files = []

        try:
            cuda_files = list(self.cwd.glob("*.cu"))

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        subprocess.run,
                        [
                            "clang++",
                            f"--cuda-gpu-arch={self.arch}",
                            *self.TORCH_INCLUDES,
                            "-g",
                            "-S",
                            "-emit-llvm",
                            cuda_file.name,
                        ],
                        cwd=self.cwd,
                        check=True,
                    )
                    for cuda_file in cuda_files
                ]
                for fut in as_completed(futures):
                    fut.result()

            ll_files = list(self.cwd.glob("*.ll"))
            commands = []
            for ll_file in ll_files:
                commands.extend(self.process_ll_file(ll_file, target_funcs))

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(subprocess.run, cmd, cwd=self.cwd, check=True)
                    for cmd in commands
                ]
                for fut in as_completed(futures):
                    fut.result()

            dot_files = list(self.cwd.glob("*.dot"))
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self._load_dot_graph, dot_file)
                    for dot_file in dot_files
                ]
                host_cfg_list = []
                for fut in as_completed(futures):
                    host_cfg_list.extend(fut.result())

            return host_cfg_list

        finally:
            self.cleanup(ll_files + list(self.cwd.glob("*.bc")))


class KernGrapher(Grapher):
    def __init__(self, arch: str = "sm_80", verbose: bool = False):
        super().__init__(arch, verbose)

    def _process_cubin(self, args: tuple[Path, Path]) -> tuple[Path, list[GRAPHLIKE]]:
        cubin, cwd = args
        dot_file = cubin.with_suffix(".dot")

        with open(dot_file, "w") as f:
            subprocess.run(
                ["nvdisasm", "-bbcfg", cubin.name],
                cwd=cwd,
                stdout=f,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )

        return self._load_dot_graph(dot_file)

    @override
    def process(self, dir: str | Path | None = None) -> list[GRAPHLIKE]:
        self.cwd = Path(dir).resolve() if dir else Path.cwd()
        cubin_files = []

        try:
            obj_files = list(self.cwd.glob("*.o"))
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        subprocess.run,
                        ["cuobjdump", obj.name, "-xelf", "all"],
                        cwd=self.cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        text=True,
                    ): obj.name
                    for obj in obj_files
                }
                for fut in as_completed(futures):
                    obj_name = futures[fut]
                    try:
                        fut.result()
                    except subprocess.CalledProcessError as e:
                        if self.verbose:
                            warnings.warn(
                                f"Failed cuobjdump on {obj_name}: {e.stderr}",
                                UserWarning,
                            )

            cubin_files = list(self.cwd.glob("*.cubin"))
            target_cubins: list[Path] = [
                f for f in cubin_files if f.name.endswith(f".{self.arch}.cubin")
            ]

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_cubin, (cubin, self.cwd))
                    for cubin in target_cubins
                ]
                kern_cfg_list = []
                for fut in as_completed(futures):
                    kern_cfg_list.extend(fut.result())

            return kern_cfg_list

        finally:
            self.cleanup(cubin_files)


KernelLike = Kernel | Iterable[Kernel]


class GraphDiff:
    # ^ Reference: https://github.com/yuxuan-z19/diffonnx/blob/main/src/diffonnx/static.py

    DEFAULT_KERNEL_CLASSES: list[type[Kernel]] = [
        WeisfeilerLehman,
        GraphletSampling,
        SubgraphMatching,
        Propagation,
    ]

    def __init__(
        self,
        kernels: Iterable[Kernel] | None = None,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self.kernels: dict[str, Kernel] = (
            {self._get_name(k): k for k in kernels}
            if kernels is not None
            else self._make_default_kernels()
        )

    def _make_default_kernels(self) -> dict[str, Kernel]:
        return {
            cls.__name__: cls(normalize=True) for cls in self.DEFAULT_KERNEL_CLASSES
        }

    def _get_name(self, kernel: Kernel) -> str:
        if not isinstance(kernel, Kernel):
            raise TypeError(f"Expected Kernel, got {type(kernel)}")
        return kernel.__class__.__name__

    def add_kernels(self, kernels: KernelLike) -> None:
        if not isinstance(kernels, Iterable):
            kernels = [kernels]

        for kernel in kernels:
            name = self._get_name(kernel)
            operation = "Replaced" if name in self.kernels else "Added"
            self.kernels[name] = kernel
            if self._verbose:
                print(f"<GraphDiff> {operation} kernel: {name}")

    def remove_kernels(self, kernels: KernelLike):
        if not isinstance(kernels, Iterable):
            kernels = [kernels]

        for kernel in kernels:
            name = self._get_name(kernel)
            if name in self.kernels:
                del self.kernels[name]
                if self._verbose:
                    print(f"<GraphDiff> Removed kernel: {name}")
            else:
                if self._verbose:
                    print(f"<GraphDiff> Kernel {name} not found, skipping.")

    def score(self, a_graph: Graph, b_graph: Graph) -> dict[str, float]:
        graph_kernel_scores: dict[str, float] = {}

        for name, kernel in self.kernels.items():
            try:
                kernel.fit_transform([a_graph])
                score = kernel.transform([b_graph])[0][0]
                graph_kernel_scores[name] = score
                if self._verbose:
                    print(f"<GraphDiff> Kernel {name}: score = {score:.4f}")
            except Exception as e:
                raise RuntimeError(
                    f"<GraphDiff> Kernel {name} failed during scoring: {e}"
                )

        return graph_kernel_scores

    def __len__(self) -> int:
        return len(self.kernels)
