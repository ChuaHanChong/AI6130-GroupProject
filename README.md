# AI6130 Group Project

## Development

```bash
git clone --recurse-submodules git@github.com:ChuaHanChong/AI6130-GroupProject.git
```

```bash
conda create --name AI6130 python=3.12
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e transformers
pip install -r requirements.txt
```

```bash
git rm -f path/to/submodule
rm -rf .git/modules/path/to/submodule
```

## References

- https://github.com/kotanarik/RAGBenchAndRGB/tree/main#
- https://github.com/RAGgroup27/RAG_project1#
- https://github.com/C23RAGGroup4/RAGBench#
- https://github.com/camel-ai/camel/wiki/RAG-Cookbook
- https://github.com/camel-ai/camel/blob/master/camel/benchmarks/ragbench.py#L319
