# Relatório IEEE

Artigo em formato IEEE de duas colunas (`IEEEtran`, modo `conference`), em português.

| Arquivo | Conteúdo |
|---|---|
| `main.tex` | Texto completo do relatório |
| `figuras/` | Figuras em PDF (cópia de `results/figures/campaign_full_30/report/`) |

Todos os números vêm da campanha `campaign_full_30` (`results/statistical_analysis.json` e `results/metrics.csv`).

## Compilar no Overleaf (recomendado)

1. *New Project → Upload Project* e envie um `.zip` com `main.tex` e a pasta `figuras/`.
2. Compilador: **pdfLaTeX** (padrão).
3. Compile duas vezes para resolver as referências cruzadas.

## Compilar localmente

Requer uma distribuição LaTeX (MiKTeX ou TeX Live):

```bash
cd docs/relatorio
latexmk -pdf main.tex
```

## Se os resultados mudarem

```bash
python scripts/analyze_results.py --run-id <RUN_ID>
python scripts/make_report_figures.py --run-id <RUN_ID>
cp results/figures/<RUN_ID>/report/*.pdf docs/relatorio/figuras/
```

Depois atualize os números no texto e nas tabelas de `main.tex`.

## Pendências antes da entrega

- [ ] Completar a lista de autores e e-mails (bloco `\author`)
- [ ] Revisar o texto
- [ ] Conferir se o professor exige um número máximo de páginas
