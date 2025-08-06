import numpy as np

from evallm.utils.bootstrap import boostrap_mean


def temperature_comparison_subtable(zero_results, nonzero_results):
    """
    Table with the following columns:
    - Model
    - Zero temperature results
    - Nonzero temperature results
    - Difference
    """
    def render(vals):
        mean = np.mean(vals)
        lo, hi = boostrap_mean(vals)
        return f"{mean:.2%} ({lo:.2%} -- {hi:.2%})".replace("%", r"\%")
    table = []
    for model in zero_results.keys():
        for prompt in zero_results[model].keys():
            prompt_disp = rf"\textsc{{{prompt}}}"
            zr = np.array(zero_results[model][prompt])  
            nr = np.array(nonzero_results[model][prompt])
            table.append([model, prompt_disp, render(zr), render(nr), render(nr - zr)])
    return table

def emit_table(cells):
    table = r"{\renewcommand{\arraystretch}{1.25}" + "\n"
    table += r"\begin{tabular}{llrrr}" + "\n"
    table += r"\hline" + "\n"
    for row in cells:
        table += " & ".join(row) + r" \\" + "\n"
        table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    table += r"}" + "\n"
    print(table)


def temperature_comparison_tables(sc, t):
    table = [
        [
            r"\textbf{Model}",
            r"\textbf{Prompt}",
            r"\textbf{Zero Temp}",
            r"\textbf{Nonzero Temp}",
            r"\textbf{Difference}",
        ]
    ]
    table += [[r"\multicolumn{5}{l}{\textbf{Sequence Completion}}"]]
    table += temperature_comparison_subtable(*sc)
    table += [[r"\multicolumn{5}{l}{\textbf{Transducer}}"]]
    table += temperature_comparison_subtable(*t)
    emit_table(table)