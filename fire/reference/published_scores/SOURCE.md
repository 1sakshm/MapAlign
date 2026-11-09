# Published per-pair FIRE errors

Downloaded on 2026-11-10 from the FIRE page's "Download scores" links
(`https://projects.ics.forth.gr/cvrl/fire/results/<file>.txt`). The live server was unreachable, so the files were
retrieved from the Internet Archive's copy of those URLs. Each line is `pair_id mean_error_px`, and `NaN` marks a
failed registration. The FIRE page asks for the corresponding papers to be cited when these results are used:

| File | Method | Paper |
|---|---|---|
| `HernandezMatas_JBHI_2020.txt` | REMPE | Hernandez-Matas, Zabulis, Argyros, IEEE JBHI 2020, DOI 10.1109/JBHI.2020.2984483 |
| `HernandezMatas_EMBC_2016.txt` | H-M 16 | Hernandez-Matas, Zabulis, Argyros, IEEE EMBC 2016, DOI 10.1109/EMBC.2016.7591421 |
| `Chen_TBME_2010.txt` | Harris-PIIFD | Chen et al., IEEE TBME 57(7), 2010, DOI 10.1109/TBME.2010.2042169 |
| `Yang_TPAMI_2007.txt` | GDB-ICP | Yang, Stewart, Sofka, Tsai, IEEE TPAMI 29(11), 2007, DOI 10.1109/TPAMI.2007.1116 |

They are used only to check how the FIRE AUC is computed (`fire/ANALYSIS_PLAN.md` §8).
