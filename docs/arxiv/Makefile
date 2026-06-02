# Build the arXiv preprint (single-column, ~12 pages).
# Requires: pdflatex + biblatex-free (uses thebibliography directly).

PDFLATEX = pdflatex -interaction=nonstopmode -file-line-error
DOC      = preprint

.PHONY: pdf clean distclean figures

# Regenerate figures/*.pdf from the canonical benchmark JSON (run from repo root).
figures:
	cd ../.. && python -m rca_pipeline.make_figures --out docs/arxiv/figures

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	-$(PDFLATEX) $(DOC).tex
	-$(PDFLATEX) $(DOC).tex
	@echo "Build done -- check $(DOC).pdf"
	@test -f $(DOC).pdf || (echo "PDF was not produced" && exit 1)

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	      $(DOC).lof $(DOC).lot $(DOC).bbl $(DOC).blg \
	      $(DOC).bcf $(DOC).run.xml $(DOC).fls $(DOC).fdb_latexmk

distclean: clean
	rm -f $(DOC).pdf
