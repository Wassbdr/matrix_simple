# Build the LaTeX report (FR, ~45 pages).
# Requires: pdflatex + biber + biblatex + babel-french + booktabs + tikz + listings + hyperref.

PDFLATEX = pdflatex -interaction=nonstopmode -file-line-error
BIBER    = biber
DOC      = rapport

.PHONY: pdf clean distclean

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex bibliography.bib
	-$(PDFLATEX) $(DOC).tex
	-$(BIBER) $(DOC)
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
