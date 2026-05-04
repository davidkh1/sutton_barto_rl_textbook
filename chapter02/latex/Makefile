BUILDDIR ?= ../output
TEXFILES = $(wildcard *.tex)
PDFS = $(patsubst %.tex,$(BUILDDIR)/%.pdf,$(TEXFILES))

all: $(PDFS)

$(BUILDDIR)/%.pdf: %.tex | $(BUILDDIR)
	pdflatex -output-directory=$(BUILDDIR) $<

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean
