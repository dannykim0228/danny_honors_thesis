
render:
	Rscript -e "quarto::quarto_render('$(file)')"
# Usage:
# make render file=path/to/your/file.qmd
#

activate:
	source .venv/bin/activate
