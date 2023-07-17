##########################################
# ----------- METHODS SCHEMATIC-----------
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  cowplot,
  quiet = T
)
