##########################################
# ----------- OVER TIME RESULTS ----------
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  quiet = T
)

source(here::here("code", "4_results", "utility.R"))

files <- list.files(
  path = here("data", "results", "02_model_results"), 
  pattern = "top-ot-mod",
  full.names = TRUE
)

model_results <- readr::read_csv(files) 

levels_summary <- model_results |> 
  dplyr::group_by(variables) |> 
  dplyr::summarise(
    anomaly_test_R2 = mean(test_R2),
    anomaly_test_r2 = mean(test_r2)
  ) 
