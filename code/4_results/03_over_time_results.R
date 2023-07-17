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
  pattern = "top-ot-mod_10-splits_2023-07-17",
  full.names = TRUE
)

model_results <- readr::read_csv(files) 

levels_summary <- model_results |> 
  dplyr::filter(anomaly == FALSE, hot_encode == TRUE) |> 
  dplyr::group_by(variables) |> 
  dplyr::summarise(
    test_R2 = mean(test_R2),
    test_r2 = mean(test_r2),
    demean_test_R2 = mean(demean_test_R2),
    demean_test_r2 = mean(demean_test_r2)
  ) 

anomaly_summary <- model_results |> 
  dplyr::filter(anomaly == TRUE, hot_encode == FALSE) |> 
  dplyr::group_by(variables) |> 
  dplyr::summarise(
    anomaly_test_R2 = mean(test_R2),
    anomaly_test_r2 = mean(test_r2)
  ) 

over_time_summary <- levels_summary |> 
  dplyr::left_join(anomaly_summary) |> 
  dplyr::arrange(desc(test_R2))  |> 
  dplyr::select(
    variables, test_R2, demean_test_R2, anomaly_test_R2,
    test_r2, demean_test_r2, anomaly_test_r2,
  )

readr::write_csv(
  over_time_summary,
  here::here("data", "results", "02_model_results", "over_time_summary.csv")
)
