##########################################
# ----------- MODEL BENCHMARK  -----------
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  quiet = T
)

model_results <- here(
  "data", 
  "results",
  "00_benchmark_models", 
  "climate_model_oos_predictions_10-splits_2023-07-05.csv") |> 
  readr::read_csv() 

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


benchmark_summary <- levels_summary |> 
  dplyr::left_join(anomaly_summary) |> 
  dplyr::arrange(desc(test_R2)) 

readr::write_csv(
  benchmark_summary,
  here::here("data", "results", "00_benchmark_models", "benchmark_summary.csv")
)
