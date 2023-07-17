##########################################
# ----------- MODEL SELECTION  -----------
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

##########################################
# ----------- LEVELS  --------------------
##########################################

levels_files <- list.files(
  path = here("data", "results", "01_model_selection"), 
  pattern = "anom-False",
  full.names = TRUE
)

levels_results <- readr::read_csv(levels_files) 

levels_summary <- levels_results |> 
  dplyr::group_by(
    satellite_1, 
    bands_1,
    num_features_1, 
    points_1, 
    month_range_1, 
    limit_months_1,
    crop_mask_1,
    weighted_avg_1,
    satellite_2,
    bands_2,
    num_features_2,
    points_2,
    month_range_2,
    limit_months_2,
    crop_mask_2,
    weighted_avg_2,
    hot_encode
  ) |> 
  dplyr::summarise(
    val_R2 = mean(val_R2)
  ) |> 
  dplyr::arrange(dplyr::desc(val_R2)) 

top_levels <- levels_summary |> 
  head(1) |> 
  dplyr::select(-val_R2)

hot_encode <- top_levels$hot_encode

top_levels <- top_levels %>% select(-hot_encode)

top_levels_long <- top_levels %>%
  pivot_longer(
    cols = everything(),
    names_to = c(".value", "sensor"),
    names_pattern = "^(.*)_(\\d)$"
  )

top_levels_long$hot_encode <- hot_encode

##########################################
# ----------- ANOMALIES  -----------------
##########################################

anom_files <- list.files(
  path = here("data", "results", "01_model_selection"), 
  pattern = "anom-True",
  full.names = TRUE
)

anom_results <- readr::read_csv(anom_files) 

anom_summary <- anom_results |> 
  dplyr::group_by(
    satellite_1, 
    bands_1,
    num_features_1, 
    points_1, 
    month_range_1, 
    limit_months_1,
    crop_mask_1,
    weighted_avg_1,
    satellite_2,
    bands_2,
    num_features_2,
    points_2,
    month_range_2,
    limit_months_2,
    crop_mask_2,
    weighted_avg_2,
    hot_encode
  ) |> 
  dplyr::summarise(
    val_R2 = mean(val_R2)
  ) |> 
  dplyr::arrange(dplyr::desc(val_R2)) 

top_anom <- anom_summary |> 
  head(1) |> 
  dplyr::select(-val_R2)

hot_encode <- top_anom$hot_encode

top_anom <- top_anom %>% select(-hot_encode)

top_anom_long <- top_anom %>%
  pivot_longer(
    cols = everything(),
    names_to = c(".value", "sensor"),
    names_pattern = "^(.*)_(\\d)$"
  )

top_anom_long$hot_encode <- hot_encode

