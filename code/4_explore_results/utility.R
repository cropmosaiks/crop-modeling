####################### LOAD PACKAGES #######################
if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  plotly,
  tidyverse,
  lubridate,
  here,
  rlang,
  latex2exp,
  arrow,
  sf,
  terra,
  tidyterra,
  ggExtra,
  cowplot,
  rnaturalearth,
  rnaturalearthdata,
  ggspatial,
  quiet = T
)
####################### FILE NAMES #######################
one_sensor_date  <- "2023-03-12"
one_anomaly_date <- "2022-12-25"
two_sensor_date  <- "2023-04-29"
two_anomaly_date <- "2022-12-18"

one_sensor_ndvi_date  <- "2023-01-14"
two_sensor_ndvi_date  <- "2023-01-14"

one_sensor_pre_tmp_ndvi_date  <- "2023-01-14"
two_sensor_pre_tmp_ndvi_date  <- "2023-01-14"

one_sensor_fn  <- paste0("results_", one_sensor_date, ".csv")
two_sensor_fn  <- paste0("2_sensor_results_", two_sensor_date, ".csv")
one_anomaly_fn <- paste0("anomaly_results_", one_anomaly_date, ".csv")
two_anomaly_fn <- paste0("2_sensor_anomaly_results_", two_anomaly_date, ".csv")
pred_suffix   <- 'landsat-8-c2-l2_bands-1-2-3-4-5-6-7_ZMB_20k-points_1000-features_yr-2013-2021_mn-4-9_lm-True_cm-True_wa-True'
# high_res_fn   <- paste0('high-res-pred_k-fold-cv_', pred_suffix, '_he-True.feather')

one_sensor_ndvi_fn  <- paste0("results_", one_sensor_ndvi_date, "_ndvi.csv")
two_sensor_ndvi_fn  <- paste0("2_sensor_results_", two_sensor_ndvi_date, "_ndvi.csv")

one_sensor_pre_tmp_ndvi_fn  <- paste0("results_", one_sensor_pre_tmp_ndvi_date, "_pre_tmp_ndvi.csv")
two_sensor_pre_tmp_ndvi_fn  <- paste0("2_sensor_results_", two_sensor_pre_tmp_ndvi_date, "_pre_tmp_ndvi.csv")


lsc <- 'landsat-c2-l2'
ls8 <- 'landsat-8-c2-l2'
s2 <- 'sentinel-2-l2a'

####################### DATA #######################
one_sensor_results <- here::here("data", "results", one_sensor_fn) %>% 
  read_csv(show_col_types = FALSE, col_select = -1) 

two_sensor_results <- here::here("data", "results", two_sensor_fn) %>%
  read_csv(show_col_types = FALSE, col_select = -1) %>% 
  mutate(
    satellite = case_when(
      satellite_1 == satellite_2 ~ paste0(satellite_1,' with ',satellite_2),
      (satellite_1 == lsc & satellite_2 == ls8) |
        (satellite_2 == lsc & satellite_1 == ls8) ~ paste0(lsc,' with ',ls8),
      (satellite_1 == lsc & satellite_2 == s2) |
        (satellite_2 == lsc & satellite_1 == s2) ~ paste0(lsc,' with ',s2),
      (satellite_1 == ls8 & satellite_2 == s2) |
        (satellite_2 == ls8 & satellite_1 == s2) ~ paste0(ls8,' with ',s2)),
    month_range = case_when(
      month_range_1 == '4-9' & month_range_2 == '4-9' ~ '4-9',
      month_range_1 == '1-12' & month_range_2 == '1-12' ~ '1-12',
      T ~ 'mixed'),
    crop_mask = case_when(
      crop_mask_1 == T & crop_mask_2 == T ~ 'TRUE',
      crop_mask_1 == F & crop_mask_2 == F ~ 'FALSE',
      T ~ 'mixed'),
    weighted_avg = case_when(
      weighted_avg_1 == T & weighted_avg_2 == T ~ 'TRUE',
      weighted_avg_1 == F & weighted_avg_2 == F ~ 'FALSE',
      T ~ 'mixed'))  

one_anomaly_results <- here::here("data", "results", one_anomaly_fn) %>% 
  read_csv(show_col_types = FALSE, col_select = -1)

two_anomaly_results <- here::here("data", "results", two_anomaly_fn) %>% 
  read_csv(show_col_types = FALSE, col_select = -1) %>%
  mutate(
    satellite = case_when(
      satellite_1 == satellite_2 ~ paste0(satellite_1,' with ',satellite_2),
      (satellite_1 == lsc & satellite_2 == ls8) |
        (satellite_2 == lsc & satellite_1 == ls8) ~ paste0(lsc,' with ',ls8),
      (satellite_1 == lsc & satellite_2 == s2) |
        (satellite_2 == lsc & satellite_1 == s2) ~ paste0(lsc,' with ',s2),
      (satellite_1 == ls8 & satellite_2 == s2) |
        (satellite_2 == ls8 & satellite_1 == s2) ~ paste0(ls8,' with ',s2)),
    month_range = case_when(
      month_range_1 == '4-9' & month_range_2 == '4-9' ~ '4-9',
      month_range_1 == '1-12' & month_range_2 == '1-12' ~ '1-12',
      T ~ 'mixed'),
    crop_mask = case_when(
      crop_mask_1 == T & crop_mask_2 == T ~ 'TRUE',
      crop_mask_1 == F & crop_mask_2 == F ~ 'FALSE',
      T ~ 'mixed'),
    weighted_avg = case_when(
      weighted_avg_1 == T & weighted_avg_2 == T ~ 'TRUE',
      weighted_avg_1 == F & weighted_avg_2 == F ~ 'FALSE',
      T ~ 'mixed')) 

# high_res_predictions <- here::here('data', 'results', high_res_fn) %>% 
#   arrow::read_feather()

####################### LEVEL PREDICTIONS ####################### 
summary_fn    <- 'predictions_fn-1_landsat-8-c2-l2_1-2-3-4-5-6-7_15_True_False_False_fn-2_sentinel-2-l2a_2-3-4-8_15_False_True_False_he-True.csv'

summary_predictions <- here::here('data', 'results', summary_fn) %>% 
  readr::read_csv() %>% 
  mutate(split = factor(split, levels = c('train', 'test')))

train_pred <- summary_predictions %>% 
  dplyr::filter(split == 'train') %>%
  dplyr::group_by(district) %>% 
  dplyr::mutate(
    avg_yield = mean(log_yield),
    avg_pred = mean(prediction),
    demean_yield = log_yield - avg_yield,
    demean_pred = prediction - avg_pred,
    # demean_cv_pred = kfold_cv_predictions - avg_pred
    
    avg_cv_pred = mean(kfold_cv_predictions, na.rm = T),
    demean_cv_pred = kfold_cv_predictions - avg_cv_pred
  ) %>% 
  dplyr::filter(split == 'train') 

test_pred <- summary_predictions %>% 
  dplyr::filter(split == 'test') %>%
  dplyr::group_by(district) %>% 
  dplyr::mutate(
    avg_yield = mean(log_yield),
    avg_pred = mean(prediction),
    demean_yield = log_yield - avg_yield,
    demean_pred = prediction - avg_pred,
  )  %>% 
  dplyr::filter(split == 'test') 

####################### ANOMALY PREDICTIONS - BEST MOD ####################### 
summary_an_fn <- 'anomaly-predictions_fn-1_landsat-8-c2-l2_1-2-3-4-5-6-7_15_True_False_False_fn-2_sentinel-2-l2a_2-3-4-8_15_False_True_False.csv'

summary_anomaly_predictions <- here::here('data', 'results', summary_an_fn) %>% 
  readr::read_csv() %>% 
  mutate(split = factor(split, levels = c('train', 'test')))

anom_train_pred <- summary_anomaly_predictions %>% 
  dplyr::filter(split == 'train')  

anom_test_pred <- summary_anomaly_predictions %>% 
  dplyr::filter(split == 'test')

####################### ANOMALY PREDICTIONS - BEST OVERALL ####################### 
best_an_fn <- 'best-anomaly-predictions_fn-1_landsat-c2-l2_r-g-b-nir-swir16-swir22_20_True_True_False_fn-2_sentinel-2-l2a_2-3-4_4_False_True_False.csv'

best_anomaly_predictions <- here::here('data', 'results', best_an_fn) %>%
  readr::read_csv() %>%
  mutate(split = factor(split, levels = c('train', 'test')))

best_anom_train_pred <- best_anomaly_predictions %>%
  dplyr::filter(split == 'train')

best_anom_test_pred <- best_anomaly_predictions %>%
  dplyr::filter(split == 'test')

####################### SPATIAL DATA ####################### 
crop_land <- here::here("data", "land_cover", "ZMB_cropland_2019_cropped.tif") %>% 
  terra::rast() 

country_shp <- here::here('data', 'geo_boundaries', 'gadm36_ZMB_2.shp') %>% 
  sf::read_sf()

zmb_union <- terra::vect(country_shp) %>% 
  terra::buffer(0.1) %>%
  terra::aggregate()

yield_sf <- summary_predictions %>% 
  mutate(residuals = prediction - log_yield) %>% 
  left_join(country_shp) %>% 
  sf::st_as_sf()

# high_res_predictions <- high_res_predictions %>% 
#   mutate(
#     prediction = case_when(
#       prediction > 1.1 ~ 1, 
#       prediction < 0 ~ 0,
#       T ~ prediction)) 

dummy_df <- country_shp %>% 
  tibble::as_tibble() %>% 
  dplyr::select(district) 


####################### CROP YIELD ####################### 
crop_yield <- here::here('data', 'crop_yield',
                         'cfs_maize_districts_zambia_2009_2022.csv') %>% 
  readr::read_csv() %>% 
  dplyr::select(year, district, yield_mt)

####################### CLIMATE ####################### 
# custom_months <- month.abb
custom_months <- c("Oct", "Nov", "Dec", "Jan", "Feb", "Mar", 
                   "Apr", "May", "Jun", "Jul", "Aug", "Sep")

zmb_precip_summary <- here::here('data', 'climate', 'precipitation_monthly_mean.csv') %>% 
  readr::read_csv() %>% 
  mutate(month = factor(month, levels = custom_months))

####################### HELPER FUNCTIONS ####################### 

colors <- c("TRUE" = 'dodgerblue', "FALSE" = "firebrick",
            'mixed' = 'darkorchid',
            "4-9" = 'dodgerblue', "1-12" = "firebrick")

expand_field <- function(fld) {
  switch(
    rlang::as_name(fld),
    'val_R2'    = 'K-Fold Validation Score (R$^2$)',
    'logo_val_R2'     = 'LOGO Validation Score (R$^2$)',
    'kfold_demean_R2' = 'K-Fold De-meaned Score (R$^2$)',
    'logo_demean_R2'  = 'LOGO De-meaned Score (R$^2$)',
    'month_range'     = 'Month\nRange',
    'hot_encode'      = 'One Hot Encoding',
    'crop_mask'       = 'Crop Mask',
    'weighted_avg'    = 'Weighted Average',
    rlang::as_name(fld)
  ) %>% latex2exp::TeX()
}

theme_set(theme_bw())
theme_update()

hist_plot <- function(data, x, y_lims = NULL) {
  x     <- enquo(x)
  x_lab <- expand_field(x)
  
  if((deparse(substitute(data))=='two_anomaly_results') |
     (deparse(substitute(data))=='two_sensor_results')){
    n_sensors = 2} else {n_sensors=1}
  if(n_sensors == 1){clr<-'dodgerblue'}else{clr<-'darkorchid'}
  if(n_sensors == 1){n_bins<-30}else{n_bins<-60}
  expansion <- expansion(mult = c(.01, .01))
  
  ggplot(data = data) +
    aes(x = !!x) +
    geom_histogram(bins = n_bins, fill = clr, color = 'black') +
    labs(x = x_lab) +
    scale_x_continuous(limits = y_lims, expand = expansion)
}

dist_plot <- function(data, x, y, clr, p_type = "box", y_lims = NULL) {
  x     <- enquo(x)
  y     <- enquo(y)
  clr   <- enquo(clr)
  x_lab <- expand_field(x)
  y_lab <- expand_field(y)
  c_lab <- expand_field(clr)
  
  base  <- ggplot(data) +
    aes(!!x, !!y, color = !!clr) +
    scale_color_manual(values = colors, limits = force) +
    labs(x = x_lab, y = y_lab, color = c_lab) + 
    facet_wrap(~satellite) +
    scale_y_continuous(limits = y_lims, expand = expansion(mult = c(.01, .01)))
  
  if (p_type == "box") {
    base + geom_boxplot() 
  } else {
    jd  <- position_jitterdodge(.2)
    base + geom_point(position = jd)
  }
}

r2_general <-function(actual, predictions, round = 2) { 
  r2 <- 1 - sum((predictions - actual) ^ 2) / sum((actual - mean(actual))^2)
  return(round(r2, round))
}

r2_pears <- function(actual, predictions) { 
  r2 <- cor(actual, predictions) ^ 2
  return(round(r2, 2))
}

pred_plot <- function(.data, x, y, label = NULL, x_lims = c(0, .82),
                      y_lims = c(0, .82), label_pos = c(.05, .75)){
  x <- enquo(x)
  y <- enquo(y)
  p <- ggplot() +
    geom_point(data = .data,
               aes(x = !!x, y = !!y, color = as.factor(year))) +
    geom_abline() +
    scale_color_viridis_d() +
    labs(color = NULL, x = 'log(1+mt/ha)', y = 'Model estimate') +
    geom_text(data = NULL, aes(x = label_pos[1], y = label_pos[2]), label = label) +
    scale_x_continuous(limits = x_lims) +
    scale_y_continuous(limits = y_lims) +
    theme(legend.position = c(.9, .35)
          ,legend.background = element_rect(fill = alpha(.75))
    ) 
  
  ggExtra::ggMarginal(
    p, type = "histogram", 
    groupFill = T
  )  
}

latex_new_lines <- function(each_line) { 
  if ("character" != class(each_line)) {
    stop("latex_lines expects a character vector")
  }
  ret <- paste0("\\normalsize{$", each_line[1], "$}")
  while(0 != length(each_line <- tail(each_line, n=-1))) {
    ret <- paste0("\\overset{", ret, "}{\\normalsize{$", each_line[1],"$}}")
  }
  return(ret)
}

plot_label_2 <- function(r, n) {
  latex2exp::TeX(
    "$\\overset{R^2 = \\r2}{n = \\n2}$", 
    user_defined = list("\\r2"=r, "\\n2" = n))
}

plot_label_3 <- function(R2, r2, n) {
  TeX(latex_new_lines(c('R^2=\\R2','r^2=\\r2', 'n=\\n')), 
      user_defined = list("\\R2" = R2, "\\r2" = r2, "\\n" = n))
}






