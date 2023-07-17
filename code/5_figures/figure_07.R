##########################################
# ----------- BENCHMARK NDVI -------------
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  latex2exp,
  ggExtra,
  quiet = T
)

source(here::here("code", "4_results", "utility.R"))

oos_preds <- here::here(
  "data",
  "results",
  "00_benchmark_models",
  "climate_model_oos_predictions_10-splits_2023-07-05.csv"
) |> 
  readr::read_csv() |> 
  dplyr::filter(
    data_fold == "test"
    , anomaly == FALSE
    , hot_encode == TRUE
    , variables == "ndvi"
  )

test_pred <- oos_preds |> 
  dplyr::group_by(data_fold, district, year) |> 
  dplyr::summarise(
    log_yield = mean(log_yield),
    oos_prediction = mean(oos_prediction)
  )

summary_stats <- oos_preds |> 
  dplyr::group_by(data_fold, split, random_state) |> 
  dplyr::summarise(
    R2 = r2_general(log_yield, oos_prediction),
    r2 = r2_pears(log_yield, oos_prediction)
  ) |> 
  dplyr::ungroup() |> 
  dplyr::group_by(data_fold) |> 
  dplyr::summarise(
    mean_R2 = mean(R2) |> round(3), 
    mean_r2 = mean(r2) |> round(3), 
    sem_R2 = (stderror(R2) * 2) |> round(3),
    sem_r2 = (stderror(r2) * 2) |> round(3)
  ) 

test_R2 <- dplyr::pull(summary_stats, mean_R2)
test_sem_R2  <- dplyr::pull(summary_stats, sem_R2)
test_r2 <- dplyr::pull(summary_stats, mean_r2)
test_sem_r2  <- dplyr::pull(summary_stats, sem_r2)

leg_pos <- c(.89, .25)

p1 <- ggplot() +
  geom_point(data = test_pred,
             aes(x = log_yield, y = oos_prediction, color = as.factor(year))) +
  geom_abline() +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha)', y = 'Model estimate') +
  geom_text(data = NULL, aes(x = .15, y = .8), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_R2, r'( ()', test_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = .15, y = .75), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_r2, r'( ()', test_sem_r2, r'())')
  )) +
  scale_x_continuous(limits = c(0, .82)) +
  scale_y_continuous(limits = c(0, 0.82)) +
  theme_bw() +
  theme(legend.position = leg_pos
        , legend.background = element_rect(fill = alpha(.75))
  ) 

p1 <- ggExtra::ggMarginal(
  p1, type = "histogram", 
  groupFill = T
) 





test_pred <- oos_preds |> 
  dplyr::group_by(data_fold, district, year) |> 
  dplyr::summarise(
    demean_log_yield = mean(demean_log_yield), 
    demean_oos_prediction = mean(demean_oos_prediction)
  )

summary_stats <- oos_preds |> 
  dplyr::group_by(data_fold, split, random_state) |> 
  dplyr::summarise(
    demean_R2 = r2_general(demean_log_yield, demean_oos_prediction),
    demean_r2 = r2_pears(demean_log_yield, demean_oos_prediction)
  ) |> 
  dplyr::ungroup() |> 
  dplyr::group_by(data_fold) |> 
  dplyr::summarise(
    mean_demean_R2 = mean(demean_R2) |> round(3), 
    mean_demean_r2 = mean(demean_r2) |> round(3), 
    sem_demean_R2 = (stderror(demean_R2) * 2) |> round(3),
    sem_demean_r2 = (stderror(demean_r2) * 2) |> round(3)
  ) 

test_demean_R2 <- dplyr::pull(summary_stats, mean_demean_R2)
test_demean_sem_R2  <- dplyr::pull(summary_stats, sem_demean_R2)
test_demean_r2 <- dplyr::pull(summary_stats, mean_demean_r2)
test_demean_sem_r2  <- dplyr::pull(summary_stats, sem_demean_r2)

leg_pos <- c(.89, .25)
limits <- c(-0.36, 0.36)

p2 <- ggplot() +
  geom_point(data = test_pred,
             aes(x = demean_log_yield, y = demean_oos_prediction, color = as.factor(year))) +
  geom_abline() +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = NULL) +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_demean_R2, r'( ()', test_demean_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_demean_r2, r'( ()', test_demean_sem_r2, r'())')
  )) +
  scale_x_continuous(limits = limits) +
  scale_y_continuous(limits = limits) +
  theme_bw() +
  theme(legend.position = leg_pos
        , legend.background = element_rect(fill = alpha(.75))
  ) 

p2 <- ggExtra::ggMarginal(
  p2, type = "histogram", 
  groupFill = T
) 







oos_anom_preds <- here::here(
  "data",
  "results",
  "00_benchmark_models",
  "climate_model_oos_predictions_10-splits_2023-07-05.csv"
) |> 
  readr::read_csv() |> 
  dplyr::filter(
    data_fold == "test"
    , anomaly == TRUE
    , hot_encode == FALSE
    , variables == "ndvi"
  )

test_anom_pred <- oos_anom_preds |> 
  dplyr::group_by(district, year) |> 
  dplyr::summarise(
    demean_log_yield = mean(demean_log_yield),
    demean_oos_prediction = mean(demean_oos_prediction)
  )

summary_anom_stats <- oos_anom_preds |> 
  dplyr::group_by(split, random_state) |> 
  dplyr::summarise(
    R2 = r2_general(demean_log_yield, demean_oos_prediction),
    r2 = r2_pears(demean_log_yield, demean_oos_prediction)
  ) |> 
  dplyr::ungroup() |> 
  dplyr::summarise(
    mean_R2 = mean(R2) |> round(3), 
    mean_r2 = mean(r2) |> round(3), 
    sem_R2 = (stderror(R2) * 2) |> round(3),
    sem_r2 = (stderror(r2) * 2) |> round(3)
  )

test_anom_R2 <- dplyr::pull(summary_anom_stats, mean_R2)
test_anom_sem_R2  <- dplyr::pull(summary_anom_stats, sem_R2)
test_anom_r2 <- dplyr::pull(summary_anom_stats, mean_r2)
test_anom_sem_r2  <- dplyr::pull(summary_anom_stats, sem_r2)

leg_pos <- c(.89, .25)
limits <- c(-0.36, 0.36)

p3 <- ggplot() +
  geom_point(data = test_anom_pred,
             aes(x = demean_log_yield, y = oos_prediction, color = as.factor(year))) +
  geom_abline() +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = NULL) +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_anom_R2, r'( ()', test_anom_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_anom_r2, r'( ()', test_anom_sem_r2, r'())')
  )) +
  scale_x_continuous(limits = limits) +
  scale_y_continuous(limits = limits) +
  theme_bw() +
  theme(legend.position = leg_pos
        , legend.background = element_rect(fill = alpha(.75))
  ) 

p3 <- ggExtra::ggMarginal(
  p3, type = "histogram", 
  groupFill = T
) 

p4 <- cowplot::plot_grid(p1, p2, p3, labels=c("(a)", "(b)", "(c)"), ncol = 3, nrow = 1)

ggsave(
  filename = "figure_07.jpeg"
  , path = here("figures")
  , plot = p4
  , device ="jpeg"
  , width = 15
  , height = 5
  , units = "in"
)