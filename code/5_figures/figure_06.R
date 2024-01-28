##########################################
# ----------- MAIZE YIELD ANOMALIES ------
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

mod = "loess"

source(here::here("code", "4_results", "utility.R"))

oos_anom_preds <- here::here(
  "data",
  "results",
  "03_model_predictions", 
  "2_sensor_top-ot-mod_oos_predictions_10-splits_2023-07-17_rcf_climate-False_anom-True.csv"
) |> 
  readr::read_csv() |> 
  dplyr::filter(data_fold == "test") 

test_pred <- oos_anom_preds |> 
  dplyr::group_by(data_fold, district, year) |> 
  dplyr::summarise(
    log_yield = mean(log_yield),
    oos_prediction = mean(oos_prediction),
  )

summary_anom_stats <- oos_anom_preds |> 
  dplyr::group_by(data_fold, split, random_state) |> 
  dplyr::summarise(
    R2 = r2_general(log_yield, oos_prediction),
    r2 = r2_pears(log_yield, oos_prediction)
  ) |> 
  dplyr::ungroup() |> 
  dplyr::group_by(data_fold) |> 
  dplyr::summarise(
    mean_R2 = mean(R2) |> round(2), 
    mean_r2 = mean(r2) |> round(2), 
    sem_R2 = (stderror(R2) * 2) |> round(2),
    sem_r2 = (stderror(r2) * 2) |> round(2)
  )

test_anom_R2 <- dplyr::pull(summary_anom_stats, mean_R2)
test_anom_sem_R2  <- dplyr::pull(summary_anom_stats, sem_R2)
test_anom_r2 <- dplyr::pull(summary_anom_stats, mean_r2)
test_anom_sem_r2  <- dplyr::pull(summary_anom_stats, sem_r2)

leg_pos <- c(.89, .25)
limits <- c(-0.36, 0.36)

p1 <- ggplot() +
  geom_abline() +
  geom_point(data = test_pred,
             aes(x = log_yield, y = oos_prediction, color = as.factor(year))) +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = 'Model estimate') +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_anom_R2, r'( ()', test_anom_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_anom_r2, r'( ()', test_anom_sem_r2, r'())')
  )) +
  # geom_smooth(data = test_pred, linewidth = .5,
  #             aes(x = log_yield, y = oos_prediction
  #                 # , color = as.factor(year)
  #                 ),
  #             method = mod,  formula = 'y ~ x'
  #             # , se=F
  #             ) +
  scale_x_continuous(limits = limits) +
  scale_y_continuous(limits = limits) +
  theme_bw() +
  theme(legend.position = leg_pos
        , legend.background = element_rect(fill = alpha(.75))
  ) 

p1 <- ggExtra::ggMarginal(
  p1, type = "histogram", 
  groupFill = T
) 

# ggsave(
#   filename = "figure_06.jpeg"
#   , path = here("figures")
#   , plot = p1
#   , device ="jpeg"
#   , width = 5
#   , height = 5
#   , units = "in"
# )


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

test_pred <- oos_anom_preds |> 
  dplyr::group_by(district, year) |> 
  dplyr::summarise(
    demean_log_yield = mean(demean_log_yield),
    oos_prediction = mean(oos_prediction)
  )

summary_anom_stats <- oos_anom_preds |> 
  dplyr::group_by(split, random_state) |> 
  dplyr::summarise(
    R2 = r2_general(demean_log_yield, oos_prediction),
    r2 = r2_pears(demean_log_yield, oos_prediction)
  ) |> 
  dplyr::ungroup() |> 
  dplyr::summarise(
    mean_R2 = mean(R2) |> round(2), 
    mean_r2 = mean(r2) |> round(2), 
    sem_R2 = (stderror(R2) * 2) |> round(2),
    sem_r2 = (stderror(r2) * 2) |> round(2)
  )

test_anom_R2 <- dplyr::pull(summary_anom_stats, mean_R2)
test_anom_sem_R2  <- dplyr::pull(summary_anom_stats, sem_R2)
test_anom_r2 <- dplyr::pull(summary_anom_stats, mean_r2)
test_anom_sem_r2  <- dplyr::pull(summary_anom_stats, sem_r2)

leg_pos <- c(.89, .25)
limits <- c(-0.36, 0.36)

p2 <- ggplot() +
  geom_abline() +
  geom_point(data = test_pred,
             aes(x = demean_log_yield, y = oos_prediction, color = as.factor(year))) +
  scale_color_viridis_d(option = "cividis") +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = NULL) +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_anom_R2, r'( ()', test_anom_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_anom_r2, r'( ()', test_anom_sem_r2, r'())')
  )) +
  # geom_smooth(data = test_pred, linewidth = .5,
  #             aes(x = demean_log_yield, y = oos_prediction
  #                 # , color = as.factor(year)
  #                 ),
  #             method = mod,  formula = 'y ~ x'
  #             # , se=F
  #             ) +
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

p3 <- cowplot::plot_grid(p1, p2, labels=c("(a)", "(b)"), ncol = 2, nrow = 1)
p3
ggsave(
  filename = "figure_06.jpeg"
  , path = here("figures")
  , plot = p3
  , device ="jpeg"
  , width = 10
  , height = 5
  , units = "in"
)
