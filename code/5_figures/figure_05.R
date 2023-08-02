##########################################
# ----------- MAIZE YIELD DEMEANED -------
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

mod = "loess"

oos_preds <- here::here(
  "data",
  "results",
  "03_model_predictions", 
  "2_sensor_top-mod_oos_predictions_10-splits_2023-07-06_rcf_climate-False_anom-False.csv"
) |> 
  readr::read_csv() |> 
  dplyr::filter(data_fold == "test") 

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

p1 <- ggplot() +
  geom_abline() +
  geom_point(data = test_pred,
             aes(x = demean_log_yield, y = demean_oos_prediction, color = as.factor(year))) +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = 'Demeaned model estimate') +
  geom_text(data = NULL, aes(x = -.24, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_demean_R2, r'( ()', test_demean_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.24, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_demean_r2, r'( ()', test_demean_sem_r2, r'())')
  )) +
  geom_smooth(data = test_pred, linewidth = .5,
              aes(x = demean_log_yield, y = demean_oos_prediction
                  # , color = as.factor(year)
                  ),
              method = mod,  formula = 'y ~ x', se=T) +
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
#   filename = "figure_05.jpeg"
#   , path = here("figures")
#   , plot = p1
#   , device ="jpeg"
#   , width = 5
#   , height = 5
#   , units = "in"
# )


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
  geom_abline() +
  geom_point(data = test_pred, #alpha = 0.8,
             aes(x = demean_log_yield, y = demean_oos_prediction, color = as.factor(year))) +
  scale_color_viridis_d(option="cividis") +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = NULL) +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_demean_R2, r'( ()', test_demean_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_demean_r2, r'( ()', test_demean_sem_r2, r'())')
  )) +
  geom_smooth(data = test_pred, linewidth = .5,
              aes(x = demean_log_yield, y = demean_oos_prediction
                  # , color = as.factor(year)
                  ),
              method = mod,  formula = 'y ~ x', se=T) +
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
  filename = "figure_05_alt.jpeg"
  , path = here("figures")
  , plot = p3
  , device ="jpeg"
  , width = 10
  , height = 5
  , units = "in"
)
