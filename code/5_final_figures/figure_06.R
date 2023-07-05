##########################################
#### FIGURE 6 - MAIZE YIELD ANOMALIES ####
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

r2_general <-function(actual, predictions) { 
  r2 <- 1 - sum((predictions - actual) ^ 2) / sum((actual - mean(actual))^2)
  return(r2)
}

r2_pears <- function(actual, predictions) { 
  r2 <- cor(actual, predictions) ^ 2
  return(r2)
}

stderror <- function(x) { 
  sd(x)/sqrt(length(x))
}

oos_anom_preds <- here::here(
  "data",
  "results",
  "2_sensor_top-mod_oos_predictions_10-splits_2023-06-08_rcf_climate-False_anom-True.csv"
) |> 
  readr::read_csv() |> 
  dplyr::filter(data_fold == "test") 

test_anom_pred <- oos_anom_preds |> 
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

p1 <- ggplot() +
  geom_point(data = test_anom_pred,
             aes(x = log_yield, y = oos_prediction, color = as.factor(year))) +
  geom_abline() +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha) - mean(log(1+mt/ha))', y = 'Model estimate') +
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

p1 <- ggExtra::ggMarginal(
  p1, type = "histogram", 
  groupFill = T
) 

ggsave(
  filename = "figure_06.jpeg"
  , path = here("figures")
  , plot = p1
  , device ="jpeg"
  , width = 5
  , height = 5
  , units = "in"
)
