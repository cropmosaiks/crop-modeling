##########################################
#### FIGURE 5 - MAIZE YIELD DEMEANED #####
##########################################

####################### R ENVIRONMENT #######################
knitr::opts_chunk$set(message = FALSE, warning = FALSE)

source(here::here('code', '4_explore_results', 'utility.R'))
oos_preds <- here::here(
  "data",
  "results",
  "2_sensor_top-mod_oos_predictions_10-splits_2023-05-24_rcf_climate-False_anom-False.csv"
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
  geom_point(data = test_pred,
             aes(x = demean_log_yield, y = demean_oos_prediction, color = as.factor(year))) +
  geom_abline() +
  scale_color_viridis_d() +
  labs(color = NULL, x = 'log(1+mt/ha)', y = 'Model estimate') +
  geom_text(data = NULL, aes(x = -.2, y = .325), label = latex2exp::TeX(
    paste0(r'($R^2 = $)', test_demean_R2, r'( ()', test_demean_sem_R2, r'())')
  )) +
  geom_text(data = NULL, aes(x = -.2, y = .275), label = latex2exp::TeX(
    paste0(r'( $r^2 = $)', test_demean_r2, r'( ()', test_demean_sem_r2, r'())')
  )) +
  scale_x_continuous(limits = limits) +
  scale_y_continuous(limits = limits) +
  theme(legend.position = leg_pos
        ,legend.background = element_rect(fill = alpha(.75))
  ) 

p1 <- ggExtra::ggMarginal(
  p1, type = "histogram", 
  groupFill = T
) 

p1
