##########################################
# ----------- ZAMBIA PRECIPITATION -------
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  quiet = T
)

custom_months <- c("Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
                   "Apr", "May", "Jun", "Jul", "Aug", "Sep")

zmb_precip_summary <- here::here('data', 'climate', 'precipitation_monthly_mean.csv') %>%
  readr::read_csv() %>%
  mutate(month = factor(month, levels = custom_months))


precip_plot <- ggplot(data = zmb_precip_summary) +
  aes(x = month, y = precipitation) +
  labs(x = "Month", y = "Precipitation (mm)") + 
  geom_col(fill = "dodgerblue3", color = "black", width = 0.80) +
  scale_y_continuous(expand = expansion(add = c(0, 20)), breaks = seq(0, 275, 50)) + 
  theme_classic() +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        axis.text.x = element_text(colour = "black"),
        axis.text.y = element_text(colour = "black")) +
  ### SOWING
  geom_segment(x = "Oct", xend = "Dec", y = 80, yend = 80,
               lineend = 'round', linewidth = 2) +
  geom_label(x = "Dec",
             y = 80, label = "Sowing",
             hjust = 1.65, vjust = .5, size = 3) +
  ### GROWING
  geom_segment(x = "Nov", xend = "May", y = 65, yend = 65,
               lineend = 'round', linewidth = 2) +
  geom_label(x = "Feb",
             y = 65, label = "Growing",
             hjust = 1.1, vjust = .5, size = 3) +
  ### CFS 
  geom_segment(x = "Mar", xend = "Apr", y = 45, yend = 45, 
               lineend = 'round', linewidth = 2) +
  geom_label(x = "Apr",
             y = 45, label = "CFS ", 
             hjust = 1.05, vjust = .5, size = 3) +
  ### HARVEST  
  geom_segment(x = "Apr", xend = "Jun", y = 25, yend = 25, 
               lineend = 'round', linewidth = 2) +
  geom_label(x = "May",
             y = 25, label = "Harvest", 
             hjust = 0.5, vjust = 0.5, size = 3) +
  ### HARVEST  TO JULY
  geom_segment(x = "Jun", xend = "Aug", y = 25, yend = 25, 
             lineend = 'round', linewidth = 1, linetype="dotted") +
 
  ### FULL MONTH RANGE
  geom_segment(x = "Oct", xend = "Sep", y = 140, yend = 140,
               lineend = 'round', linewidth = 2, color = "grey50") +
  geom_text(aes(x = "Jul", y = 150, label = "Full month range"),
            angle = 0,
            size = 3,
            color = "black") +
  ### LIMITED MONTH RANGE
  geom_segment(x = "Apr", xend = "Sep", y = 110, yend = 110,
               lineend = 'round', linewidth = 2, color = "grey50") +
  geom_text(aes(x = "Jul", y = 120, label = "Limited month range"),
            angle = 0,
            size = 3,
            color = "black") 

  # ### WET
  # geom_segment(x = "Nov", xend = "Apr", y = 110, yend = 110,
  #              lineend = 'round', linewidth = 2, color = "darkblue") +
  # geom_label(x = "Feb",
  #            y = 110, label = "Rainy", 
  #            hjust = 1.05, vjust = .5, size = 3) +
  # ### DRY & HOT  
  # geom_segment(x = 10.5, xend = 12, y = 110, yend = 110, 
  #              lineend = 'round', linewidth = 2) +
  # geom_label(x= "Oct",
  #            y = 110, label = "Hot & Dry", 
  #            hjust = 1.05, vjust = .5, size = 3) +
  # ### COOL & DRY  
  # geom_segment(x = "May", xend = 10.5, y = 110, yend = 110, 
  #              lineend = 'round', linewidth = 2, color = "purple") +
  # geom_label(x = "Jul",
  #            y = 110, label = "Cool & Dry", 
  #            hjust = 1.05, vjust = .5, size = 3)

ggsave(
  filename = "figure_02.jpeg"
  , path = here("figures")
  , plot = precip_plot
  , device ="jpeg"
  , width = 6
  , height = 4
  , units = "in"
)

  