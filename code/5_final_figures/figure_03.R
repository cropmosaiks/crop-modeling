##########################################
####### FIGURE 3 - ZAMBIA CROPLAND #######
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  sf,
  terra,
  tidyterra,
  cowplot,
  quiet = T
)

country_shp <- here::here('data', 'geo_boundaries', 'gadm36_ZMB_2.shp') %>%
  sf::read_sf()

country_vect <- country_shp %>%
  sf::st_union() %>%
  terra::vect()

crop_land <- here::here("data", "land_cover", "ZMB_cropland_2019.tif") %>%
  terra::rast() %>%
  terra::crop(country_vect, mask=TRUE) %>% 
  terra::aggregate(fact=33, fun='mean', na.rm = TRUE)

crop_land <- crop_land * 100

ybreaks <- c(-8, -10, -12, -14, -16, -18)
ylabs <- paste0(ybreaks,'°S')

xbreaks <- c(22, 24, 26, 28, 30, 32, 34)
xlabs <- paste0(xbreaks,'°E')

ggplot() +
  tidyterra::geom_spatraster(data = crop_land) +
  geom_sf(data = country_shp, color = 'white', fill = NA, linewidth = .7) +
  scale_fill_viridis_c(na.value = NA, guide = guide_colorbar(title.position = "top")) +
  scale_x_continuous(breaks = xbreaks, labels = xlabs) +
  scale_y_continuous(breaks = ybreaks, labels = ylabs) +
  labs(fill = "Cropland Percentage") +
  theme_bw() +
  theme(legend.position = "bottom")


ggsave(
  filename = "figure_03.jpeg"
  , path = here("figures")
  , plot = ggplot2::last_plot()
  , device ="jpeg"
  , width = 8
  , height = 7
  , units = "in"
)
